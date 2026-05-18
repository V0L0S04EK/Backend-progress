import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from contextlib import asynccontextmanager
import concurrent.futures

from llama_index.core import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from dotenv import load_dotenv
import nest_asyncio

nest_asyncio.apply()
load_dotenv()

HF_API_KEY = os.getenv('API_KEY')
if not HF_API_KEY:
    raise ValueError("API_KEY не найден в .env файле!")

search_index = None
index_loading_task = None

class QueryRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = []

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = []

def load_pages():
    """Синхронная загрузка страниц"""
    urls_to_scrape = [
        "https://xn--c1aezdfcia.fun/data/ai-construction-part1.html",
        "https://xn--c1aezdfcia.fun/data/ai-construction-part2.html"
    ]
    
    reader = SimpleWebPageReader(html_to_text=True)
    documents = reader.load_data(urls_to_scrape)
    print(f"Загружено {len(documents)} документов")
    return documents

def create_search_engine(documents):
    """Синхронное создание индекса"""
    db = chromadb.PersistentClient(path="./progress_fun_db")
    chroma_collection = db.get_or_create_collection("knowledge_base")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    embed_model = HuggingFaceInferenceAPIEmbedding(
        api_key=HF_API_KEY,
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=50),
            embed_model,
        ],
        vector_store=vector_store
    )
    
    nodes = pipeline.run(documents=documents)
    print(f"Проиндексировано {len(nodes)} фрагментов")
    
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        embed_model=embed_model
    )
    return index

def search_on_site(query: str, index):
    llm = HuggingFaceInferenceAPI(
        model_name="Qwen/Qwen2.5-72B-Instruct", 
        token=HF_API_KEY,
        temperature=0.3
    )
    
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,  
        response_mode="tree_summarize"
    )
    
    from llama_index.core import PromptTemplate
    
    custom_prompt = PromptTemplate(
        "Ты — эксперт по саморазвитию и продуктивности. У тебя есть база знаний сайта progress.fun.\n"
        "Отвечай на русском языке, используя только информацию из контекста ниже.\n"
        "Отвечай только фрагментами текста из приведенного контекста.\n"
        "Если информация отсутствует, скажи честно, что не нашёл.\n"
        "Контекст: {context_str}\n"
        "Вопрос: {query_str}\n"
        "Ответ:"
    )
    query_engine.update_prompts({"response_synthesizer:text_qa_template": custom_prompt})
    
    response = query_engine.query(query)
    
    sources = []
    if hasattr(response, 'source_nodes'):
        sources = [node.node.text[:200] + "..." for node in response.source_nodes]
    
    return response.response, sources

async def load_index_background():
    """Фоновая загрузка индекса в отдельном потоке"""
    global search_index
    
    loop = asyncio.get_event_loop()
    
    try:
        print("📚 Загрузка документов...")
        docs = await loop.run_in_executor(None, load_pages)
        
        print("🔨 Создание индекса...")
        search_index = await loop.run_in_executor(None, create_search_engine, docs)
        
        print("✅ Индекс успешно загружен!")
    except Exception as e:
        print(f"❌ Ошибка загрузки индекса: {e}")
        import traceback
        traceback.print_exc()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index_loading_task
    print("✅ Сервер запускается, начинаю фоновую загрузку индекса...")
    index_loading_task = asyncio.create_task(load_index_background())
    
    yield 
    
    # Shutdown
    print("Сервер останавливается...")
    if index_loading_task:
        index_loading_task.cancel()

app = FastAPI(lifespan=lifespan)

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "RAG API", "status": "running"}

@app.get("/health")
async def health():
    """Health check endpoint для Render"""
    if search_index is None:
        return {"status": "loading", "message": "Индекс загружается, попробуйте через минуту"}
    return {"status": "healthy", "index_loaded": True}

@app.get("/ready")
async def ready():
    """Ready probe для Render"""
    if search_index is not None:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Индекс еще не загружен")

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if search_index is None:
        raise HTTPException(status_code=503, detail="Сервер загружается, попробуйте через минуту")
    
    try:
        answer, sources = search_on_site(request.question, search_index)
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)