import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from contextlib import asynccontextmanager

from llama_index.core import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv('API_KEY')
if not HF_API_KEY:
    raise ValueError("API_KEY не найден в .env файле!")

search_index = None

class QueryRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = []

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_index
    print("Загрузка данных с сайта...")
    docs = load_pages()
    print("Создание поискового индекса...")
    search_index = create_search_engine(docs)
    print("Сервер готов к работе!")
    yield
    print("Сервер останавливается...")

app = FastAPI(lifespan=lifespan)
origins = ["https://xn--c1aezdfcia.fun",
            "https://xn--c1aezdfcia.fun/data/ai-construction-part1.html",
            "https://xn--c1aezdfcia.fun/data/ai-construction-part2.html"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_pages():
    urls_to_scrape = [
        "https://xn--c1aezdfcia.fun/data/ai-construction-part1.html",
        "https://xn--c1aezdfcia.fun/data/ai-construction-part2.html"
    ]
    
    reader = SimpleWebPageReader(html_to_text=True)
    documents = reader.load_data(urls_to_scrape)
    print(f"Загружено {len(documents)} документов")
    return documents

def create_search_engine(documents):
    db = chromadb.PersistentClient(path="./progress_fun_db")
    chroma_collection = db.get_or_create_collection("knowledge_base")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    embed_model = HuggingFaceInferenceAPIEmbedding(
        model_name="BAAI/bge-small-en-v1.5",  
        token=HF_API_KEY
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

@app.get("/")
async def root():
    return {"message": "RAG API", "status": "running"}

@app.get("/health")
async def health():
    if search_index is None:
        raise HTTPException(status_code=503, detail="Индекс еще не загружен")
    return {"status": "healthy", "index_loaded": True}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if search_index is None:
        raise HTTPException(status_code=503, detail="Сервер загружается, попробуйте через минуту")
    
    try:
        answer, sources = search_on_site(request.question, search_index)
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)