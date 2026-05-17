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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()
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
    print("Сервер запускается... Индекс будет загружен при первом запросе")
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
    
    embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu" 
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

def get_or_create_index():
    global search_index
    if search_index is None:
        print("Индекс не найден, загружаем...")
        docs = load_pages()
        search_index = create_search_engine(docs)
        print("Индекс успешно загружен!")
    return search_index

@app.get("/health")
async def health():
    try:
        index = get_or_create_index()
        return {"status": "healthy", "index_loaded": index is not None}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        index = get_or_create_index()
        if index is None:
            raise HTTPException(status_code=503, detail="Индекс не может загрузиться")
        
        answer, sources = search_on_site(request.question, index)
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)