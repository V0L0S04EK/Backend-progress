import os
from llama_index.core import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

HF_API_KEY = "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"  

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-72B-Instruct", 
    token=HF_API_KEY,
    temperature=0.3
)

embed_model = HuggingFaceInferenceAPIEmbedding(
    model_name="BAAI/bge-small-en-v1.5",  
    token=HF_API_KEY
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
    
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=50),
            embed_model,
        ],
        vector_store=vector_store
    )
    
    nodes = pipeline.run(documents=documents)
    print(f" Проиндексировано {len(nodes)} фрагментов")
    
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        embed_model=embed_model
    )
    return index

def search_on_site(query: str, index):
    
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
    return response


if __name__ == "__main__":
    print("Загрузка данных с сайта...")
    docs = load_pages()
    
    print("Создание поискового индекса...")
    index = create_search_engine(docs)
    
    while True:
        user_query = input("\n Ваш вопрос про саморазвитие (или 'выход'): ")
        if user_query.lower() == 'выход':
            break
        
        print(" Ищу ответ...")
        answer = search_on_site(user_query, index)
        print(f"\n Ответ:\n{answer}\n")