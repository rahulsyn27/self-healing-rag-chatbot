import time
from langchain_chroma import Chroma
from backend.app.embeddings import get_embedding_function
from backend.app.config import VECTOR_DB_DIR
from backend.app.generator import generate_final_answer

def run_normal_rag(query: str, k: int = 3) -> dict:
    """Executes a standard, open-loop naive RAG pipeline without HyDE, CRAG, or reranking."""
    start_time = time.time()
    
    # 1. Direct vector similarity search
    vector_store = Chroma(
        persist_directory=VECTOR_DB_DIR, 
        embedding_function=get_embedding_function()
    )
    retrieved_docs = vector_store.similarity_search(query, k=k)
    
    # 2. Direct generation
    answer = generate_final_answer(query, retrieved_docs)
    latency = round(time.time() - start_time, 2)
    
    return {
        "answer": answer,
        "retrieved_count": len(retrieved_docs),
        "docs": [doc.page_content for doc in retrieved_docs],
        "latency": latency
    }