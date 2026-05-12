from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

# Import our pipeline modules
from backend.app.embeddings import get_embedding_function
from backend.app.hyde import retrieve_with_hyde
from backend.app.crag import filter_documents
from backend.app.reranker import rerank_documents
from backend.app.generator import generate_final_answer

app = FastAPI(
    title="Self-Healing RAG API",
    description="Production-grade Self-Healing RAG backend using LangChain & Groq"
)

# --- Define Request and Response Schemas ---
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    diagnostics: Dict[str, Any]

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Self-Healing RAG API is running"}

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Executes the full Self-Healing RAG Pipeline."""
    
    query = request.query
    diagnostics = {}

    print(f"\n========== NEW REQUEST: {query} ==========")

    # STEP 1: HyDE Retrieval
    raw_docs = retrieve_with_hyde(query, k=10)
    diagnostics["hyde_retrieved"] = len(raw_docs)

    # STEP 2: CRAG Document Grading & Query Rewriting
    crag_results = filter_documents(query, raw_docs)
    surviving_docs = crag_results["documents"]
    
    diagnostics["crag_passed"] = len(surviving_docs)
    diagnostics["crag_status"] = crag_results["status"]
    
    # Use the rewritten query if CRAG determined the original was bad
    active_query = crag_results["new_query"] 
    diagnostics["active_query"] = active_query

    # STEP 3: Cross-Encoder Reranking
    final_docs = []
    if surviving_docs:
        final_docs = rerank_documents(active_query, surviving_docs, top_k=3)
    diagnostics["reranked_final"] = len(final_docs)

    # STEP 4: Final Generation
    answer = generate_final_answer(active_query, final_docs)

    print("========== PIPELINE COMPLETE ==========\n")

    return ChatResponse(
        answer=answer,
        diagnostics=diagnostics
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)