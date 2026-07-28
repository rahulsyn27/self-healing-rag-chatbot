import time
import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any

from backend.app.config import VECTOR_DB_DIR
from backend.app.embeddings import get_embedding_function
from langchain_chroma import Chroma
from backend.app.judge import evaluate_pipelines
from backend.app.hyde import retrieve_with_hyde
from backend.app.crag import filter_documents
from backend.app.reranker import rerank_documents
from backend.app.generator import generate_final_answer

app = FastAPI(title="Self-Healing RAG API")

class ChatRequest(BaseModel):
    query: str

class JudgeRequest(BaseModel):
    query: str
    normal_res: Dict[str, Any]
    sh_res: Dict[str, Any]

async def stream_naive_rag(query: str):
    """Yields live status updates while running Naive RAG."""
    start_time = time.time()
    
    yield json.dumps({"step": "Searching Vector Database directly..."}) + "\n"
    await asyncio.sleep(0.1) # Force network flush
    
    vector_store = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=get_embedding_function())
    retrieved_docs = vector_store.similarity_search(query, k=3)
    
    yield json.dumps({"step": "Synthesizing answer..."}) + "\n"
    await asyncio.sleep(0.1)
    
    answer = generate_final_answer(query, retrieved_docs)
    
    yield json.dumps({
        "answer": answer,
        "latency": round(time.time() - start_time, 2),
        "retrieved_count": len(retrieved_docs),
        "docs": [doc.page_content for doc in retrieved_docs]
    }) + "\n"

async def stream_self_healing(query: str):
    """Yields live status updates while traversing the Self-Healing pipeline."""
    start_time = time.time()
    diagnostics = {}

    yield json.dumps({"step": "Generating HyDE hypothetical document..."}) + "\n"
    await asyncio.sleep(0.1)
    raw_docs = retrieve_with_hyde(query, k=10)
    diagnostics["hyde_retrieved"] = len(raw_docs)

    yield json.dumps({"step": f"Running CRAG relevance grading on {len(raw_docs)} docs..."}) + "\n"
    await asyncio.sleep(0.1)
    crag_results = filter_documents(query, raw_docs)
    surviving_docs = crag_results["documents"]
    diagnostics["crag_passed"] = len(surviving_docs)
    active_query = crag_results["new_query"]

    yield json.dumps({"step": f"Cross-Encoder reranking {len(surviving_docs)} docs..."}) + "\n"
    await asyncio.sleep(0.1)
    final_docs = rerank_documents(active_query, surviving_docs, top_k=3) if surviving_docs else []
    diagnostics["reranked_final"] = len(final_docs)

    yield json.dumps({"step": "Synthesizing final answer with verified context..."}) + "\n"
    await asyncio.sleep(0.1)
    answer = generate_final_answer(active_query, final_docs)
    
    yield json.dumps({
        "answer": answer,
        "diagnostics": diagnostics,
        "latency": round(time.time() - start_time, 2)
    }) + "\n"

@app.post("/api/chat/naive")
def chat_naive_endpoint(request: ChatRequest):
    return StreamingResponse(stream_naive_rag(request.query), media_type="application/x-ndjson")

@app.post("/api/chat/self-healing")
def chat_self_healing_endpoint(request: ChatRequest):
    return StreamingResponse(stream_self_healing(request.query), media_type="application/x-ndjson")

@app.post("/api/judge")
def judge_endpoint(request: JudgeRequest):
    verdict = evaluate_pipelines(request.query, request.normal_res, request.sh_res)
    return {"verdict": verdict}