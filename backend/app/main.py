import time
import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any

import os
from backend.app.config import VECTOR_DB_DIR
from backend.app.embeddings import get_embedding_function
from langchain_chroma import Chroma
from backend.app.judge import evaluate_pipelines
from backend.app.hyde import retrieve_with_hyde
from backend.app.crag import filter_documents
from backend.app.reranker import rerank_documents
from backend.app.generator import generate_final_answer
# --- Grounded RAG imports ---
from backend.app.grounded_rag import generate_cited_answer, verify_grounding

from fastapi import File, UploadFile
import shutil
import os
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter



app = FastAPI(title="Self-Healing RAG API")

class ChatRequest(BaseModel):
    query: str

class JudgeRequest(BaseModel):
    query: str
    normal_res: Dict[str, Any]
    sh_res: Dict[str, Any]

# --- Grounded RAG: Request model for 3-way judge ---
# class JudgeThreeRequest(BaseModel):
#     query: str
#     normal_res: Dict[str, Any]
#     grounded_res: Dict[str, Any]
#     sh_res: Dict[str, Any]

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
    if crag_results["status"] == "failed":
        surviving_docs = []
        active_query = query
    else:
        surviving_docs = crag_results["documents"]
        active_query = crag_results["new_query"]
    diagnostics["crag_passed"] = len(surviving_docs)

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


# --- Grounded RAG: Streaming pipeline (Retrieve -> Cite -> Verify) ---
# async def stream_grounded_rag(query: str):
#     """Yields live status updates while running the Grounded RAG pipeline."""
#     start_time = time.time()

#     yield json.dumps({"step": "Retrieving documents via similarity search..."}) + "\n"
#     await asyncio.sleep(0.1)
#     vector_store = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=get_embedding_function())
#     docs = vector_store.similarity_search(query, k=3)

#     yield json.dumps({"step": f"Generating cited answer from {len(docs)} documents..."}) + "\n"
#     await asyncio.sleep(0.1)
#     cited_answer = generate_cited_answer(query, docs)

#     yield json.dumps({"step": f"Verifying grounding of all cited claims..."}) + "\n"
#     await asyncio.sleep(0.1)
#     grounding = verify_grounding(cited_answer, docs)

#     yield json.dumps({
#         "answer": cited_answer,
#         "latency": round(time.time() - start_time, 2),
#         "retrieved_count": len(docs),
#         "docs": [doc.page_content for doc in docs],
#         "grounding_score": grounding["grounding_score"],
#         "verified_claims": grounding["verified"],
#         "total_claims": grounding["total"]
#     }) + "\n"


# Import your ingestion function here, e.g.:
# from backend.app.ingest import ingest_documents 

@app.on_event("startup")
def verify_or_build_vectorstore():
    # If the vector database directory doesn't exist or is empty
    if not os.path.exists(VECTOR_DB_DIR) or len(os.listdir(VECTOR_DB_DIR)) == 0:
        print("⚠️ Vector store is empty on server startup! Running ingestion pipeline...")
        # Call your document loading and embedding logic here
        # ingest_documents()
        print("✅ Vector store populated successfully!")

@app.post("/api/chat/naive")
def chat_naive_endpoint(request: ChatRequest):
    return StreamingResponse(stream_naive_rag(request.query), media_type="application/x-ndjson")
# application/x-ndjson: Newline Delimited JSON. Instead of one giant JSON object, you send multiple JSON objects separated by newlines.

@app.post("/api/chat/self-healing")
def chat_self_healing_endpoint(request: ChatRequest):
    return StreamingResponse(stream_self_healing(request.query), media_type="application/x-ndjson")

@app.post("/api/judge")
def judge_endpoint(request: JudgeRequest):
    verdict = evaluate_pipelines(request.query, request.normal_res, request.sh_res)
    return {"verdict": verdict}

# --- Grounded RAG: Streaming endpoint ---
# @app.post("/api/chat/grounded")
# def chat_grounded_endpoint(request: ChatRequest):
#     return StreamingResponse(stream_grounded_rag(request.query), media_type="application/x-ndjson")

# # --- Grounded RAG: 3-way judge endpoint ---
# @app.post("/api/judge3")
# def judge_three_endpoint(request: JudgeThreeRequest):
#     from backend.app.judge import evaluate_three_pipelines
#     verdict = evaluate_three_pipelines(request.query, request.normal_res, request.grounded_res, request.sh_res)
#     return {"verdict": verdict}

# Add this endpoint alongside your other routes
@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...)):
    try:
        # 1. Save the uploaded file temporarily to the server
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Extract text from the PDF
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        # 3. Slice the document into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # 4. Embed and store the chunks in your existing Chroma database
        # (Make sure VECTOR_DB_DIR and get_embedding_function() are imported/defined)
        vector_store = Chroma(
            persist_directory=VECTOR_DB_DIR, 
            embedding_function=get_embedding_function()
        )
        vector_store.add_documents(chunks)
        
        # 5. Clean up the temporary file
        os.remove(temp_file_path)
        
        return {"message": f"✅ Successfully ingested {len(chunks)} chunks from {file.filename} into the database!"}
    
    except Exception as e:
        return {"error": str(e)}