# Grounded RAG Pipeline
# Approach: Output-side verification (DOWNSTREAM quality control)
#
# Contrast with other pipelines:
#   Naive RAG        -> No quality control anywhere
#   Grounded RAG     -> Verifies the GENERATED ANSWER (downstream)
#   Self-Healing RAG -> Verifies RETRIEVED DOCUMENTS (upstream via HyDE+CRAG+Rerank)
#
# Steps: Retrieve -> Generate with inline [DOC N] citations -> Verify each claim -> Score

import time
import json
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.app.embeddings import get_embedding_function
from backend.app.config import GROQ_API_KEY, VECTOR_DB_DIR


# --- Step 1: Standard Retrieval (same as Naive RAG, intentionally) ---
# Plain similarity search so quality gain comes from grounding, not better retrieval
def retrieve_documents(query: str, k: int = 3):
    """Standard cosine similarity search against Chroma vector store."""
    vector_store = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=get_embedding_function())
    return vector_store.similarity_search(query, k=k)


# --- Step 2: Generate Answer with Inline Citations ---
# Forces LLM to end EVERY factual claim with [DOC N] so we know what to verify
def generate_cited_answer(query: str, documents: list) -> str:
    """Generates an answer where every factual claim ends with a [DOC N] citation tag."""
    if not documents:
        return "I couldn't find relevant information to answer your question."

    context = "\n\n".join([f"[DOC {i+1}]:\n{doc.page_content}" for i, doc in enumerate(documents)])

    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0.1)

    system_prompt = """You are a research assistant. Answer the question in 3-5 clean sentences.
Rules:
- Every factual claim MUST end with a citation tag like [DOC 1], [DOC 2], or [DOC 3].
- Write in your OWN words — do NOT copy raw text from documents.
- Only include information traceable to the provided documents.
- Maximum 5 cited claims in total. Be concise.
Example: "FedAvg aggregates model weights on a central server [DOC 1]. This reduces communication cost [DOC 2]."
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Documents:\n{context}\n\nQuestion: {query}")
    ])
    return (prompt | llm | StrOutputParser()).invoke({"context": context, "query": query})


# --- Step 3: Grounding Verification ---
# Makes (1 + N) LLM calls: 1 to extract claims, N to verify each one
# grounding_score = verified_claims / total_claims
def verify_grounding(answer: str, documents: list) -> dict:
    """Verifies every cited claim in the answer against its referenced source document."""
    if not documents:
        return {"grounding_score": 0.0, "verified": 0, "total": 0, "details": []}

    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0)

    # Build lookup: "DOC 1" -> page_content
    doc_lookup = {f"DOC {i+1}": doc.page_content for i, doc in enumerate(documents)}

    # Substep 3a: Extract cited claims as structured JSON from the answer
    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract all cited claims from the answer text.
Return ONLY a valid JSON array. No markdown, no code blocks, no preamble.
Format: [{{"claim": "<factual claim>", "doc_ref": "DOC 1"}}, ...]
Extract at most 8 claims. If no citations found, return: []"""),
        ("human", "Answer text:\n{answer}")
    ])
    raw_extracted = (extract_prompt | llm | StrOutputParser()).invoke({"answer": answer})

    try:
        # Strip any accidental markdown fences the model might add
        clean = raw_extracted.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        claims = json.loads(clean)
    except Exception:
        return {"grounding_score": 0.0, "verified": 0, "total": 0, "details": []}

    if not claims:
        return {"grounding_score": 0.0, "verified": 0, "total": 0, "details": []}

    # Substep 3b: Verify each claim against its cited source document
    verify_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict fact-checker.
Does the source document support the given claim?
Answer ONLY with the word 'yes' or 'no'. Nothing else."""),
        ("human", "Claim: {claim}\n\nSource Document:\n{document}")
    ])
    verify_chain = verify_prompt | llm | StrOutputParser()

    details = []
    verified_count = 0

    for item in claims:
        claim_text = item.get("claim", "").strip()
        doc_ref = item.get("doc_ref", "").strip()
        doc_text = doc_lookup.get(doc_ref, "")

        if not doc_text:
            # Claim references a document that does not exist (hallucinated citation)
            details.append({"claim": claim_text, "doc_ref": doc_ref, "supported": False, "reason": "invalid_ref"})
            continue

        result = verify_chain.invoke({"claim": claim_text, "document": doc_text})
        supported = "yes" in result.strip().lower()
        if supported:
            verified_count += 1
        details.append({"claim": claim_text, "doc_ref": doc_ref, "supported": supported})

    total = len(claims)
    score = round(verified_count / total, 2) if total > 0 else 0.0
    return {"grounding_score": score, "verified": verified_count, "total": total, "details": details}


# --- Full synchronous runner (for offline testing; streaming version is in main.py) ---
def run_grounded_rag(query: str, k: int = 3) -> dict:
    start_time = time.time()
    docs = retrieve_documents(query, k=k)
    cited_answer = generate_cited_answer(query, docs)
    grounding = verify_grounding(cited_answer, docs)
    return {
        "answer": cited_answer,
        "retrieved_count": len(docs),
        "docs": [doc.page_content for doc in docs],
        "grounding_score": grounding["grounding_score"],
        "verified_claims": grounding["verified"],
        "total_claims": grounding["total"],
        "grounding_details": grounding["details"],
        "latency": round(time.time() - start_time, 2)
    }
