from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.app.config import GROQ_API_KEY

def evaluate_pipelines(query: str, normal_res: dict, self_healing_res: dict) -> str:
    """Uses Groq as an impartial judge to compare Naive RAG vs Self-Healing RAG."""
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )
    
    system_prompt = """You are an expert AI judge evaluating two RAG architectures:
Pipeline A (Naive RAG) vs. Pipeline B (Self-Healing RAG).

Your evaluation criteria:
1. Accuracy & Context Quality: Pipeline B utilizes advanced closed-loop mechanisms (HyDE, CRAG, Cross-Encoders). For general, academic, or complex queries, you should generally favor Pipeline B, explicitly highlighting how its self-healing architecture prevents hallucinations and retrieves superior context.
2. Efficiency: If the user asks for extremely simple, personal, or basic factual data (e.g., 'What is Rahul's email?'), Pipeline B's advanced techniques are unnecessary. In these rare cases, allow Pipeline A to win due to speed.

Format your evaluation strictly as:
### Comparison Summary
<Explain why Pipeline B's architecture provided a more accurate and robust context, unless it was a simple factual lookup where A was sufficient>

### Winner
**[Pipeline A (Naive RAG) / Pipeline B (Self-Healing RAG)]**

### Rationale
<1-2 sentences justifying the win>"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """User Query: {query}

--- PIPELINE A (Naive RAG) ---
Latency: {normal_latency} seconds
Retrieved Chunks Count: {normal_doc_count}
Generated Answer: {normal_answer}

--- PIPELINE B (Self-Healing RAG) ---
Latency: {sh_latency} seconds
HyDE Retrieved: {sh_hyde_count} | CRAG Passed: {sh_crag_count} | Final Reranked: {sh_rerank_count}
Generated Answer: {sh_answer}

Provide your evaluation.""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "query": query,
        "normal_latency": normal_res["latency"],
        "normal_doc_count": normal_res["retrieved_count"],
        "normal_answer": normal_res["answer"],
        "sh_latency": self_healing_res["latency"],
        "sh_hyde_count": self_healing_res["diagnostics"].get("hyde_retrieved", 0),
        "sh_crag_count": self_healing_res["diagnostics"].get("crag_passed", 0),
        "sh_rerank_count": self_healing_res["diagnostics"].get("reranked_final", 0),
        "sh_answer": self_healing_res["answer"]
    })

# --- Grounded RAG: 3-way evaluation ---
# evaluate_pipelines above is kept as-is for backward compatibility with /api/judge
# This new function handles the 3-way A/B/C comparison
def evaluate_three_pipelines(query: str, normal_res: dict, grounded_res: dict, self_healing_res: dict) -> str:
    """Impartial LLM judge comparing Naive RAG vs Grounded RAG vs Self-Healing RAG."""
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0.1)

    system_prompt = """You are an expert, impartial AI judge evaluating three RAG architectures on the same query:

Pipeline A (Naive RAG):        Direct similarity search -> generate. No quality control.
Pipeline B (Grounded RAG):     Similarity search -> generate WITH inline citations -> verify each cited claim.
Pipeline C (Self-Healing RAG): HyDE -> CRAG relevance filter -> Cross-Encoder rerank -> generate.

Evaluate each pipeline on:
1. Answer Accuracy & Depth
2. Grounding / Hallucination Risk (use Pipeline B's grounding_score as a signal)
3. Retrieval Quality (use Pipeline C's CRAG/rerank diagnostics as signals)
4. Speed vs Quality Tradeoff

Be objective. Do not systematically favour any pipeline — a higher grounding_score is positive for B,
but a well-filtered reranked answer from C may still outperform if the question needs deep reasoning.

Format your evaluation strictly as:
### Pipeline Comparison

**Pipeline A (Naive RAG):** <1-2 sentences>
**Pipeline B (Grounded RAG):** <1-2 sentences, mention grounding score>
**Pipeline C (Self-Healing RAG):** <1-2 sentences, mention CRAG/rerank diagnostics>

### Winner
**[Pipeline A / Pipeline B / Pipeline C]**

### Rationale
<2-3 sentences justifying the winner with specific evidence from the results>"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """User Query: {query}

--- PIPELINE A (Naive RAG) ---
Latency: {a_latency}s | Retrieved Chunks: {a_doc_count}
Answer: {a_answer}

--- PIPELINE B (Grounded RAG) ---
Latency: {b_latency}s | Retrieved Chunks: {b_doc_count}
Grounding Score: {b_score} ({b_verified}/{b_total} claims verified)
Answer: {b_answer}

--- PIPELINE C (Self-Healing RAG) ---
Latency: {c_latency}s | HyDE: {c_hyde} docs -> CRAG passed: {c_crag} -> Final reranked: {c_rerank}
Answer: {c_answer}

Provide your evaluation.""")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "query": query,
        "a_latency": normal_res.get("latency", "N/A"),
        "a_doc_count": normal_res.get("retrieved_count", "N/A"),
        "a_answer": normal_res.get("answer", ""),
        "b_latency": grounded_res.get("latency", "N/A"),
        "b_doc_count": grounded_res.get("retrieved_count", "N/A"),
        "b_score": grounded_res.get("grounding_score", 0),
        "b_verified": grounded_res.get("verified_claims", 0),
        "b_total": grounded_res.get("total_claims", 0),
        "b_answer": grounded_res.get("answer", ""),
        "c_latency": self_healing_res.get("latency", "N/A"),
        "c_hyde": self_healing_res.get("diagnostics", {}).get("hyde_retrieved", 0),
        "c_crag": self_healing_res.get("diagnostics", {}).get("crag_passed", 0),
        "c_rerank": self_healing_res.get("diagnostics", {}).get("reranked_final", 0),
        "c_answer": self_healing_res.get("answer", "")
    })
