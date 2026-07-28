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
    
    system_prompt = """You are an impartial expert AI judge evaluating two RAG architectures:
Pipeline A (Naive / Normal RAG) vs. Pipeline B (Self-Healing RAG).

Evaluate based on:
1. Faithfulness & Groundedness: Is the answer strictly derived from context without hallucination?
2. Context Quality: Which pipeline retrieved higher quality, relevant chunks?
3. Efficiency vs Over-Engineering: If both yield the exact same correct answer, penalize Pipeline B for unnecessary complexity/latency. For simple facts, direct entity lookups, or resumes, Naive RAG is preferred if equal in accuracy.

Format your evaluation strictly as:
### Comparison Summary
<2-3 sentences evaluating context quality, answer accuracy, and speed trade-offs>

### Winner
**[Pipeline A (Naive RAG) / Pipeline B (Self-Healing RAG) / Tie]**

### Rationale
<1-2 sentences explaining why this winner was selected for this specific query>"""

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