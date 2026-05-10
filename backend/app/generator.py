from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.app.config import GROQ_API_KEY

def generate_final_answer(query: str, documents: list) -> str:
    """Generates the final answer using ONLY the verified context."""
    
    if not documents:
        return "I couldn't find any highly relevant information in the database to answer your question."

    # Format the context cleanly
    context = "\n\n".join([f"--- Document {i+1} ---\n{doc.page_content}" for i, doc in enumerate(documents)])

    llm = ChatGroq(
        api_key=GROQ_API_KEY, 
        model_name="llama-3.3-70b-versatile", 
        # Low temperature to prevent hallucinations
        temperature=0.2 
    )

    system_prompt = """You are an expert AI research assistant. 
    Answer the user's question using ONLY the provided context from research papers. 
    If the context does not contain the answer, explicitly state that you do not know. 
    Do not hallucinate external information. 
    Where applicable, reference the specific document (e.g., 'According to Document 1...')."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    chain = prompt | llm | StrOutputParser()
    
    print("\n--- GENERATING FINAL ANSWER ---")
    return chain.invoke({"context": context, "question": query})