from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from backend.app.config import GROQ_API_KEY
from backend.app.hyde import retrieve_with_hyde  # Using your new file name!

# --- 1. Define the Structured Output Schema ---
# class GradeDocuments(BaseModel):
#     """Binary score for relevance check on retrieved documents."""
#     binary_score: str = Field(
#         description="Documents are relevant to the question, 'yes' or 'no'"
#     )

# --- 2. Initialize the Grader ---
def get_document_grader():
    """Creates a grading chain that strictly outputs 'yes' or 'no' as plain text."""
    llm = ChatGroq(
        api_key=GROQ_API_KEY, 
        model_name="llama-3.3-70b-versatile", 
        temperature=0
    )
    
    # We remove structured output and demand a single word.
    system_prompt = """You are a strict grading assistant assessing the relevance of a retrieved document to a user question.
    If the document contains keywords, concepts, or semantic meaning related to the question, grade it as 'yes'.
    If it is completely unrelated, grade it as 'no'.
    You must output ONLY the word 'yes' or 'no'. Do not include any punctuation, formatting, or reasoning."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ])
    
    # StrOutputParser guarantees we just get the raw string back
    return grade_prompt | llm | StrOutputParser()


# --- 3. Initialize the Query Rewriter ---
def rewrite_query(question: str) -> str:
    """Rewrites a poorly worded query to optimize it for vector retrieval."""
    llm = ChatGroq(
        api_key=GROQ_API_KEY, 
        model_name="llama-3.3-70b-versatile", 
        temperature=0.1
    )
    
    system_prompt = """You are an expert AI query rewritter. 
    Look at the input question and reason about the underlying semantic intent. 
    Formulate a better, highly specific version of the question optimized for searching academic ML research papers.
    Return ONLY the rewritten question, no intro or filler text."""
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Here is the initial question: \n\n {question} \n\n Formulate an improved question.")
    ])
    
    rewriter_chain = rewrite_prompt | llm | StrOutputParser()
    return rewriter_chain.invoke({"question": question})

# --- 4. The CRAG Execution Logic ---
def filter_documents(question: str, documents: list) -> dict:
    """Grades all documents. If all fail, rewrites the query."""
    grader = get_document_grader()
    
    relevant_docs = []
    print(f"\n--- CRAG GRADING {len(documents)} DOCUMENTS ---")
    
    for i, doc in enumerate(documents):
        # Grade the document and clean the string output
        raw_score = grader.invoke({"question": question, "document": doc.page_content})
        grade = raw_score.strip().lower()
        
        # Add a safety check in case the LLM hallucinates extra text
        if "yes" in grade:
            print(f"Document {i+1}: RELEVANT ✅")
            relevant_docs.append(doc)
        else:
            print(f"Document {i+1}: IRRELEVANT ❌ (Discarded)")
            
    if not relevant_docs:
        print("\n--- ALL DOCUMENTS IRRELEVANT. REWRITING QUERY ---")
        better_query = rewrite_query(question)
        print(f"Old Query: {question}")
        print(f"New Query: {better_query}")
        
        return {"status": "rewritten", "new_query": better_query, "documents": []}
        
    return {"status": "success", "new_query": question, "documents": relevant_docs}

    
# if __name__ == "__main__":
#     # Let's test the full flow: HyDE -> CRAG
#     test_question = "what is difference between fedavg and fedprox"
    
#     # 1. Get raw documents from your HyDE script
#     raw_docs = retrieve_with_hyde(test_question, k=5)
    
#     # 2. Filter them through CRAG
#     crag_results = filter_documents(test_question, raw_docs)
    
#     print(f"\nFinal relevant documents kept: {len(crag_results['documents'])}")