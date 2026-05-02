from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from backend.app.embeddings import get_embedding_function
from backend.app.config import GROQ_API_KEY, VECTOR_DB_DIR

def generate_hypothetical_document(query: str) -> str:
    """Uses Groq to generate a hypothetical ArXiv paper extract answering the query."""
    
    # We use Llama 3 70B via Groq for blazing fast generation
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )
    
    # Prompting Groq to sound like a research paper
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI researcher. Write a concise, academic paragraph that hypothetically answers the user's question. Use formal language and machine learning terminology. Do not include introductory filler like 'Here is the answer'."),
        ("human", "{query}")
    ])
    
    chain = prompt | llm | StrOutputParser()  # string output parser is used to parse the output of the chain into a string.
    # In LangChain, .invoke() is the command that actually "pulls the trigger." You spent the previous lines building the pipeline (chain = prompt | llm | parser), but the pipeline just sits there doing nothing until you call .invoke()
    return chain.invoke({"query": query})  # the second query (without quotes) is the actual user statement. LangChain sees the variable query and unpacks it: {"query": "How does PCA work?"}.

def retrieve_with_hyde(query: str, k: int = 5):
    """Generates a hypothetical document and uses it to retrieve real documents from Chroma.
    The k most mathematically similar chunks are selected, and the rest chunks are discarded."""
    
    # 1. Generate the hypothetical document
    print(f"--- Original Query ---\n{query}\n")
    hypothetical_doc = generate_hypothetical_document(query)
    print(f"--- HyDE Generated Document (Fake) ---\n{hypothetical_doc}\n")
    
    # 2. Connect to the local ChromaDB
    vector_store = Chroma(
        persist_directory=VECTOR_DB_DIR, 
        embedding_function=get_embedding_function()
    )
    
    # 3. Retrieve using the hypothetical document, NOT the short query
    print("--- Searching Vector DB using the HyDE Document ---")
    results = vector_store.similarity_search(hypothetical_doc, k=k)
    
    # for result in results:
    #     print(result.page_content[:100])
    
    return results

# if __name__ == "__main__":
#     # Test the HyDE retrieval with a machine learning question
#     test_query = "What is the role of attention mechanisms in neural networks?"
    
#     retrieved_docs = retrieve_with_hyde(test_query)
    
#     print(f"\n[Retrieved {len(retrieved_docs)} real documents from ArXiv papers]")
#     for i, doc in enumerate(retrieved_docs):
#         print(f"\n--- Real Document {i+1} ---")
#         # Print the first 300 characters of the matched chunk
#         print(f"{doc.page_content[:300]}...\n")