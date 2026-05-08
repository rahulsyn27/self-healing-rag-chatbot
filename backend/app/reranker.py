from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from backend.app.hyde import retrieve_with_hyde
from backend.app.crag import filter_documents

# Initialize globally so the model only loads into memory once.
# This specific MS-MARCO model is small, fast, and optimized for QA relevance.
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

def rerank_documents(query: str, documents: list[Document], top_k: int = 3) -> list[Document]:
    """Scores query-document pairs and returns the highest precision matches."""
    
    if not documents:
        print("No documents to rerank.")
        return []

    print(f"\n--- RERANKING {len(documents)} DOCUMENTS ---")
    
    # The cross-encoder expects a list of pairs: [[query, doc1], [query, doc2], ...]
    pairs = [[query, doc.page_content] for doc in documents]
    
    # Predict relevance scores
    scores = reranker_model.predict(pairs)
    
    # Pair up the original documents with their new scores
    scored_docs = list(zip(documents, scores))
    
    # Sort descending (highest score first)
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Print the scoring for our terminal diagnostics
    for i, (doc, score) in enumerate(scored_docs):
        print(f"Rank {i+1} | Score: {score:.4f} | Preview: {doc.page_content[:60]}...")
        
    # Return strictly the top_k best documents
    best_docs = [doc for doc, score in scored_docs[:top_k]]
    
    return best_docs

# if __name__ == "__main__":
#     # Test the pipeline: HyDE -> CRAG -> Rerank
#     test_question = "What is the role of SGD in fedavg"
    
#     print("\n[STEP 1: HyDE Retrieval]")
#     raw_docs = retrieve_with_hyde(test_question, k=8)
    
#     print("\n[STEP 2: CRAG Grading]")
#     crag_results = filter_documents(test_question, raw_docs)
#     surviving_docs = crag_results["documents"]
    
#     print("\n[STEP 3: Cross-Encoder Reranking]")
#     final_docs = rerank_documents(test_question, surviving_docs, top_k=2)
    
#     print(f"\nFinal pipeline retained {len(final_docs)} perfectly matched documents for generation.")