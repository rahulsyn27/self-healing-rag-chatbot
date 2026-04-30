import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from backend.app.embeddings import get_embedding_function
from backend.app.config import VECTOR_DB_DIR

def ingest_documents(data_dir: str = "data/papers"):
    """Loads PDFs, splits them into semantic chunks, and stores them in ChromaDB."""
    
    # 1. Load PDFs from the directory
    print(f"Scanning for PDFs in {data_dir}...")
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    
    if not documents:
        print(f"No documents found in {data_dir}. Please add some PDFs!")
        return {"status": "error", "message": "No documents found."}
        
    print(f"Successfully loaded {len(documents)} document pages.")

    # 2. Split text into chunks
    # A 1000-character chunk with a 200-character overlap prevents cutting off 
    # crucial context or formulas across chunk boundaries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split text into {len(chunks)} chunks.")

    # 3. Create embeddings and persist to ChromaDB
    print(f"Embedding chunks and saving to {VECTOR_DB_DIR} (this might take a minute on CPU)...")
    
    # We create the Chroma DB locally. It automatically persists to disk.
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=get_embedding_function(), 
        persist_directory=VECTOR_DB_DIR
    )
    
    print("Ingestion complete! Your vector database is ready.")
    return {"status": "success", "total_pages": len(documents), "total_chunks": len(chunks)}

if __name__ == "__main__":
    # Ensure the data directory exists
    os.makedirs("data/papers", exist_ok=True)
    ingest_documents()
