from langchain_huggingface import HuggingFaceEmbeddings
from backend.app.config import EMBEDDING_MODEL_NAME

def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFace BAAI/bge-small-en-v1.5 embedding model.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings
