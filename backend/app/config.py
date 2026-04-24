import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./chroma_db")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in environment variables.")
