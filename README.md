# Self-Healing RAG Chatbot

A production-style Retrieval-Augmented Generation (RAG) system with a closed-loop "self-healing" pipeline. Built with LangChain, Groq, ChromaDB, and a Streamlit frontend.

## Pipeline

1. **HyDE** — Generates a hypothetical research excerpt to improve vector retrieval
2. **CRAG** — Grades document relevance; rewrites the query if all docs fail
3. **Cross-Encoder Reranking** — Precision reranking with MS-MARCO MiniLM
4. **Generation** — Llama 3 70B (via Groq) synthesizes answers from verified context

## Setup

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env   # add your GROQ_API_KEY

# Ingest papers into ChromaDB
python -m backend.app.ingestion

# Start API
uvicorn backend.app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
backend/app/   FastAPI pipeline modules
frontend/      Streamlit UI
data/papers/   ArXiv PDFs for ingestion
```
