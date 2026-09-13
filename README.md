# Self-Healing RAG Chatbot

A Retrieval-Augmented Generation (RAG) system with a closed-loop "self-healing" pipeline. Built with LangChain, Groq, ChromaDB, and a Streamlit frontend.

---

## Architecture Overview

```mermaid
flowchart TB
    Q["User Query"]
    H["HyDE<br/>Generate Hypothetical Document"]
    R["Vector Retrieval<br/>Retrieve Top-k Documents"]
    DB[("Vector Database")]
    C["CRAG<br/>Relevance Grader"]
    D{"Are Retrieved Documents<br/>Relevant?"}
    RR["Cross-Encoder<br/>Reranker"]
    RW["Query Rewriter"]
    G["Final Context<br/>Generation"]

    %% Main top-down flow
    Q --> H
    H --> R
    R --> DB
    DB --> R
    R --> C
    C --> D

    %% CRAG decision
    D -->|"Yes"| RR
    RR --> G

    D -->|"No"| RW
    RW --> H

    %% Styling
    classDef query fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef hyde fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef retrieval fill:#f5f5f5,stroke:#616161,stroke-width:2px;
    classDef crag fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef decision fill:#fff8e1,stroke:#f57f17,stroke-width:2px;
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px;

    class Q query;
    class H hyde;
    class R,DB retrieval;
    class C,RW,RR crag;
    class D decision;
    class G output;
```
---

## Pipeline Architecture Details

### 1. Naive RAG (Baseline)
**Purpose:** Direct vector similarity search without quality control.

```
Query → Vector Search (k=3) → Generator → Answer
```

- **Vector Search:** ChromaDB similarity search using BAAI/bge-small-en-v1.5 embeddings
- **Generator:** Llama 3.3 70B (via Groq) with temperature 0.2
- **Use Case:** Simple factual lookups where speed matters more than precision

---

### 2. Self-Healing RAG (Primary Pipeline)
**Purpose:** Closed-loop retrieval with iterative quality control.

```
Query
  │
  ▼
┌─────────────────────────────────────┐
│  HyDE: Hypothetical Document Gen    │  ← Llama 3.3 70B generates academic-style
│  (Expands short query to full doc)  │     hypothetical paper excerpt
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│  HyDE Vector Search (k=10)          │  ← Search with dense hypothetical doc
└─────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│  CRAG: Corrective RAG Grader        │  ← Binary relevance grading (yes/no)
│  • Grades each retrieved doc        │     Strict: only 'yes' or 'no' output
│  • Filters irrelevant docs          │
└─────────────────────────────────────┘
  │
  ├─► If ANY relevant → Cross-Encoder Rerank
  │
  └─► If ALL failed → Query Rewriter (max 2 retries)
        │
        ▼
    Rewrite Query → HyDE Retry (recursive)
```

**CRAG Logic:**
- Grades each document independently
- If ≥1 document passes → proceed with filtered set
- If 0 documents pass → rewrite query semantically for academic search → retry (max 2 attempts)

**Cross-Encoder Reranking:**
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Scores query-document pairs directly (not independent embeddings)
- Returns top-k (default 3) highest-precision matches

**Final Generation:**
- Llama 3.3 70B with strict grounding instructions
- Temperature 0.2 to minimize hallucination
- References specific documents by number

---

### 3. LLM Judge (Evaluation)
**Purpose:** Impartial comparison of pipeline outputs.

**2-Way Judge (Naive vs Self-Healing):**
- Favors Self-Healing for complex/academic queries
- Favors Naive for simple factual lookups (speed)

---


### Request/Response Formats

**Chat Request:**
```json
{"query": "What is the difference between FedAvg and FedProx?"}
```

**Streaming Response (NDJSON - one JSON per line):**
```json
{"step": "Generating HyDE hypothetical document..."}
{"step": "Running CRAG relevance grading on 10 docs..."}
{"step": "Cross-Encoder reranking 7 docs..."}
{"step": "Synthesizing final answer with verified context..."}
{"answer": "FedAvg...", "diagnostics": {"hyde_retrieved": 10, "crag_passed": 7, "reranked_final": 3}, "latency": 4.2}
```

---

## Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  PDF Files  │────▶│  Ingestion   │────▶│  ChromaDB   │◀───▶│  All Pipelines│
│  (data/     │     │  • PyPDF     │     │  • Embed:   │     │  • Query      │
│   papers/)  │     │  • Chunk     │     │    bge-small│     │  • Retrieve   │
└─────────────┘     │  • Embed     │     │  • Persist  │     └──────────────┘
                    └──────────────┘     └─────────────┘
                           │
                    Chunk: 1000 chars
                    Overlap: 200 chars
                    Separators: \n\n, \n, ., " ", ""
```

---

## Project Structure

```
Self-Healing RAG chatbot/
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── config.py           # Environment config (API keys, paths, model names)
│       ├── embeddings.py       # BAAI/bge-small-en-v1.5 embedding function
│       ├── ingestion.py        # PDF loading, chunking, ChromaDB persistence
│       ├── hyde.py             # HyDE hypothetical document generation + retrieval
│       ├── crag.py             # CRAG grading, query rewriting, recursive retry
│       ├── reranker.py         # Cross-Encoder MS-MARCO reranking
│       ├── generator.py        # Final answer synthesis with citations
│       ├── judge.py            # LLM-as-judge evaluation (2-way & 3-way)
│       └── main.py             # FastAPI endpoints, streaming orchestration
├── frontend/
│   ├── requirements.txt
│   └── app.py                  # Streamlit UI with A/B/C comparison tabs
├── data/
│   └── papers/                 # Source PDFs for ingestion
├── chroma_db/                  # Persistent vector database
├── .env                        # GROQ_API_KEY, VECTOR_DB_DIR
├── .env.example
├── render.yaml                 # Render.com deployment config
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Groq API Key (free at https://console.groq.com)

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp ../.env.example ../.env
# Edit .env and add your GROQ_API_KEY

# Ingest documents (run once)
python -m backend.app.ingestion

# Start API server
uvicorn backend.app.main:app --reload --port 8000
```

### Frontend (separate terminal)

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### Access
- Frontend: http://localhost:8501
- API Docs: http://localhost:8000/docs

---

## Key Features

| Feature | Description |
|---------|-------------|
| **HyDE** | Expands short queries into hypothetical documents for better semantic matching |
| **CRAG** | Binary relevance grading with automatic query rewriting on total failure (max 2 retries) |
| **Cross-Encoder Reranking** | Precision scoring of query-doc pairs using MS-MARCO MiniLM |
| **LLM Judge** | Impartial multi-pipeline evaluation with structured rationale |
| **Streaming UI** | Live step-by-step telemetry for all pipelines simultaneously |
| **PDF Upload** | Dynamic ingestion of custom documents at runtime |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | - | Groq API key for Llama 3.3 70B |
| `VECTOR_DB_DIR` | No | `./chroma_db` | ChromaDB persistence directory |
| `EMBEDDING_MODEL_NAME` | No | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |

---

## License

MIT License - Feel free to use and modify for research or production.
