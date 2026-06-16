# RAG Journey 🧠

Building production-grade Retrieval-Augmented Generation (RAG) systems from core basics to advanced architectures using LangChain, FastAPI, and Docker.

## What is RAG?
Instead of relying only on static training data, RAG enables Large Language Models to dynamically **query external knowledge bases** and generate factually grounded responses.
`Document` → `Chunks` → `Embeddings` → `Vectorstore` → `Retriever` → `LLM` → `Answer`


## Project Structure
```bash
RAG-Journey/
├── 01_basics/
│   ├── loader.py       # Document loading (PDF, TXT, CSV)
│   ├── splitter.py     # Optimized text splitting strategies
│   ├── embeddings.py   # Vector embeddings & FAISS index storage
│   └── retriever.py    # Basic LCEL RAG chain orchestration
└── 02_advanced/        
│   ├── 01_reranking.ipynb             # Context optimization via CrossEncoder reranking
│   ├── 02_hybrid_search.ipynb         # Lexical (BM25) + Semantic (FAISS) via RRF
│   ├── 03_query_transformation.ipynb  # Expansion patterns (Multi-Query & HyDE)
│   ├── 04_evaluation.ipynb            # Rigorous pipeline auditing with RAGAS
│   ├── 05_parent_child.ipynb          # Decoupled Retrieval (Parent-Document Retriever)
│   ├── 06_metadata_filtering.ipynb    # Structured Metadata Filtering via Custom LCEL Schema
│   └── 07_agentic_rag.ipynb           # Coming soon
└── 03_deployment/
│   ├── app/
│   │   └── main.py      # Production FastAPI application with hot-reloading
│   ├── ingestion/
│   │   └── ingest.py    # Standalone script to load source documents and compile FAISS index
│   ├── data/            # Local directory holding the mounted FAISS index binaries
│   ├── docker-compose.yml# Multi-container orchestration layer
│   ├── Dockerfile       # Optimized, lightweight Python environment
└── └── requirements.txt # Deployment-specific service requirements
```
## Tech Stack
- **Orchestration:** LangChain (LCEL Pattern)
- **API Layer:** FastAPI + Uvicorn
- **Containerization:** Docker & Docker Compose
- **Inference Engine:** Groq API (`llama-3.1-8b-instant`)
- **Embedding Model:** HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Database:** FAISS (Local Binary Storage)
- **Lexical Search:** BM25
- **Reranking:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
- **Evaluation:** RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
- **Data Validation:** Pydantic v2

## Setup & Local Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/MrJouH4/RAG-Journey.git](https://github.com/MrJouH4/RAG-Journey.git)
cd RAG-Journey
```

### 2. Environment Configuration
Create a .env file in the root directory (or inside 03_deployment/) to securely handle your credentials:
```bash
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

### 3. Run via Docker (Recommended)
The deployment module is fully dockerized. It mounts your local index database dynamically and boots up an asynchronous web server:

```Bash
cd 03_deployment
docker-compose up --build
```
The API will be live at http://localhost:8000. You can explore the interactive documentation directly via Swagger UI at http://localhost:8000/docs.

## API Endpoints

### 1. Health Check
* **Method:** `GET`
* **Path:** `/`
* **Response:**
```json
  {
    "status": "online",
    "message": "Welcome to the RAG Journey Retrieval API"
  }
```

### 2. Query the RAG Pipeline
* **Method:** `POST`
* **Path:** `/ask`
* **Headers:** `Content-Type: application/json`
* **Payload:**
```JSON
{
  "question": "What is the primary topic discussed in the document?"
}
```
* **Response:**
```JSON
{
  "answer": "The primary topic discussed in the document is...",
  "context": [
    "Extracted text chunk 1 used for context matching...",
    "Extracted text chunk 2 used for context matching..."
  ]
}
```

## Roadmap
- [x] Document Loading & Text Splitting
- [x] Embeddings Extraction + Local FAISS Indexing
- [x] Basic RAG Chain Orchestration
- [x] Context Optimization via CrossEncoder Reranking
- [x] Hybrid Search Integration (BM25 + FAISS + RRF)
- [x] Query Transformation (Multi-Query + HyDE)
- [x] Enterprise Pipeline Evaluation (RAGAS Metrics Baseline)
- [x] Decoupled Retrieval Strategies (Parent-Document Retrieval)
- [x] Dynamic Metadata Filtering (Custom LCEL Query Constructor + Pydantic Guardrails)
- [x] Production Deployment (FastAPI, Uvicorn, Dockerization with volume mounting)
- [ ] Adaptive RAG & Agentic Guardrails

---
Built by [MrJouH4](https://github.com/MrJouH4) | Learning in public 🚀
