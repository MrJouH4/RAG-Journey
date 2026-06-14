# RAG Journey 🧠

Building production-grade Retrieval-Augmented Generation (RAG) systems from core basics to advanced architectures using LangChain.

## What is RAG?
Instead of relying only on static training data, RAG enables Large Language Models to dynamically **query external knowledge bases** and generate factually grounded responses.
Document → Chunks → Embeddings → Vectorstore → Retriever → LLM → Answer

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
└── └── 07_agentic_rag.ipynb           # Coming soon
```
## Tech Stack
- Orchestration: LangChain (LCEL Pattern)
- Inference Engine: Groq API (llama-3.1-8b-instant)
- Embedding Model: HuggingFace (all-MiniLM-L6-v2)
- Vector Database: FAISS
- Lexical Search: BM25
- Reranking: Cross-Encoder (ms-marco-MiniLM-L-6-v2)
- Evaluation: RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
- Data Validation: Pydantic

## Setup

```bash
git clone https://github.com/MrJouH4/RAG-Journey.git
cd RAG-Journey
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

```bash
export HF_TOKEN=your_huggingface_token
export GROQ_API_KEY=your_groq_api_key
```

## Roadmap
- [x] Document Loading
- [x] Text Splitting
- [x] Embeddings + FAISS
- [x] Basic RAG Chain
- [x] Reranking with CrossEncoder
- [x] Hybrid Search (BM25 + FAISS + RRF)
- [x] Query Transformation (Multi-Query + HyDE)
- [x] Enterprise Pipeline Evaluation (RAGAS Metrics Baseline)
- [x] Decoupled Retrieval Strategies (Parent-Document Retrieval)
- [x] Dynamic Metadata Filtering (Custom LCEL Query Constructor + Pydantic Guardrails)
- [ ] Adaptive RAG & Agentic Guardrails

---
Built by [MrJouH4](https://github.com/MrJouH4) | Learning in public 🚀
