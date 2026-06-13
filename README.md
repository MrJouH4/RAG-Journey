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
│   ├── 06_metadata_filtering.ipynb    # Coming soon
└── └── 07_agentic_rag.ipynb           # Coming soon
```
## Stack
- **LangChain** — RAG framework
- **HuggingFace** — Embeddings (all-MiniLM-L6-v2)
- **FAISS** — Vector store
- **Groq** — LLM inference (Llama 3.1 8B)
- **CrossEncoder** — Reranking (ms-marco-MiniLM-L-6-v2)
- **BM25** — Keyword search
- **RRF** — Reciprocal Rank Fusion
- **RAGAS** — RAG evaluation framework (Faithfulness, Answer Relevancy, Context Precision, Context Recall)

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
- [ ] Structured Metadata Filtering & Self-Querying Retrievers
- [ ] Adaptive RAG & Agentic Guardrails

---
Built by [MrJouH4](https://github.com/MrJouH4) | Learning in public 🚀
