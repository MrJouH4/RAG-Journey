# RAG Journey 🧠

Building Retrieval Augmented Generation (RAG) from basics to advanced using LangChain.

## What is RAG?
Instead of relying only on what the LLM was trained on, RAG allows the model to **search your own documents** and answer based on them.
Document → Chunks → Embeddings → Vectorstore → Retriever → LLM → Answer

## Project Structure
```bash
RAG-Journey/
├── 01_basics/
│   ├── loader.py       # Load PDF, TXT, CSV files
│   ├── splitter.py     # Split docs into chunks
│   ├── embeddings.py   # Embed chunks + FAISS vectorstore
│   └── retriever.py    # RAG chain with Groq LLM
└── 02_advanced/        
│   ├── 01_reranking.ipynb             # Reranking with CrossEncoder
│   ├── 02_hybrid_search.ipynb         # BM25 + FAISS + RRF
│   ├── 03_query_transformation.ipynb  # Coming soon
│   ├── 04_evaluation.ipynb            # Coming soon
│   ├── 05_parent_child.ipynb          # Coming soon
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
- [ ] Evaluation
- [ ] Parent-Child
- [ ] Metadata Filtering
- [ ] Agentic RAG

---
Built by [MrJouH4](https://github.com/MrJouH4) | Learning in public 🚀
