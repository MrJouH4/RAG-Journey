import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = FastAPI(title="RAG Journey API", version="1.0")

# Define paths
DATA_DIR = "data"

# Global variables to hold our loaded assets in memory
embeddings = None
vector_store = None

@app.on_event("startup")
def load_vector_store():
    """Loads the embedding model and FAISS index into memory when the server starts."""
    global embeddings, vector_store
    print("Initializing FastAPI server components...")
    
    # 1. Initialize the same embedding model used during ingestion
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Check if the FAISS index files exist
    if not os.path.exists(os.path.join(DATA_DIR, "index.faiss")):
        print(f"Warning: No FAISS index found in '{DATA_DIR}'. Please run ingestion first.")
        return

    # 3. Load the local FAISS index
    print(f"Loading FAISS index from ./{DATA_DIR}...")
    # allow_dangerous_deserialization is required because LangChain uses pickle to load index.pkl
    vector_store = FAISS.load_local(
        DATA_DIR, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    print("FAISS Index loaded successfully. Vector search is ready.")

# Define Request and Response Schemas
class SearchQuery(BaseModel):
    query: str
    top_k: int = 3

class SearchResult(BaseModel):
    chunk_content: str
    metadata: dict

@app.get("/")
def read_root():
    return {"status": "online", "message": "Welcome to the RAG Journey Retrieval API"}

@app.post("/search", response_model=list[SearchResult])
def search_chunks(payload: SearchQuery):
    """Accepts a query, performs a vector search, and returns the top_k matching chunks."""
    global vector_store
    
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store index is not available.")
    
    try:
        # Perform similarity search
        docs = vector_store.similarity_search(payload.query, k=payload.top_k)
        
        # Format output payload
        results = [
            SearchResult(chunk_content=doc.page_content, metadata=doc.metadata)
            for doc in docs
        ]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")