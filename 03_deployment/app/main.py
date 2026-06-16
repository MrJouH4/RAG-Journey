import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="RAG Journey API", version="1.0")

DATA_DIR = "data"
embeddings = None
vector_store = None
llm = None

@app.on_event("startup")
def load_vector_store():
    """Loads the embedding model and FAISS index into memory when the server starts."""
    global embeddings, vector_store, llm
    print("Initializing FastAPI server components...")
    
    # Embeddings and Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(os.path.join(DATA_DIR, "index.faiss")):
        vector_store = FAISS.load_local(DATA_DIR, embeddings, allow_dangerous_deserialization=True)
        print("FAISS Index loaded successfully.")
    
    
    api_key = os.environ.get("GROQ_API_KEY", "mock_key_for_local_testing")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    print("Groq LLM initialized.")


class QueryRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    answer: str
    context: list[str]

@app.get("/")
def read_root():
    return {"status": "online", "message": "Welcome to the RAG Journey Retrieval API"}

@app.post("/query", response_model=RAGResponse)
def ask_rag(payload: QueryRequest):
    """Retrieves context chunks and generates an answer using Groq."""
    global vector_store, llm

    if vector_store is None or llm is None:
        raise HTTPException(status_code=503, detail="RAG components are not fully initialized.")
    
    try:
        # 1. Retrieve the top 3 relevant chunks
        docs = vector_store.similarity_search(payload.question, k=3)
        context_texts = [doc.page_content for doc in docs]
        
        # 2. Construct the prompt
        system_prompt = (
            "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
            "If you do not know the answer based on the context, say 'I cannot find the answer in the document.'\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

        # 3. Chain and execute
        chain = prompt | llm
        formatted_context = "\n---\n".join(context_texts)
        response = chain.invoke({"context": formatted_context, "question": payload.question})
        
        return RAGResponse(answer=response.content, context=context_texts)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG execution failed: {str(e)}")