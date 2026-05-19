from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from splitter import split_recursive
from loader import load_document

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def create_vectorstore(chunks):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def save_vectorstore(vectorstore, path="vectorstore"):
    vectorstore.save_local(path)
    print(f"Vectorstore saved to {path}/")

def load_vectorstore(path="vectorstore"):
    embeddings = get_embeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    docs = load_document("test.pdf")
    chunks = split_recursive(docs)
    
    print("Creating vectorstore...")
    vectorstore = create_vectorstore(chunks)
    save_vectorstore(vectorstore)
    
    # Quick similarity test
    query = "what is this project about?"
    results = vectorstore.similarity_search(query, k=3)
    
    print(f"\nTop 3 results for: '{query}'")
    for i, doc in enumerate(results):
        print(f"\n--- Chunk {i+1} ---\n{doc.page_content[:200]}")