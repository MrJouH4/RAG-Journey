import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def extract_text_from_pdf(pdf_path):
    print(f"Opening PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    print(f"Extraction complete. Total raw characters: {len(full_text)}")
    return full_text

if __name__ == "__main__":
    pdf_file = os.path.join("ingestion", "rag_paper.pdf")
    output_dir = "data"  # Saves straight to the folder mounted by Docker Compose
    
    if not os.path.exists(pdf_file):
        print(f"Error: Could not find {pdf_file}.")
    else:
        # 1. Extract and Split Text
        raw_text = extract_text_from_pdf(pdf_file)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(raw_text)
        print(f"Generated {len(chunks)} chunks from PDF.")
        
        # 2. Initialize Local Embedding Model (Download happens once ~90MB)
        print("Loading local all-MiniLM-L6-v2 embedding model...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 3. Build FAISS Vector Store and Embed the Chunks
        print("Computing embeddings and building FAISS index...")
        vector_store = FAISS.from_texts(chunks, embeddings)
        
        # 4. Serialize and Save Locally
        print(f"Saving FAISS index files to ./{output_dir}...")
        vector_store.save_local(output_dir)
        print("Success! Data ingestion pipeline complete.")