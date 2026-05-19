from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)
from loader import load_document

def split_by_character(docs, chunk_size=1000, chunk_overlap=200):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def split_recursive(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def split_by_token(docs, chunk_size=250):
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)

if __name__ == "__main__":
    docs = load_document("test.pdf")
    
    chunks = split_recursive(docs)
    
    print(f"Original pages: {len(docs)}")
    print(f"Chunks after split: {len(chunks)}")
    print(f"\nChunk sample:\n{chunks[0].page_content[:300]}")