import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter

def test_text_splitting():
    sample_text = "RAG stands for Retrieval-Augmented Generation. It bridges the gap between static LLMs and dynamic real-time data data streams."
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
        separators=[" ", ""]
    )
    chunks = text_splitter.split_text(sample_text)
    
    assert isinstance(chunks, list), "Output should be a list of text chunks."
    assert len(chunks) > 1, "the text should be split into multiple chunks."

    for chunk in chunks:
        assert len(chunk) <= 50, "Chunk exceeded maximum size limit: '{chunk}' ({len(chunk)} characters)"
