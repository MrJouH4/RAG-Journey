from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader
)

def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()

def load_txt(path: str):
    loader = TextLoader(path)
    return loader.load()

def load_csv(path: str):
    loader = CSVLoader(path)
    return loader.load()

def load_document(path: str):
    """Auto-detect file type and load"""
    if path.endswith(".pdf"):
        return load_pdf(path)
    elif path.endswith(".txt"):
        return load_txt(path)
    elif path.endswith(".csv"):
        return load_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

# Quick test
if __name__ == "__main__":
    docs = load_document("test.pdf")
    print(f"Loaded {len(docs)} pages")
    print(f"First 200 chars: {docs[0].page_content[:200]}")