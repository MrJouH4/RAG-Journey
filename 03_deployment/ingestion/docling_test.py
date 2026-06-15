from langchain_docling.loader import DoclingLoader



if __name__ == "__main__":

    FILE_PATH = "F:/machine-learning/RAG-Journey/03_deployment/ingestion/rag_paper.pdf"

    loader = DoclingLoader(file_path=FILE_PATH)
    docs = loader.load()
    for d in docs[:3]:
        print(f"- {d.page_content=}")