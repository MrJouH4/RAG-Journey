
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from embeddings import load_vectorstore

def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_tokens=512
    )

def get_retriever(vectorstore, k=3):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

def create_qa_chain(retriever):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the context to answer. If you don't know, say you don't know.\n\nContext:\n{context}"),
        ("human", "{input}"),
    ])
    chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, chain)

if __name__ == "__main__":
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)
    rag_chain = create_qa_chain(retriever)
    
    question = "what is this project about?"
    result = rag_chain.invoke({"input": question})
    print(f"\nAnswer: {result['answer']}")
