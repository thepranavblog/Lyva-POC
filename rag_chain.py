from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os
import tempfile

def ingest_docs(uploaded_files):
    all_texts = []

    for file in uploaded_files:
        suffix = os.path.splitext(file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".txt":
            loader = TextLoader(tmp_path)
        else:
            continue

        docs = loader.load()
        all_texts.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_texts)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def setup_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.7, "max_length": 512})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    return qa_chain
