import streamlit as st
from rag_chain import setup_qa_chain, ingest_docs

st.title("🔍 RAG App - Ask Your Documents")

uploaded_files = st.file_uploader("Upload documents (PDF/TXT)", accept_multiple_files=True)
if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded.")
    docs = ingest_docs(uploaded_files)
    qa_chain = setup_qa_chain(docs)

    question = st.text_input("Ask a question about your documents")
    if question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(question)
            st.markdown("**Answer:**")
            st.write(answer)
