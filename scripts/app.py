import streamlit as st
from rag_qa import rag_qa, conversation_history

st.set_page_config(page_title="Dr. X RAG QA", layout="wide")

st.title("🧠 Dr. X Research Q&A")
st.markdown("Ask questions about the documents. The system uses offline RAG (FAISS + LLaMA)")

# User input
user_input = st.text_input("Ask a question")

if user_input:
    answer, chunks = rag_qa(user_input)
    conversation_history.append((user_input, answer))

    st.markdown("### 📘 Answer")
    st.write(answer)

    st.markdown("### 📚 Retrieved Context")
    for i, chunk in enumerate(chunks, 1):
        st.markdown(f"**Chunk {i}** - {chunk['source_file']} (Page {chunk['page']})")
        st.code(chunk['text'], language='markdown')
