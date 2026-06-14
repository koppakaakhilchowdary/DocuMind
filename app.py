from calendar import c
from json import load
import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from source import Source
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
# from langchain.chains.retrieval import create_retrieval_chain


st.set_page_config(page_title="DocuMind", page_icon=":books:", layout="wide")
st.header("DocuMind 📚")

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
with st.sidebar:
    st.title("Settings")
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
if uploaded_file:
    if not os.path.exists("data"): os.mkdir("data")
    file_path = os.path.join("data",uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Processing PDF..."):
        vector_db = Source.process_pdf(file_path)
        st.success("PDF processed successfully!")
    
    llm = ChatGroq(model = "openai/gpt-oss-120b")
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever(search_kwargs={"k":3})
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    user_input = st.text_input("Ask a question about your document:")
    if user_input:
        with st.spinner("Generating answer..."):
            response = qa_chain.invoke({"input": user_input})
            st.markdown("AI Response:")
            st.write(response['answer'])
            with st.expander("Source Documents"):
                for doc in response['context']:
                    st.write(f"**Page Content:** {doc.page_content[:200]}")
                #     st.write(f"**Metadata:** {doc.metadata}")
                #     st.write("---")    