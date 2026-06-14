from json import load
import os
from unittest import loader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

class Source:
    def __init__(self):
        load_dotenv()
        os.environ["HF_API_KEY"] = os.getenv("HF_API_KEY")
    def process_pdf(file_path, persist_directory="./Chromadb"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", "","."])
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings()
        vector_db = Chroma.from_documents(texts, embeddings,persist_directory=persist_directory)
        return vector_db