import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import os
from tempfile import NamedTemporaryFile

# Use Streamlit secrets for API key
if "ANTHROPIC_API_KEY" not in st.secrets:
    st.error("ANTHROPIC_API_KEY not found in secrets. Please add it to your .streamlit/secrets.toml file or Streamlit Cloud secrets.")
    st.stop()

def process_pdf(uploaded_file):
    with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        loader = PyPDFLoader(tmp_file.name)
        documents = loader.load()
        os.unlink(tmp_file.name)  # Delete the temporary file
        return documents

def initialize_chain(documents=None):
    if not documents:
        st.warning("Please upload a PDF document to start.")
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(documents)
    
    if not splits:
        st.warning("No text content found in the uploaded document.")
        return None
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatAnthropic(
            api_key=st.secrets["ANTHROPIC_API_KEY"],
            model="claude-3-opus-20240229", 
            temperature=0
        ),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return chain

# Streamlit interface
st.title("PDF Document Assistant")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document",