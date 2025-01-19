import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import os

# Use Streamlit secrets for API key
if "ANTHROPIC_API_KEY" not in st.secrets:
    st.error("ANTHROPIC_API_KEY not found in secrets. Please add it to your .streamlit/secrets.toml file or Streamlit Cloud secrets.")
    st.stop()

def initialize_chain():
    documents = []
    for file in os.listdir("documents"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"documents/{file}")
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(documents)
    
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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = initialize_chain()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    response = st.session_state.chain({
        "question": prompt,
        "chat_history": [(m["content"], r["content"]) 
                        for m, r in zip(st.session_state.messages[::2], 
                                      st.session_state.messages[1::2])]
    })
    
    # Add AI response to chat history
    ai_response = response["answer"]
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)