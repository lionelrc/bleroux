import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import os
import tempfile

# Initialize session state for document storage
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

def process_file(uploaded_file):
    # Create a temporary file to store the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Load and process the PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    
    # Clean up the temporary file
    os.unlink(tmp_path)
    
    return documents

def initialize_chain(documents):
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
        llm=ChatAnthropic(model="claude-3-opus-20240229", temperature=0),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return chain

# Streamlit interface
st.title("Multi-Document Chat Assistant")

# File uploader
uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

# Process new uploads
if uploaded_files:
    all_documents = []
    
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        # Check if file is already processed
        if uploaded_file.name not in [doc['name'] for doc in st.session_state.processed_files]:
            with st.spinner(f'Processing {uploaded_file.name}...'):
                documents = process_file(uploaded_file)
                st.session_state.processed_files.append({
                    'name': uploaded_file.name,
                    'documents': documents
                })

# Display processed files
if st.session_state.processed_files:
    st.write("Processed Documents:")
    for file in st.session_state.processed_files:
        st.write(f"- {file['name']}")

    # Combine all documents
    all_documents = []
    for file in st.session_state.processed_files:
        all_documents.extend(file['documents'])

    # Initialize or update chain with all documents
    if all_documents:
        st.session_state.chain = initialize_chain(all_documents)

        # Initialize chat messages if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents"):
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

# Clear chat button
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Clear documents button
if st.button("Clear All Documents"):
    st.session_state.processed_files = []
    st.session_state.messages = []
    st.rerun()