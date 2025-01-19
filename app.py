import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import os
from tempfile import NamedTemporaryFile

# Create a directory for ChromaDB
CHROMA_DB_DIR = "chroma_db"
if not os.path.exists(CHROMA_DB_DIR):
    os.makedirs(CHROMA_DB_DIR)

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
    
    # Initialize ChromaDB with persistent directory
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    vectorstore.persist()  # Make sure to persist the database
    
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

# Add to .gitignore
if not os.path.exists(".gitignore"):
    with open(".gitignore", "a") as f:
        f.write("\nchroma_db/\n")

# Streamlit interface
st.title("PDF Document Assistant")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# Initialize or update session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# Process uploaded file
if uploaded_file and (not st.session_state.chain or "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name):
    with st.spinner("Processing document..."):
        documents = process_pdf(uploaded_file)
        st.session_state.chain = initialize_chain(documents)
        st.session_state.current_file = uploaded_file.name
        st.success("Document processed successfully!")

# Display chat interface only if chain is initialized
if st.session_state.chain:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
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
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.info("Upload a PDF document to start chatting!")