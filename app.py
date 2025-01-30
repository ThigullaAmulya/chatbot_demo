# app.py
import streamlit as st
 
# Set page config first
st.set_page_config(page_title="RAG ChatBot", layout="wide")
 
# Import other dependencies
import os
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
 
# Load environment variables
load_dotenv()
 
# Configuration
URLS = [
    "https://www.geeksforgeeks.org/stock-price-prediction-project-using-tensorflow/",
    "https://www.geeksforgeeks.org/training-of-recurrent-neural-networks-rnn-in-tensorflow/"
]
INDEX_NAME = "huggingface"
MODEL_NAME = "HuggingFaceH4/zephyr-7b-alpha"
 
# Custom prompt template to get only assistant answers
PROMPT_TEMPLATE = """
Answer the question in detail based on the context below. 
If the question cannot be answered using the context, say "I don't know".
 
Context: {context}
Question: {question}
 
Detailed Answer:
"""
 
@st.cache_resource
def initialize_rag():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(INDEX_NAME):
        vectorstore = FAISS.load_local(INDEX_NAME, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = WebBaseLoader(URLS)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_NAME)
    
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    
    llm = HuggingFaceHub(
        repo_id=MODEL_NAME,
        model_kwargs={
            "temperature": 0.7,  # Increased for more creative/longer answers
            "max_new_tokens": 1024,  # Increased token limit for longer responses
            "max_length": 2048
        }
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PromptTemplate.from_template(PROMPT_TEMPLATE)}
    )
 
# Initialize RAG system
qa_chain = initialize_rag()
 
# Custom CSS for chat interface
st.markdown("""
<style>
    .title {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5em;
        padding: 20px;
        border-bottom: 2px solid #007bff;
        margin-bottom: 30px;
    }
    .chat-container {
        padding-bottom: 100px;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 15px 20px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 70%;
        float: left;
        clear: both;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .bot-message {
        background-color: #f8f9fa;
        color: #2c3e50;
        padding: 15px 20px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 70%;
        float: right;
        clear: both;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border: 1px solid #dee2e6;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: white;
        border-top: 1px solid #ddd;
        padding: 15px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)
 
# App title
st.markdown('<div class="title"> RAG Knowledge Assistant</div>', unsafe_allow_html=True)
 
# Session state management
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Chat container
chat_container = st.container()
 
# Display chat history
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
 
# Input form
with st.form("chat_input", clear_on_submit=True):
    cols = st.columns([0.85, 0.15])
    query = cols[0].text_input("Enter your query:", placeholder="Ask me anything...", label_visibility="collapsed")
    submitted = cols[1].form_submit_button(" Send")
 
# Process query
if submitted and query:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    
    try:
        # Get RAG response
        response = qa_chain.invoke({"query": query})
        answer = response["result"].split("Detailed Answer:")[-1].strip()
    except Exception as e:
        answer = f"⚠️ Error processing request: {str(e)}"
    
    # Add bot response to history
    st.session_state.messages.append({"role": "bot", "content": answer})
    
    # Rerun to update display
    st.rerun()
 