import os
import tempfile
import streamlit as st
import streamlit_authenticator as stauth
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from docx import Document as DocxDocument
from io import BytesIO
import yaml
from yaml.loader import SafeLoader

# Page config
st.set_page_config(page_title="Hansard Insight Analyzer", layout="wide")

# Initialize session states
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'name' not in st.session_state:
    st.session_state['name'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None

# Load credentials and configuration from Streamlit Secrets
credentials = yaml.safe_load(st.secrets["general"]["credentials"])
cookie_name = st.secrets["general"]["cookie_name"]
cookie_key = st.secrets["general"]["cookie_key"]
cookie_expiry_days = st.secrets["general"]["cookie_expiry_days"]
openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]

# Create the authenticator object
authenticator = stauth.Authenticate(
    credentials,
    cookie_name,
    cookie_key,
    cookie_expiry_days
)

# Sidebar layout
with st.sidebar:
    if st.session_state.authentication_status:
        st.title("Session Settings")
        col1, col2 = st.columns([1,1])
        with col2:
            if authenticator.logout('Logout', 'sidebar'):
                st.session_state['authentication_status'] = None
                st.session_state['name'] = None
                st.session_state['username'] = None
                st.rerun()

# Main content
if not st.session_state.authentication_status:
    # Show login form
    st.title("Hansard Insight Analyzer - Login")
    name, authentication_status, username = authenticator.login()
    
    if authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')
    
    # Update session state
    st.session_state['authentication_status'] = authentication_status
    st.session_state['name'] = name
    st.session_state['username'] = username
    
    if authentication_status:
        st.rerun()

elif st.session_state.authentication_status:
    # Show main application
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.title("Hansard Insight Analyzer")
    
    # Model selection
    model_name = st.sidebar.selectbox("Select Model", ["gpt-4o-mini", "gpt-4o"])
    
    # File upload
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    # Query input
    query = st.text_input("Enter your query", value="What is the position of the Liberal Party on Carbon Pricing?")

    @st.cache_data(show_spinner=False)
    def process_documents(openai_api_key, model_name, uploaded_files, query):
        try:
            embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_api_key)
            llm = ChatOpenAI(temperature=0, model_name=model_name, max_tokens=4000, openai_api_key=openai_api_key)

            loading_progress = st.progress(0)
            processing_progress = st.progress(0)

            docs = []
            total_files = len(uploaded_files)
            for i, uploaded_file in enumerate(uploaded_files):
                with
