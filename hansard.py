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
st.set_page_config(page_title="Hansard Analyzer", layout="wide")

def init_session_state():
    """Initialize session state variables"""
    session_vars = {
        'authentication_status': None,
        'name': None,
        'username': None,
        'init_done': False
    }
    for var, value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = value

def load_config():
    """Load configuration from secrets"""
    try:
        return {
            'credentials': yaml.safe_load(st.secrets["general"]["credentials"]),
            'cookie_name': st.secrets["general"]["cookie_name"],
            'cookie_key': st.secrets["general"]["cookie_key"],
            'cookie_expiry_days': st.secrets["general"]["cookie_expiry_days"],
            'openai_api_key': st.secrets["general"]["OPENAI_API_KEY"]
        }
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        st.stop()

@st.cache_data(show_spinner=False)
def process_documents(openai_api_key, model_name, uploaded_files, query):
    # [Previous implementation remains the same]
    # Removed for brevity - no changes needed here
    pass

def perform_logout():
    """Perform logout and clear all session state"""
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Reinitialize essential states
    st.session_state.authentication_status = None
    st.session_state.name = None
    st.session_state.username = None
    
    # Force a rerun to clear the page
    st.rerun()

def main():
    # Initialize session state
    init_session_state()
    
    # Load configuration
    config = load_config()
    
    # Create authenticator
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie_name'],
        config['cookie_key'],
        config['cookie_expiry_days']
    )

    # Show title only if not authenticated
    if not st.session_state.authentication_status:
        st.title("Hansard Analyzer")

    # Handle Authentication
    if not st.session_state.authentication_status:
        try:
            authentication_status, name, username = authenticator.login()
            
            if authentication_status:
                st.session_state.authentication_status = authentication_status
                st.session_state.name = name
                st.session_state.username = username
                st.rerun()  # Rerun to update the UI with authenticated state
            elif authentication_status is False:
                st.error("Username/password is incorrect")
            else:
                st.warning("Please enter your username and password")
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            st.stop()
    
    # Main application
    if st.session_state.authentication_status:
        # Sidebar setup
        with st.sidebar:
            st.title("Hansard Analyzer")
            st.write(f"Logged in as: {st.session_state.name}")
            
            # Logout button in sidebar
            if st.button("Logout", key="logout_button"):
                perform_logout()
            
            st.title("Analysis Settings")
            
            # Model selection
            model_name = st.selectbox(
                "Select Model",
                ["gpt-4o-mini", "gpt-4o"]
            )
            
            # File upload
            uploaded_files = st.file_uploader(
                "Upload PDF files",
                type="pdf",
                accept_multiple_files=True
            )
        
        # Main content
        st.title("Hansard Insight Analyzer")
        
        # Query input
        query = st.text_input(
            "Enter your query",
            value="What is the position of the Liberal Party on Carbon Pricing?"
        )

        # Analysis button
        if st.button("Analyze Documents"):
            if uploaded_files and query:
                with st.spinner("Processing documents..."):
                    answer, buffer = process_documents(
                        config['openai_api_key'],
                        model_name,
                        uploaded_files,
                        query
                    )
                    if answer and buffer:
                        st.success("Analysis complete!")
                        st.markdown(f"Question: {query}\n\nAnswer: {answer}")
                        st.session_state['buffer'] = buffer
            else:
                st.warning("Please upload PDF files and enter a query.")

        # Download button
        if 'buffer' in st.session_state:
            st.download_button(
                label="Download Analysis",
                data=st.session_state['buffer'],
                file_name="hansard_analysis.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

if __name__ == "__main__":
    main()
