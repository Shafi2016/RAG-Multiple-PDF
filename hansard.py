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

# Initialize session state at the very beginning
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'name' not in st.session_state:
    st.session_state['name'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'logout' not in st.session_state:
    st.session_state['logout'] = False

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

# Your process_documents function remains the same
@st.cache_data(show_spinner=False)
def process_documents(openai_api_key, model_name, uploaded_files, query):
    """Process documents and generate analysis"""
    # Implementation remains the same
    pass

def main():
    config = load_config()
    
    # Create authenticator
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie_name'],
        config['cookie_key'],
        config['cookie_expiry_days']
    )

    # Check if logout was clicked in the previous run
    if st.session_state['logout']:
        st.session_state['logout'] = False
        st.session_state['authentication_status'] = None
        st.session_state['name'] = None
        st.session_state['username'] = None
        for key in list(st.session_state.keys()):
            if key not in ['authentication_status', 'name', 'username', 'logout']:
                del st.session_state[key]
        st.rerun()

    # Show title
    st.title("Hansard Analyzer")

    # Handle Authentication
    if not st.session_state['authentication_status']:
        # Login form
        authentication_status, name, username = authenticator.login()
        
        st.session_state['authentication_status'] = authentication_status
        st.session_state['name'] = name
        st.session_state['username'] = username
        
        if authentication_status is False:
            st.error("Username/password is incorrect")
            return
        elif authentication_status is None:
            st.warning("Please enter your username and password")
            return

    # Main application
    if st.session_state['authentication_status']:
        # Sidebar
        st.sidebar.title("Hansard Analyzer")
        st.sidebar.write(f"Logged in as: {st.session_state['name']}")
        
        # Simple logout button
        if st.sidebar.button('Logout', key='logout_button'):
            st.session_state['logout'] = True
            st.rerun()
        
        st.sidebar.title("Analysis Settings")
        
        # Model selection
        model_name = st.sidebar.selectbox(
            "Select Model",
            ["gpt-4o-mini", "gpt-4o"]
        )
        
        # File upload
        uploaded_files = st.sidebar.file_uploader(
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
