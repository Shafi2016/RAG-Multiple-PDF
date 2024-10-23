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

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'name' not in st.session_state:
    st.session_state.name = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'current_query' not in st.session_state:
    st.session_state.current_query = None

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

def clear_session_state():
    """Clear all session state variables"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]

@st.cache_data(show_spinner=False)
def process_documents(openai_api_key, model_name, uploaded_files, query):
    """Process documents and generate analysis"""
    # [Previous implementation remains the same]
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

    st.title("Hansard Analyzer")

    # Handle Authentication
    if not st.session_state.get('logged_in', False):
        try:
            authentication_status, name, username = authenticator.login()
            
            if authentication_status:
                st.session_state.logged_in = True
                st.session_state.name = name
                st.session_state.username = username
                st.rerun()
            elif authentication_status is False:
                st.error("Username/password is incorrect")
            else:
                st.warning("Please enter your username and password")
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
        return

    # Main application (only runs if logged in)
    if st.session_state.logged_in:
        # Sidebar for settings and logout
        with st.sidebar:
            st.write(f"Welcome *{st.session_state.name}*")
            
            # Logout button
            if st.button('Logout'):
                clear_session_state()
                st.rerun()
            
            st.markdown("### Settings")
            
            model_name = st.selectbox(
                "Select Model",
                ["gpt-4o-mini", "gpt-4o"]
            )
            
            uploaded_files = st.file_uploader(
                "Upload PDF files",
                type="pdf",
                accept_multiple_files=True
            )
        
        # Main content area
        st.markdown("## Hansard Insight Analyzer")
        
        query = st.text_input(
            "Enter your query",
            value="What is the position of the Liberal Party on Carbon Pricing?"
        )

        if st.button("Apply", type="primary"):
            if uploaded_files and query:
                with st.spinner("Analyzing documents..."):
                    answer, buffer = process_documents(
                        config['openai_api_key'],
                        model_name,
                        uploaded_files,
                        query
                    )
                    if answer and buffer:
                        st.session_state.analysis_result = answer
                        st.session_state.current_query = query
                        st.session_state.buffer = buffer
                        st.success("Analysis complete!")
            else:
                st.warning("Please upload PDF files and enter a query.")
        
        # Display results
        if st.session_state.get('analysis_result'):
            st.markdown(f"**Question: {st.session_state.current_query}**")
            st.markdown(f"**Answer:** {st.session_state.analysis_result}")
            
            # Download button
            st.download_button(
                label="Download Analysis",
                data=st.session_state.buffer,
                file_name="hansard_analysis.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

if __name__ == "__main__":
    main()
