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
if 'authentication_status' not in st.session_state:
    st.session_state.authentication_status = None

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
    """Process documents and generate analysis"""
    try:
        # Initialize progress indicators
        progress_text = st.empty()
        progress_text.text("Loading documents...")
        loading_progress = st.progress(0)
        processing_progress = st.progress(0)

        # Initialize language models
        embeddings = OpenAIEmbeddings(
            model='text-embedding-3-small',
            openai_api_key=openai_api_key
        )
        llm = ChatOpenAI(
            temperature=0,
            model_name=model_name,
            max_tokens=4000,
            openai_api_key=openai_api_key
        )

        # Process documents
        docs = []
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress_text.text(f"Processing file {i+1} of {total_files}...")
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(file_path=tmp_file_path)
            docs.extend(loader.load())
            
            os.remove(tmp_file_path)
            loading_progress.progress((i + 1) / total_files)

        progress_text.text("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )
        splits = text_splitter.split_documents(docs)
        processing_progress.progress(0.4)

        progress_text.text("Creating vector store...")
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        processing_progress.progress(0.6)

        prompt = ChatPromptTemplate.from_template("""
            You are provided with a context extracted from Canadian parliamentary debates (Hansard) concerning various political issues.
            Answer the question by focusing on the relevant party based on the question. Provide the five to six main points and conclusion.
            {context}
            Question: {input}
        """)

        chain = (
            RunnableParallel(
                {"context": retriever, "input": RunnablePassthrough()}
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        
        processing_progress.progress(0.8)
        response = chain.invoke(query)
        processing_progress.progress(1.0)

        # Create document
        buffer = BytesIO()
        doc = DocxDocument()
        doc.add_paragraph(f"Question: {query}\n\nAnswer:\n")
        doc.add_paragraph(response)
        doc.save(buffer)
        buffer.seek(0)

        # Clear progress indicators
        progress_text.empty()
        loading_progress.empty()
        processing_progress.empty()

        return response, buffer

    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return None, None

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
    if not st.session_state["authentication_status"]:
        # Updated login method call without deprecated parameters
        authentication_status, name, username = authenticator.login()
        
        if authentication_status == False:
            st.error('Username/password is incorrect')
        elif authentication_status == None:
            st.warning('Please enter your username and password')
        
        if authentication_status:
            st.session_state["authentication_status"] = True
            st.session_state["name"] = name
            st.session_state["username"] = username
            st.rerun()

    # Main application
    if st.session_state["authentication_status"]:
        # Sidebar for settings and logout
        with st.sidebar:
            st.write(f"Welcome *{st.session_state['name']}*")
            
            if st.button("Logout", key='unique_logout'):
                st.session_state["authentication_status"] = None
                st.session_state["name"] = None
                st.session_state["username"] = None
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
                        st.markdown(f"**Question: {query}**")
                        st.markdown(f"**Answer:** {answer}")
                        
                        st.download_button(
                            label="Download Analysis",
                            data=buffer,
                            file_name="hansard_analysis.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
            else:
                st.warning("Please upload PDF files and enter a query.")

if __name__ == "__main__":
    main()
