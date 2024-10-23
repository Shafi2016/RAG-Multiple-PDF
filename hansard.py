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
    # Initialize authentication state
    if 'authentication_status' not in st.session_state:
        st.session_state.authentication_status = None
    if 'name' not in st.session_state:
        st.session_state.name = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'stored_data' not in st.session_state:
        st.session_state.stored_data = None

def load_config():
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
        return None

def process_documents(openai_api_key, model_name, uploaded_files, query):
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

        if not docs:
            st.error("No documents were processed successfully.")
            return None, None

        progress_text.text("Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )
        splits = text_splitter.split_documents(docs)
        processing_progress.progress(0.4)

        if not splits:
            st.error("Document splitting failed.")
            return None, None

        progress_text.text("Creating vector store...")
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        processing_progress.progress(0.6)

        progress_text.text("Analyzing content...")
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

        if not response:
            st.error("No response generated from the analysis.")
            return None, None

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
        st.error(f"Error in document processing: {str(e)}")
        return None, None

def main():
    init_session_state()
    config = load_config()
    
    if not config:
        st.error("Failed to load configuration")
        return

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie_name'],
        config['cookie_key'],
        config['cookie_expiry_days']
    )

    if not st.session_state.authentication_status:
        st.title("Hansard Analyzer")
        authentication_status, name, username = authenticator.login()
        st.session_state.authentication_status = authentication_status
        st.session_state.name = name
        st.session_state.username = username
        
        if authentication_status is False:
            st.error("Username/password is incorrect")
            return
        elif authentication_status is None:
            st.warning("Please enter your username and password")
            return
    
    if st.session_state.authentication_status:
        # Sidebar
        with st.sidebar:
            st.write(f"Welcome *{st.session_state.name}*")
            
            # Logout button
            if st.button('Logout'):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.session_state.authentication_status = None
                st.experimental_rerun()
            
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

        # Main content
        st.markdown("## Hansard Insight Analyzer")
        
        query = st.text_input(
            "Enter your query",
            value="What is the position of the Liberal Party on Carbon Pricing?"
        )

        if st.button("Apply", type="primary"):
            if uploaded_files and query:
                with st.spinner("Analyzing documents..."):
                    try:
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
                            st.error("Failed to generate analysis. Please try again.")
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
            else:
                st.warning("Please upload PDF files and enter a query.")

if __name__ == "__main__":
    main()
