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
    if 'authentication_status' not in st.session_state:
        st.session_state['authentication_status'] = None
    if 'name' not in st.session_state:
        st.session_state['name'] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'logout_clicked' not in st.session_state:
        st.session_state['logout_clicked'] = False

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
    try:
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

        # Progress bars
        loading_progress = st.progress(0)
        processing_progress = st.progress(0)

        # Process documents
        docs = []
        total_files = len(uploaded_files)
        for i, uploaded_file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            pdf_loader = PyPDFLoader(file_path=tmp_file_path)
            docs += pdf_loader.load()
            loading_progress.progress((i + 1) / total_files)
            os.remove(tmp_file_path)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )
        documents = text_splitter.split_documents(docs)
        processing_progress.progress(33)

        # Create vector store
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        processing_progress.progress(66)

        template = """You are provided with a context extracted from Canadian parliamentary debates (Hansard) concerning various political issues.
        Answer the question by focusing on the relevant party based on the question. Provide the five to six main points and conclusion.
        
        Context: {context}
        Question: {question}
        
        Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            RunnableParallel(
                {"context": retriever, "question": RunnablePassthrough()}
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        processing_progress.progress(100)
        
        response = chain.invoke(query)

        buffer = BytesIO()
        doc = DocxDocument()
        doc.add_paragraph(f"Question: {query}\n\nAnswer:\n")
        doc.add_paragraph(response)
        doc.save(buffer)
        buffer.seek(0)

        return response, buffer
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return None, None

def handle_logout():
    """Handle logout functionality"""
    try:
        st.session_state['logout_clicked'] = True
        st.session_state['authentication_status'] = None
        st.session_state['name'] = None
        st.session_state['username'] = None
        # Clear any other session state variables
        for key in list(st.session_state.keys()):
            if key not in ['logout_clicked', 'authentication_status', 'name', 'username']:
                del st.session_state[key]
        st.rerun()
    except Exception as e:
        st.error(f"Error during logout: {str(e)}")

def main():
    init_session_state()
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
    if not st.session_state['authentication_status']:
        name, authentication_status, username = authenticator.login("Login", "main")
        st.session_state['name'] = name
        st.session_state['authentication_status'] = authentication_status
        st.session_state['username'] = username
        
        if authentication_status is False:
            st.error("Username/password is incorrect")
            st.stop()
        elif authentication_status is None:
            st.warning("Please enter your username and password")
            st.stop()
    
    # Main application
    if st.session_state['authentication_status']:
        # Sidebar setup
        st.sidebar.title("Hansard Analyzer")
        st.sidebar.write(f"Logged in as: {st.session_state['name']}")
        
        # Custom logout button in sidebar
        if st.sidebar.button("Logout"):
            handle_logout()
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
