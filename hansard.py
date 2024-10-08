import os
import tempfile
import streamlit as st
import streamlit_authenticator as stauth
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from docx import Document as DocxDocument
from io import BytesIO
import yaml
from yaml.loader import SafeLoader

# Load credentials and configuration from Streamlit Secrets
credentials = yaml.safe_load(st.secrets["general"]["credentials"])
cookie_name = st.secrets["general"]["cookie_name"]
cookie_key = st.secrets["general"]["cookie_key"]
cookie_expiry_days = st.secrets["general"]["cookie_expiry_days"]

openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]

# Create an authenticator object with hashed passwords
authenticator = stauth.Authenticate(
    credentials,
    cookie_name,
    cookie_key,
    cookie_expiry_days,
    None  # No preauthorized emails in this example
)

# Display the login form in the main area
name, authentication_status, username = authenticator.login('main')

if authentication_status:
    # Successful login
    authenticator.logout('Logout', 'main')  # Added logout option
    st.write(f'Welcome *{name}*')
    # Main app functionality
    st.title("Hansard Insight Analyzer")
    st.sidebar.title("Session Settings")
    
    model_name = st.sidebar.selectbox("Select Model", ["gpt-4o-mini", "gpt-4o"])

    # Upload PDF Files
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    # Query Input
    query = st.text_input("Enter your query", value="What is the position of the Liberal Party on Carbon Pricing?")

    @st.cache_data(show_spinner=False)
    def process_documents(openai_api_key, model_name, uploaded_files, query):
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_api_key)
        llm = ChatOpenAI(temperature=0, model_name=model_name, max_tokens=4000, openai_api_key=openai_api_key)

        loading_progress = st.progress(0)
        processing_progress = st.progress(0)

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

        prompt = ChatPromptTemplate.from_template("""
            You are provided with a context extracted from Canadian parliamentary debates (Hansard) concerning various political issues.
            Answer the question by focusing on the relevant party based on the question. Provide the five to six main points and conclusion.
            {context}
            Question: {input}
        """)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        documents = text_splitter.split_documents(docs)
        processing_progress.progress(33)

        vector = FAISS.from_documents(documents, embeddings)
        processing_progress.progress(66)

        retriever = vector.as_retriever(search_kwargs={"k": 5})
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        processing_progress.progress(100)

        response = retrieval_chain.invoke({"input": query})

        buffer = BytesIO()
        doc = DocxDocument()
        doc.add_paragraph(f"### Question: {query}\n\n**Answer:**\n")
        doc.add_paragraph(response['answer'])
        doc.save(buffer)
        buffer.seek(0)

        return response['answer'], buffer

    if st.button("Apply"):
        if uploaded_files and openai_api_key and query:
            st.sidebar.success("Files uploaded successfully!")
            answer, buffer = process_documents(openai_api_key, model_name, uploaded_files, query)
            st.markdown(f"### Question: {query}\n\n**Answer:** {answer}\n")
            st.session_state['buffer'] = buffer
        else:
            st.sidebar.warning("Please upload PDFs and enter your query.")

    if 'buffer' in st.session_state:
        st.download_button(
            label="Download .docx",
            data=st.session_state['buffer'],
            file_name="response.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')
