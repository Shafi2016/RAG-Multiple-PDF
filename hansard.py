import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from docx import Document as DocxDocument
from io import BytesIO

# Streamlit UI setup
st.title("Hansard Insight Analyzer")
st.sidebar.title("Session Settings")

# # 1. Input API Key
# api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]
# 2. Model Selection
model_name = st.sidebar.selectbox("Select Model", ["gpt-4o-mini", "gpt-4o"])

# 3. Upload PDF Files
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# 4. Query Input
query = st.text_input("Enter your query", value="What is the position of the Liberal Party on Carbon Pricing?")

# Function to process documents and generate response and file
@st.cache_data(show_spinner=False)  # Cache the function to avoid re-running
def process_documents(api_key, model_name, uploaded_files, query):
    # Initialize the embeddings and LLM
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=api_key)
    llm = ChatOpenAI(temperature=0, model_name=model_name, max_tokens=4000, openai_api_key=api_key)

    # Initialize progress bars
    loading_progress = st.progress(0)
    processing_progress = st.progress(0)

    # Process PDFs
    docs = []
    total_files = len(uploaded_files)
    for i, uploaded_file in enumerate(uploaded_files):
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load the PDF using the temporary file path
        pdf_loader = PyPDFLoader(file_path=tmp_file_path)
        docs += pdf_loader.load()

        # Update progress for loading
        loading_progress.progress((i + 1) / total_files)

        # Clean up temporary file
        os.remove(tmp_file_path)
    
    # Prepare the Chat Prompt Template
    prompt = ChatPromptTemplate.from_template("""
        You are provided with a context extracted from Canadian parliamentary debates (Hansard) concerning various political issues. 
        Answer the question by focusing on the relevant party based on the question. Provide the five to six main points and conclusion. 
    
        If the question asks about the Liberal Party, focus on the Liberal Party's viewpoint. If the question asks about the Conservative Party, focus on the Conservative Party's viewpoint. 
    
        Provide detailed information including their proposals, policy stance, and any arguments made during the debate.
    
        <context>
        {context}
        </context>
    
        Question: {input}
    
        **Main Points:**
        1- Six main points summarizing the party's stance 
    
        **Supporting Quotes:**
        2- List specific quotes that support the analysis, including the names of the individuals who made them or references from the debates
    
        **Potential Implications of Each Party's Stance:**
        3 - Any significant points raised during the debate, including potential implications of each party's stance for each Question
    
        **Conclusion:**
        4- Summarize the party's stance and its implications
    """)

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    documents = text_splitter.split_documents(docs)
    processing_progress.progress(33)

    # Create document embeddings
    vector = FAISS.from_documents(documents, embeddings)
    processing_progress.progress(66)

    # Build the retrieval chain
    retriever = vector.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant chunks
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    processing_progress.progress(100)

    # Get the response from the model
    response = retrieval_chain.invoke({"input": query})

    # Generate the .docx file in memory
    buffer = BytesIO()
    doc = DocxDocument()
    doc.add_paragraph(f"### Question: {query}\n\n**Answer:**\n")
    doc.add_paragraph(response['answer'])
    doc.save(buffer)
    buffer.seek(0)

    return response['answer'], buffer

# 5. Add an "Apply" button to trigger the model processing
if st.button("Apply"):
    if uploaded_files and api_key and query:
        st.sidebar.success("Files uploaded successfully!")

        # Process documents and generate the response and .docx file
        answer, buffer = process_documents(api_key, model_name, uploaded_files, query)

        # Display the response in Markdown
        st.markdown(f"### Question: {query}\n\n**Answer:** {answer}\n")

        # Store buffer in session state for download later
        st.session_state['buffer'] = buffer
    else:
        st.sidebar.warning("Please upload PDFs, enter your API key, and input a query.")

# 6. Add a "Download" button to download the file after "Apply" has been clicked
if 'buffer' in st.session_state:
    st.download_button(
        label="Download .docx",
        data=st.session_state['buffer'],
        file_name="response.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
