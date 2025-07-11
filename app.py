# 🦩 Llama-based RAG Streamlit App to Extract Disease Info (with Ollama)

import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import os



# 1. Load and preprocess data (assuming a .csv file with a 'text' column)
@st.cache_data
def load_data():
    try:
        file_path = "medical_data.csv"
        loader = CSVLoader(file_path=file_path, source_column="output", encoding="utf-8")
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return splitter.split_documents(docs)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error(f"Currentstr working directory: {os.getcwd()}")
        return []

# 2. Set up vectorstore with HuggingFace embeddings
@st.cache_resource
def create_vectorstore(_documents):  # fixed argument name
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(_documents, embeddings)

# 3. Initialize Ollama Llama model
@st.cache_resource
def load_llama():
    return Ollama(model="llama3.2:latest")  # Replace with your pulled model from `ollama list`

# 4. Set up RAG chain
@st.cache_resource
def setup_rag_chain(_vectorstore):
    prompt_template = PromptTemplate(
        input_variables=["context"],
        template="""
You are an expert medical assistant. From the medical text below, extract the following details clearly and present them in simple plain English (no JSON):

- Symptoms
- Treatment
- Impact on Pregnancy
- Mental Health Impact
- Age group affected

Use bullet points or short paragraphs for clarity. Do not include any formatting or code blocks. Dont write text like Here are the extracted details in simple plain English in response:

Text:
{context}
"""
    )

    retriever = _vectorstore.as_retriever()
    llama = load_llama()
    return RetrievalQA.from_chain_type(
        llm=llama,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )

# 5. Streamlit UI
st.title("🩺 Medical Data ChatBot")

query = st.text_input("Enter a disease name to extract info:")
if query:
    with st.spinner("Processing..."):
        documents = load_data()
        st.write(f"Loaded {len(documents)} documents.")
        vectorstore = create_vectorstore(documents)
        rag_chain = setup_rag_chain(vectorstore)
        response = rag_chain.run(query)

        st.markdown("### Information")
        st.write(response)  # Output as plain text

st.markdown("---")

