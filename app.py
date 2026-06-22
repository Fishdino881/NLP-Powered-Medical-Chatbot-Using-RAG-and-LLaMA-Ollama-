import os
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# --- Helper to get API key from secrets or environment ---
def get_groq_api_key():
    # 1. Try Streamlit secrets (recommended for Streamlit Cloud)
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    # 2. Fallback to environment variable
    return os.environ.get("GROQ_API_KEY")

# 1. Load and preprocess data (cached once)
@st.cache_data
def load_data():
    try:
        file_path = "medical_data.csv"
        if not os.path.exists(file_path):
            st.error(f"❌ File '{file_path}' not found. Please upload it or place it in the root directory.")
            return []
        loader = CSVLoader(file_path=file_path, source_column="output", encoding="utf-8")
        docs = loader.load()
        if not docs:
            st.warning("No documents loaded. Check your CSV content.")
            return []
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return splitter.split_documents(docs)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return []

# 2. Vectorstore with HuggingFace embeddings (cached)
@st.cache_resource
def create_vectorstore(_documents):
    if not _documents:
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(_documents, embeddings)

# 3. Load LLM – using Groq (free API key required)
@st.cache_resource
def load_llm():
    api_key = get_groq_api_key()
    if not api_key:
        st.error("🚫 GROQ_API_KEY not set. Please add it to your Streamlit secrets.")
        st.info("Go to your app dashboard → Settings → Secrets and add: `GROQ_API_KEY = \"your-key\"`")
        return None
    try:
        return ChatGroq(
            model="llama-3.3-70b-versatile",   # <-- UPDATED: actively supported model
            api_key=api_key,
            temperature=0.2,
            max_retries=2,
        )
    except Exception as e:
        st.error(f"Failed to initialize Groq LLM: {str(e)}")
        return None

# 4. RAG chain setup (cached)
@st.cache_resource
def setup_rag_chain(_vectorstore, _llm):
    if _vectorstore is None or _llm is None:
        return None

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an expert medical assistant. Use the following medical text to answer the question.

Question: {question}

Medical text:
{context}

Extract the following details clearly in plain English (no bullet points, just short paragraphs):
- Symptoms
- Treatment
- Impact on Pregnancy
- Mental Health Impact
- Age group affected

Answer:"""
    )

    retriever = _vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=_llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=False,
    )

# 5. Streamlit UI
st.set_page_config(page_title="Medical Data ChatBot", layout="centered")
st.title("🩺 Medical Data ChatBot")

# Load data once at startup
documents = load_data()
vectorstore = create_vectorstore(documents)
llm = load_llm()
rag_chain = setup_rag_chain(vectorstore, llm)

query = st.text_input("Enter a disease name to extract info:")

if query:
    if rag_chain is None:
        st.error("⚠️ RAG chain is not ready. Check your data, API key, and dependencies.")
    else:
        with st.spinner("🔍 Searching and generating answer..."):
            try:
                response = rag_chain.run(query)
                st.markdown("### 📋 Extracted Information")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
else:
    st.info("Enter a disease name above to get started.")

st.markdown("---")
st.caption("Powered by Groq Llama 3, FAISS, and sentence‑transformers.")
