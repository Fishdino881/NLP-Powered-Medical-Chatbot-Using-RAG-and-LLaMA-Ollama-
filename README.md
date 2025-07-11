🧠 NLP-Powered Medical Chatbot using RAG and LLaMA (Ollama)

This project is a privacy-preserving, NLP-driven chatbot designed to extract structured information about medical conditions from unstructured text. It utilizes a Retrieval-Augmented Generation (RAG) architecture combining semantic search with local language generation through Ollama’s LLaMA model.

Built using LangChain, FAISS, HuggingFace Embeddings, and a fully local Streamlit interface, this chatbot runs entirely offline—making it ideal for environments where data privacy, low infrastructure, or internet independence is essential.

---

🚀 Key Features

- Accepts free-text disease queries from users
- Extracts and presents:
  - ✔️ Symptoms
  - 💊 Treatment options
  - 🤰 Impact on pregnancy
  - 🧠 Mental health implications
  - 👶 Age groups affected
- Fully local NLP pipeline (no cloud APIs)
- Vector similarity search using FAISS
- Language generation using LLaMA via Ollama
- Designed with a clear, minimal Streamlit UI
- Reproducible and easy to extend for new medical domains

---

📚 How It Works (NLP Pipeline Explained)

1. Data Ingestion:  
   Loads raw medical data (e.g., disease descriptions) from `medical_data.csv` using LangChain’s `CSVLoader`.

2. Text Chunking:
   Long documents are broken into manageable segments using `CharacterTextSplitter` for effective vectorization and retrieval.

3. Semantic Embeddings:
   Text chunks are converted into dense vector representations using **HuggingFace’s `sentence-transformers`** model (`all-MiniLM-L6-v2`).

4. Vector Indexing:
   Vectors are stored in a **FAISS** index, enabling fast and accurate retrieval based on semantic similarity.

5. Query Input:
   The user enters a disease name or related term (e.g., "diabetes").

6. Context Retrieval:
   FAISS returns the most semantically relevant text chunks for the given query.

7. LLM Generation via Ollama:
   Retrieved context is passed to LLaMA (running locally through Ollama), which generates structured answers using a custom prompt template.

8. Response Display:
   The final result is rendered in the Streamlit app as clear, plain-language medical information.

---

📁 Project Structure

```

📦 medical-nlp-chatbot/
├── app.py                 # Streamlit frontend and NLP logic
├── medical\_data.csv       # Medical knowledge base (raw text)
├── requirements.txt       # Python dependencies

````

---

🛠️ Installation & Usage

1. Clone the Repository

```bash
git clone https://github.com/yourusername/medical-nlp-chatbot.git
cd medical-nlp-chatbot
````

2. Install Dependencies

Ensure Python 3.8+ is installed, then run:

```bash
pip install -r requirements.txt
```

3. Pull the LLaMA Model via Ollama

Install Ollama from [ollama.com](https://ollama.com) if you haven’t already:

```bash
ollama pull llama3.2:latest
```

4. Launch the App

```bash
streamlit run app.py
```

The app will open in your browser. Type a disease name to receive structured medical information.

---

💻 Example Interaction

User Input:
`Enter disease name: asthma`

Response Output:

* Symptoms: Shortness of breath, wheezing, coughing
* Treatment: Inhaled corticosteroids, bronchodilators
* Impact on Pregnancy: May require dose adjustments
* Mental Health Impact: Anxiety or panic triggered by breathlessness
* Age Group Affected: Children, adolescents, and adults

---

🔐 Data Privacy

This chatbot runs **entirely offline** on the user’s machine. All NLP processing, vector indexing, and language generation occur locally. This ensures:

* No third-party API calls
* No data storage or transmission
* Compliance with privacy-sensitive workflows
* Usability in air-gapped, low-resource, or clinical environments

---

🎯 Use Cases

* ✅ Healthcare & life science research assistants
* ✅ Medical education tools for students and universities
* ✅ Rural and offline medical triage tools
* ✅ Prototype for privacy-first healthcare NLP applications
* ✅ Teaching Retrieval-Augmented Generation (RAG) concepts

---

🔧 Future Enhancements

* PDF/document ingestion pipeline
* Live chat interface with memory
* Multilingual support
* Fine-tuned medical QA models
* Audio-based query input
* Real-time vector index updates from new data

---

📄 License

This project is intended strictly for research and educational purposes. It is not certified for clinical or diagnostic use.

---

🤝 Contributing

Contributions are welcome! If you'd like to fix bugs, add features, or improve model performance:

1. Fork this repo
2. Create a feature branch
3. Open a pull request with detailed context

Let's build responsible, accessible, and privacy-first NLP solutions for healthcare together.

---

```

---

