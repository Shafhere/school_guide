# 📘 CBSE CLASS 5 EASY FRIEND — RAG Chatbot 

A curriculum-grounded chatbot that answers Class 5 English & Science questions using NCERT textbooks.

**Live demo:** [your-app-url].streamlit.app

## Stack
- LlamaIndex (ingestion) · ChromaDB (vector store) · MiniLM (embeddings)
- LangChain (RAG chain) · Groq `openai/gpt-oss-20b` (generation)
- Streamlit (UI)

## Run locally
```bash
pip install -r requirements.txt
python ingest.py
streamlit run app.py
