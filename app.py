# app.py
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="NCERT Buddy — Class 5 Tutor",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Secrets — works locally via .streamlit/secrets.toml AND on Cloud
# ---------------------------------------------------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ---------------------------------------------------------------------
# Cached resources (load once per session)
# ---------------------------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory="./chroma_data",
        collection_name="education",
        embedding_function=load_embeddings(),
    )

@st.cache_resource
def load_llm():
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=GROQ_API_KEY)

vectorstore = load_vectorstore()
llm = load_llm()

# ---------------------------------------------------------------------
# Retrievers per subject
# ---------------------------------------------------------------------
retrievers = {
    "English 📖": vectorstore.as_retriever(search_kwargs={"k": 4, "filter": {"subject": "english"}}),
    "Science 🔬": vectorstore.as_retriever(search_kwargs={"k": 4, "filter": {"subject": "science"}}),
    "All subjects": vectorstore.as_retriever(search_kwargs={"k": 4}),
}

prompt = ChatPromptTemplate.from_template(
    """You are a friendly teacher for a 5th grade student.
Use ONLY the context below to answer. If the answer is not in the context, politely say you don't know.

Context:
{context}

Question:
{question}

Reply in 2–4 short sentences, simple words, and one small example if helpful.
"""
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def format_context(docs):
    return "\n\n".join(d.page_content for d in docs)

def extract_sources(docs):
    seen, out = set(), []
    for d in docs:
        fname = d.metadata.get("file_name", "unknown.pdf")
        page  = d.metadata.get("page_label", "?")
        subj  = d.metadata.get("subject", "?")
        key = (fname, page)
        if key not in seen:
            seen.add(key)
            out.append({"file": fname, "page": page, "subject": subj})
    return out

def answer(question: str, subject_label: str):
    retriever = retrievers[subject_label]
    docs = retriever.invoke(question)
    if not docs:
        return "I couldn't find anything in the textbook about that. Try rephrasing!", []

    context = format_context(docs)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})
    return response, extract_sources(docs)

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/books.png", width=64)
    st.title("📘 NCERT Buddy")
    st.caption("A friendly tutor for Class 5 English & Science.")

    subject_label = st.selectbox(
        "📚 Choose subject",
        options=list(retrievers.keys()),
        index=2,
    )

    st.divider()
    st.markdown("**💡 Try asking:**")
    examples = [
        "What is evaporation?",
        "Why do plants need sunlight?",
        "What is the poem Papa's Spectacles about?",
        "Why does the Moon change shape?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_question = ex

    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by Groq · MiniLM embeddings · ChromaDB")

# ---------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

st.title("📘 Ask NCERT Buddy")
st.caption("I answer using Class 5 textbooks. Ask anything from English or Science!")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "📘"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['file']}** — page {s['page']} · *{s['subject']}*")

# Handle example-button click OR chat input
question = st.session_state.pending_question
if question is None:
    question = st.chat_input("Ask a question from your textbook...")

if question:
    st.session_state.pending_question = None

    # User bubble
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)

    # Assistant bubble
    with st.chat_message("assistant", avatar="📘"):
        with st.spinner("Thinking like a teacher…"):
            reply, sources = answer(question, subject_label)
        st.markdown(reply)
        if sources:
            with st.expander(f"📚 Sources ({len(sources)})"):
                for s in sources:
                    st.markdown(f"- **{s['file']}** — page {s['page']} · *{s['subject']}*")

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "sources": sources,
    })