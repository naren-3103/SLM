import streamlit as st
import os

# --- Page Config ---
st.set_page_config(page_title="SLM Toolkit Studio", page_icon="🚀", layout="wide")

# Internal modules
from app.config import DOCUMENT_PATH
from rag_pipeline.index_documents import build_vector_db

from services.mcq_generator import MCQGenerator
from services.notes_generator import NotesGenerator
from services.rag import RAGPipeline
from services.summarizer import TextSummarizer
from services.translator import Translator
from services.generator import TextGenerator

# Ensure document path exists
os.makedirs(DOCUMENT_PATH, exist_ok=True)

# Cache initialization of services to avoid re-instantiation warnings in UI 
# Note: ModelLoader handles the LLM Singleton natively.
@st.cache_resource
def get_mcq_gen():
    return MCQGenerator()

@st.cache_resource
def get_notes_gen():
    return NotesGenerator()

@st.cache_resource
def get_rag():
    # Only instantiated when RAG is selected or document is rebuilt
    return RAGPipeline(force_rebuild=False)

@st.cache_resource
def get_summarizer():
    return TextSummarizer()

@st.cache_resource
def get_translator():
    return Translator()

@st.cache_resource
def get_text_gen():
    return TextGenerator()

# --- SIDEBAR & DOCUMENT UPLOAD ---
st.sidebar.title("🗂️ Document Manager")
existing_docs = os.listdir(DOCUMENT_PATH)

if existing_docs:
    st.sidebar.markdown("**Current Workspace Documents:**")
    for doc in existing_docs:
        if doc.endswith(".pdf") or doc.endswith(".txt"):
            st.sidebar.caption(f"📄 {doc}")
else:
    st.sidebar.info("No documents found in workspace.")

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload New PDF to Workspace", type=["pdf"])

if uploaded_file is not None:
    # Save the file to data/documents/
    save_path = os.path.join(DOCUMENT_PATH, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.sidebar.success(f"Successfully saved `{uploaded_file.name}`")
    
    with st.spinner("Rebuilding Vector Knowledge Base... This may take a minute ⏳"):
        build_vector_db(force_rebuild=True)
        # Clear the RAG service cache to force reload the newest Vector Store
        get_rag.clear() 
        get_mcq_gen.clear()
        get_notes_gen.clear()
    
    st.sidebar.success("Database Indexed!")

st.sidebar.markdown("---")

# --- SERVICES ---
st.sidebar.title("🛠️ Services")
options = [
    "📝 Question Generation", 
    "🎓 Notes Generation", 
    "💬 Ask Document (RAG)", 
    "✂️ Text Summarization", 
    "🌍 Translation", 
    "⚡ General Text Generation"
]
selected_service = st.sidebar.radio("Choose App Service", options)


# --- MAIN CONTENT DYNAMICALLY SWITCHES ---

st.title(selected_service)
st.markdown("---")

# 1. Question Generation
if selected_service == "📝 Question Generation":
    st.markdown("Generate multiple choice questions dynamically utilizing your documents.")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    topic = col1.text_input("📚 Topic / Concept", placeholder="e.g., Photosynthesis")
    count = col2.number_input("🔢 Number of Questions", min_value=1, max_value=20, value=3)
    difficulty = col3.selectbox("🎚️ Difficulty", ["easy", "medium", "difficult"])
    
    if st.button("Generate Questions 🚀", use_container_width=True):
        if not topic.strip():
            st.error("Please enter a topic.")
        else:
            with st.spinner(f"Generating {count} {difficulty} questions about {topic}... 🧠"):
                gen = get_mcq_gen()
                output = gen.generate_questions(topic, count, difficulty)
                st.success("Generation Complete!")
                st.markdown(output)

# 2. Notes Generation
elif selected_service == "🎓 Notes Generation":
    st.markdown("Produce student-friendly, easy to digest notes from your vector database.")
    topic = st.text_input("📚 Subject / Topic", placeholder="e.g., Quantum Mechanics")
    
    if st.button("Create Notes 🚀", use_container_width=True):
        if not topic.strip():
            st.error("Please enter a valid topic.")
        else:
            with st.spinner(f"Drafting notes for '{topic}'... 🧠"):
                gen = get_notes_gen()
                output = gen.generate_notes(topic)
                st.success("Notes Created!")
                st.markdown(output)

# 3. Ask Document (RAG)
elif selected_service == "💬 Ask Document (RAG)":
    st.markdown("Execute a direct Q&A interaction based on your ingested documents.")
    question = st.text_input("❓ Question", placeholder="What are the key points in the quarterly report?")
    
    if st.button("Query Knowledge Base 🔎", use_container_width=True):
        if not question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Searching documents and inferencing models... 🧠"):
                rag = get_rag()
                result = rag.ask(question)
                st.success("Answer Loaded!")
                st.write(f"**Answer:** \n\n {result['answer']}")
                
                if result.get("sources"):
                    with st.expander("📚 View Sources & Citations"):
                        for i, src in enumerate(result["sources"], start=1):
                            st.write(f"**[{i}]** {src['source']} — Page {src['page']} (Score: {src['score']:.3f})")

# 4. Text Summarization
elif selected_service == "✂️ Text Summarization":
    st.markdown("Extract robust summaries concisely. (Max limit: `10,000` characters)")
    text = st.text_area("📝 Text to Summarize", height=250, max_chars=10000)
    
    if st.button("Summarize 🚀", use_container_width=True):
        if not text.strip():
            st.error("Text area cannot be empty.")
        else:
            with st.spinner("Processing Summary... 🧠"):
                summarizer = get_summarizer()
                summary = summarizer.summarize_text(text)
                st.success("Done!")
                st.write("### Summary Result:")
                st.code(summary, language="markdown")

# 5. Translation
elif selected_service == "🌍 Translation":
    st.markdown("Translate small chunks to your target language. (Max limit: `1,000` characters)")
    target_lang = st.text_input("🌐 Target Language", placeholder="e.g. Spanish, French, German")
    text = st.text_area("📝 Text to Translate", height=150, max_chars=1000)
    
    if st.button("Translate 🚀", use_container_width=True):
        if not text.strip() or not target_lang.strip():
            st.error("Text and target language cannot be empty.")
        else:
            with st.spinner(f"Translating to {target_lang}... 🧠"):
                translator = get_translator()
                translated = translator.translate(text, target_lang)
                st.success("Translating Complete!")
                st.write(f"### {target_lang} Output:")
                st.code(translated, language="markdown")

# 6. General Text Generation
elif selected_service == "⚡ General Text Generation":
    st.markdown("Provide raw text prompts to the SLM manually.")
    prompt = st.text_area("✍️ Text Prompt", height=150, placeholder="Write a short blog post about AI in Education...")
    
    if st.button("Generate Output 🚀", use_container_width=True):
        if not prompt.strip():
            st.error("Prompt cannot be empty.")
        else:
            with st.spinner("Generating raw output... 🧠"):
                gen = get_text_gen()
                output = gen.generate(prompt)
                st.success("Sequence Generation Complete!")
                st.code(output, language="markdown")
