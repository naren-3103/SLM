# SLM AI Toolkit

## Introduction to Small Language Models (SLMs)
Small Language Models (SLMs) are compact iterations of traditional Large Language Models (LLMs), designed to run efficiently with fewer parameters. Despite their reduced size, modern SLMs deliver impressive reasoning capabilities while maintaining a significant edge in performance speed and hardware requirements.

For **Retrieval-Augmented Generation (RAG)** use cases, SLMs offer key advantages:
- **Optimized Hardware Usage:** We avoid bulky offline models that consume massive amounts of RAM. Instead, the model is loaded efficiently via Hugging Face, balancing resource usage while keeping data internal to the application's runtime.
- **Cost-Effective:** Because they require far less compute power, operational and inference costs are strictly kept to a minimum compared to calling paid cloud LLMs.
- **Fast Inference:** SLMs generate answers quickly, ensuring real-time responsiveness when querying local documents, extracting context, or summarizing insights.

---

## Core Capabilities

### 📖 Retrieval-Augmented Generation (RAG)
The RAG pipeline is the cornerstone of this toolkit, allowing you to reliably query local documents. It connects document chunking, embeddings, and a specialized Vector Database to root the SLM's answers in factual context.
* Features **Hybrid Retrieval** (combining sparse BM25 and dense vector search) for accurate context fetching.
* Incorporates **MMR reranking** to diversify results.
* Highlights in-text source citations based on the context retrieved, heavily reducing hallucinations.

### 📝 Notes Generation
Automatically distills substantial context into structural, student-friendly study materials.
* Leverages the RAG pipeline to pull context on specific subjects.
* Instructs the SLM to output structured markdown with targeted headers, bullet lists, and relevant examples directly from the ingested texts.

### ❓ Question Generation (MCQs)
An interactive learning and evaluation module that formulates Multiple-Choice Questions from your data.
* Capable of generating context-grounded MCQs based on specific difficulty parameters (e.g., easy, medium, difficult) and predefined quantities.
* Ensures four distinct options per question.

*(Other tools in the suite include document summarization and text translation)*

### 🤖 Lightweight Model Backbone
The application is pre-configured to use **`Mistral-7B-Instruct-v0.2`** (quantized version fetched securely using `hf_hub_download`) loaded dynamically from Hugging Face. This specific model is selected because of its exceptional performance-to-size ratio. Despite being highly lightweight and avoiding severe RAM bottlenecks via 4-bit quantization, it consistently rivals much larger, traditional LLMs in formatting, logical structuring, and reasoning capabilities vital for RAG.

---

## Codebase Overview

The project is structured modularly to separate the frontend interface, core RAG mechanics, and specialized services:

- **`data/`**: The local storage directory where you should place all your source documents (PDFs, TXT, etc.) for ingestion.
- **`rag_pipeline/`**: The core mechanics of the RAG system. It handles loading documents, semantic chunking, generating embeddings (`embedding.py`), and managing the vector database (`vector_store.py` and `index_documents.py`).
- **`services/`**: Contains execution logic for specialized tasks.
  - `rag.py`: Orchestrates hybrid retrieval and context injection into model prompts.
  - `notes_generator.py`: Applies specific instruction generation for creating structured study notes from retrieved context.
  - `questions.py` / MCQ features: Formulates targeted MCQs with varying difficulty levels based on the vector context.
- **`models/`**: Manages the local SLM lifecycle (`model_loader.py`), keeping the language model in memory for fast execution and reducing load times.
- **`streamlit_app.py`**: The unified frontend interface that lets users interact visually with the backend services.

---

## Step-by-Step Workflow Execution

Follow these steps to initialize and run the SLM AI toolkit locally.

### Step 1: Document Ingestion
Place any educational materials, PDFs, or raw text files you want to query against strictly inside the `data/` folder. The system will look here for the knowledge base.

### Step 2: Model Setup
Configure any necessary environment variables (like a Hub token) in the `.env` or `app/config.py` file. The application is designed to automatically fetch the lightweight model from Hugging Face dynamically. You do *not* need to manually download or store heavy offline `.gguf` files, significantly reducing setup time and system RAM requirements.

### Step 3: Run the Application
You have two execution methods depending on your needs.

**Method A: Streamlit Web UI (Recommended)**
The primary, unified workflow is streamlined through a graphical interface.
1. Open your terminal at the root of the project.
2. Start the application by running:
   ```bash
   streamlit run streamlit_app.py
   ```
3. **Automatic Indexing:** Once booted, the backend (`index_documents.py`) will automatically fetch your files from `data/`, generate chunks, create vector embeddings, and save the indexed Vector database locally. 

**Method B: Terminal CLI (Testing)**
A Command-Line Interface is included mainly for testing and developer verification. 
1. Run the core CLI entrypoint:
   ```bash
   python main.py
   ```
2. Use this stripped-down terminal environment to test model inference, logic routing, and vector retrieval without needing to load the frontend web layer.

### Step 4: Interact with Services
Go to your browser at `http://localhost:8501`.
1. **Choose a Feature:** Use the sidebar to select between RAG (Q&A), Notes Generation, or Question Generation.
2. **Execute Actions:** 
   - *For RAG*: Ask specific questions regarding the ingested files.
   - *For Notes*: Provide a topic and get structured markdown study notes.
   - *For MCQs*: Specify the quantity and difficulty to generate an interactive quiz.
3. The underlying service handles context retrieval, routes the prompt to the `ModelLoader`, and streams the output directly to the UI.

---

## Installation 

Start by creating a virtual environment and installing the core dependencies:
```bash
pip install -r requirements.txt
```

### Important Note on `llama-cpp-python`
`llama-cpp-python` is a critical backend dependency. If the standard pip installation of `llama-cpp-python` fails (often due to missing C++ compilers), you should use the pre-built CPU wheel. Run the following command instead:

```bash
pip install llama-cpp-python==0.2.90 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary :all:
```
