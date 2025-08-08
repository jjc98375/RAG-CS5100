# 🎓 Northeastern University RAG Chatbot System

An intelligent question-answering system for Northeastern University international students, powered by RAG (Retrieval-Augmented Generation) with local LLMs.

## 📋 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [File Structure](#file-structure)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
- [Features](#features)
- [Troubleshooting](#troubleshooting)

## 🌟 Overview

This system helps international students get instant answers about:
- 🛂 Visa & Immigration procedures
- 💼 Employment regulations
- 📚 Course Registration
- 💰 Tuition & Fees
- 🎓 Scholarships
- ✈️ Arrival procedures
- 💳 Billing & Payments
- 📋 Forms & Documents

**Key Features:**
- 100% Local - No API keys required
- Privacy-focused - All data stays on your machine
- Category-based search for accurate answers
- Light/Dark mode UI
- Downloadable source PDFs

## 🏗️ System Architecture

```
Your Computer (Everything Local)
├── Ollama (Local LLM)
│   ├── llama2/llama3.2 (Answer generation)
│   └── nomic-embed-text (Document embeddings)
│
├── ChromaDB (Vector Database)
│   └── Stores PDF content as searchable vectors
│
├── PDF Documents (./northeastern_docs/)
│   ├── Visa/
│   ├── Employment/
│   ├── Scholarships/
│   └── [Other categories...]
│
└── Streamlit Web UI (localhost:8501)
    └── Interactive chat interface
```

## 📁 File Structure

```
RAG_cs5100/
│
├── app.py                      # Main Streamlit web application
├── rag.py                      # RAG system core logic
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── northeastern_docs/          # PDF documents folder
│   ├── Arrival/               # Arrival-related PDFs
│   ├── Billing/               # Billing-related PDFs
│   ├── Course Registration/   # Course registration PDFs
│   ├── Employment/            # Employment-related PDFs
│   ├── forms/                 # Various forms
│   ├── Scholarships/          # Scholarship information
│   ├── Tuition&Fees/         # Tuition and fees PDFs
│   └── Visa/                  # Visa-related documents
│
└── northeastern_unified_db/    # Vector database (auto-generated)
    └── [ChromaDB files]
```

### File Descriptions:

| File | Purpose |
|------|---------|
| **app.py** | Streamlit web interface with chat UI, theme toggle, and category selection |
| **rag.py** | Core RAG system: PDF loading, vector DB creation, query processing |
| **requirements.txt** | All Python package dependencies |
| **northeastern_docs/** | Your PDF documents organized by category |
| **northeastern_unified_db/** | Auto-generated vector database (created on first run) |

## 🔧 Prerequisites

### 1. System Requirements
- **OS**: macOS, Linux, or Windows
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space
- **Python**: 3.8 or higher

### 2. Ollama Installation
Ollama runs LLMs locally on your machine.

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [https://ollama.com/download/windows](https://ollama.com/download/windows)

### 3. Pull Required Models
```bash
# Pull the language model (choose one)
ollama pull llama2        # Good performance (3.8GB)
# OR
ollama pull llama3.2      # Better quality (2GB)

# Pull the embedding model (required)
ollama pull nomic-embed-text
```

### 4. Verify Ollama is Running
```bash
ollama list  # Should show your downloaded models
```

## 📦 Installation Guide

### Step 1: Clone or Download the Project
```bash
# If using git
git clone <your-repository>
cd RAG_cs5100

# Or simply extract the ZIP file
```

### Step 2: Set Up Python Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Python Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

**What gets installed:**
- `streamlit` - Web UI framework
- `langchain` - LLM orchestration
- `chromadb` - Vector database
- `pypdf` - PDF processing
- Other supporting libraries

### Step 4: Add Your PDF Documents
Place your PDF files in the appropriate folders under `northeastern_docs/`:
```
northeastern_docs/
├── Visa/           # F-1 visa, I-20, SEVIS documents
├── Employment/     # CPT, OPT, on-campus work rules
├── Scholarships/   # Scholarship information
└── [etc...]
```

### Step 5: Start Ollama Service
```bash
# Make sure Ollama is running
ollama serve  # Run in a separate terminal
```

## 🚀 Usage

### Starting the Application

1. **Ensure Ollama is running:**
```bash
ollama serve
```

2. **Run the Streamlit app:**
```bash
streamlit run app.py
```

3. **Open your browser:**
- Navigate to `http://localhost:8501`
- The browser should open automatically

### Using the Chatbot

1. **Select a Category:**
   - Click on a category in the sidebar (e.g., "Visa & Immigration")
   - The system will only search within that category's documents

2. **Ask Your Question:**
   - Type your question in the chat box
   - Example: "What documents do I need for F-1 visa?"

3. **Get Answers with Sources:**
   - The system provides detailed answers
   - Source PDFs are shown and downloadable

4. **Toggle Theme:**
   - Click the theme button (🌙/☀️) in the top-right corner

5. **Lock Category (Optional):**
   - Click "🔒 Lock Category" to prevent accidental changes

## ✨ Features

### 🎨 User Interface
- **Light/Dark Mode**: Toggle between themes for comfort
- **Category-Based Search**: Ensures relevant, accurate answers
- **Download Sources**: Click to download referenced PDFs
- **Chat History**: Full conversation history during session
- **System Status**: View database and model information

### 🔍 Search Capabilities
- **Semantic Search**: Understands context and meaning
- **Multi-Document Synthesis**: Combines information from multiple PDFs
- **Category Filtering**: Searches only relevant documents

### 🔒 Privacy & Security
- **100% Local**: No data sent to external servers
- **No API Keys**: No subscription or API costs
- **Data Control**: All your documents stay on your machine

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. "Ollama not found" Error
```bash
# Check if Ollama is installed
ollama --version

# If not installed, install it:
curl -fsSL https://ollama.com/install.sh | sh
```

#### 2. "Model not found" Error
```bash
# Pull the required models
ollama pull llama2
ollama pull nomic-embed-text
```

#### 3. "Port 8501 already in use"
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

#### 4. "No vector database found"
- The database is created automatically on first run
- Make sure PDF documents are in `northeastern_docs/` folders
- Delete `northeastern_unified_db/` folder to rebuild

#### 5. Formatting Issues in Responses
- The app includes automatic text cleaning
- Consider using `llama3.2` for better output quality

#### 6. Slow Response Times
- First query takes longer (model loading)
- Consider using a smaller model
- Ensure sufficient RAM available

### Performance Tips

1. **Use llama3.2** for better quality and speed:
```python
# In rag.py, change:
model_name="llama3.2"  # Instead of llama2
```

2. **Reduce chunk size** for faster processing:
```python
# In rag.py:
chunk_size=500  # Smaller chunks
```

3. **Limit search results**:
```python
# In rag.py:
search_kwargs={"k": 3}  # Fewer documents
```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Ensure PDFs are in the correct folders
4. Check Ollama is running (`ollama serve`)

## 📄 License

This project is for educational purposes. Ensure you have the right to use any PDF documents you add to the system.

---

## 🚀 Quick Start Summary

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull models
ollama pull llama2
ollama pull nomic-embed-text

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Add PDFs to northeastern_docs/ folders

# 5. Start Ollama
ollama serve

# 6. Run the app
streamlit run app.py

# 7. Open browser to http://localhost:8501
```

Enjoy your intelligent document assistant! 🎓