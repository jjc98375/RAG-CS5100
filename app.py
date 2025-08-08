"""
Streamlit Chat Interface for Northeastern University RAG System
Run with: streamlit run app.py
"""

import streamlit as st
import time
from pathlib import Path
import sys
import os

# Import your RAG system
from rag import NortheasternRAG

# Page configuration
st.set_page_config(
    page_title="Northeastern University Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Theme toggle function
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Apply theme-based CSS
if st.session_state.dark_mode:
    # Dark mode CSS
    st.markdown("""
    <style>
        /* Dark Mode Theme */
        .stApp {
            background-color: #1a1a1a;
            color: #e0e0e0;
        }
        
        .main-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        .category-card {
            background: #2d2d2d;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #404040;
            margin-bottom: 0.5rem;
            transition: all 0.3s ease;
            color: #e0e0e0;
        }
        
        .category-card:hover {
            border-color: #4a9eff;
            box-shadow: 0 2px 8px rgba(74, 158, 255, 0.3);
            background: #333333;
        }
        
        .info-box {
            background: #2a3f5f;
            border: 1px solid #3d5a80;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            color: #e0e0e0;
        }
        
        .warning-box {
            background: #3d3319;
            border: 1px solid #5c4e29;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            color: #ffd700;
        }
        
        .source-badge {
            background: #3a3a3a;
            color: #e0e0e0;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.85rem;
            margin-right: 0.5rem;
            display: inline-block;
            margin-top: 0.5rem;
            border: 1px solid #505050;
        }
        
        div[data-testid="stSidebar"] {
            background-color: #252525;
            border-right: 1px solid #404040;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #4a9eff 0%, #3d7dd6 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(74, 158, 255, 0.4);
            background: linear-gradient(135deg, #5aa3ff 0%, #4687e0 100%);
        }
        
        .stTextInput > div > div > input {
            background-color: #2d2d2d;
            color: #e0e0e0;
            border: 1px solid #404040;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #4a9eff;
            box-shadow: 0 0 0 1px #4a9eff;
        }
        
        .theme-toggle {
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 999;
            background: #3a3a3a;
            border: 1px solid #505050;
            border-radius: 25px;
            padding: 0.5rem 1rem;
            color: #e0e0e0;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .theme-toggle:hover {
            background: #4a4a4a;
            transform: scale(1.05);
        }
        
        /* Message styling for dark mode */
        .stChatMessage {
            background-color: #2d2d2d;
            border: 1px solid #404040;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        
        [data-testid="stChatMessageContent"] {
            color: #e0e0e0;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #2d2d2d;
            color: #e0e0e0;
        }
        
        /* Info messages */
        .stAlert {
            background-color: #2a3f5f;
            color: #e0e0e0;
            border: 1px solid #3d5a80;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    # Light mode CSS
    st.markdown("""
    <style>
        /* Light Mode Theme */
        .stApp {
            background-color: #f8f9fa;
            color: #212529;
        }
        
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .category-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #dee2e6;
            margin-bottom: 0.5rem;
            transition: all 0.3s ease;
            color: #212529;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        .category-card:hover {
            border-color: #667eea;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
            transform: translateY(-2px);
        }
        
        .info-box {
            background: #e7f3ff;
            border: 1px solid #b3d9ff;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            color: #004085;
        }
        
        .warning-box {
            background: #fff3cd;
            border: 1px solid #ffeeba;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            color: #856404;
        }
        
        .source-badge {
            background: #f0f2f5;
            color: #495057;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.85rem;
            margin-right: 0.5rem;
            display: inline-block;
            margin-top: 0.5rem;
            border: 1px solid #dee2e6;
        }
        
        div[data-testid="stSidebar"] {
            background-color: white;
            border-right: 1px solid #dee2e6;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            background: linear-gradient(135deg, #7689ee 0%, #8759a8 100%);
        }
        
        .stTextInput > div > div > input {
            background-color: white;
            color: #212529;
            border: 1px solid #ced4da;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.25);
        }
        
        .theme-toggle {
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 999;
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 25px;
            padding: 0.5rem 1rem;
            color: #212529;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .theme-toggle:hover {
            background: #f8f9fa;
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        /* Message styling for light mode */
        .stChatMessage {
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        [data-testid="stChatMessageContent"] {
            color: #212529;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            color: #212529;
        }
        
        /* Info messages */
        .stAlert {
            background-color: #e7f3ff;
            color: #004085;
            border: 1px solid #b3d9ff;
        }
    </style>
    """, unsafe_allow_html=True)

# Theme toggle button
theme_icon = "🌙" if not st.session_state.dark_mode else "☀️"
theme_text = "Dark Mode" if not st.session_state.dark_mode else "Light Mode"

# Add theme toggle in the top right
col1, col2, col3 = st.columns([8, 1, 1])
with col3:
    if st.button(f"{theme_icon} {theme_text}", key="theme_toggle"):
        toggle_theme()
        st.rerun()

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.loading_complete = False

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'current_category' not in st.session_state:
    st.session_state.current_category = None

if 'category_locked' not in st.session_state:
    st.session_state.category_locked = False

# Category information
CATEGORIES = {
    "visa": {
        "name": "Visa & Immigration",
        "icon": "🛂",
        "description": "F-1 visa, I-20, SEVIS, travel documents"
    },
    "employment": {
        "name": "Employment",
        "icon": "💼",
        "description": "On-campus work, CPT, OPT, work authorization"
    },
    "arrival": {
        "name": "Arrival & Check-in",
        "icon": "✈️",
        "description": "Port of entry, I-94, arrival procedures"
    },
    "billing": {
        "name": "Billing & Payments",
        "icon": "💳",
        "description": "Tuition payments, billing statements, payment methods"
    },
    "course": {
        "name": "Course Registration",
        "icon": "📚",
        "description": "Course enrollment, registration procedures"
    },
    "scholarships": {
        "name": "Scholarships",
        "icon": "🎓",
        "description": "Financial aid, scholarships, grants"
    },
    "tuition": {
        "name": "Tuition & Fees",
        "icon": "💰",
        "description": "Tuition costs, fees, charges"
    },
    "forms": {
        "name": "Forms & Documents",
        "icon": "📋",
        "description": "Required forms and documentation"
    }
}

def clean_response_text(text):
    """Clean up improperly formatted markdown in response text"""
    import re
    
    # Fix common formatting issues
    # Remove single asterisks that aren't properly paired
    text = re.sub(r'(?<!\*)\*(?!\*)', '', text)
    
    # Fix dollar amounts that might have asterisks
    text = re.sub(r'\$(\d+)\*+', r'$\1', text)
    
    # Fix numbers with asterisks
    text = re.sub(r'(\d+)\*+(\s)', r'\1\2', text)
    
    # Clean up any remaining standalone asterisks
    text = re.sub(r'\s\*\s', ' ', text)
    
    # Fix broken bullet points (single asterisk at start of line)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # If line starts with "* " it's likely a bullet point, keep it
        if line.strip().startswith('* '):
            cleaned_lines.append(line)
        else:
            # Remove any other standalone asterisks
            line = re.sub(r'(?<=[a-zA-Z0-9])\*(?=[a-zA-Z0-9])', '', line)
            cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    return text

def find_source_file(filename, category):
    """Find the full path of a source file"""
    if not category or not filename:
        return None
    
    # Get the folder name for the category
    folder_map = {
        "visa": "Visa",
        "employment": "Employment",
        "arrival": "Arrival",
        "billing": "Billing",
        "course": "Course Registration",
        "scholarships": "Scholarships",
        "tuition": "Tuition&Fees",
        "forms": "forms"
    }
    
    if category not in folder_map:
        return None
    
    # Build the path
    base_path = Path("./northeastern_docs") / folder_map[category]
    
    # Search for the file (including subdirectories)
    for pdf_file in base_path.glob(f"**/{filename}"):
        return str(pdf_file)
    
    return None

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system (cached to avoid reloading)"""
    with st.spinner("🚀 Initializing Northeastern University RAG System..."):
        rag = NortheasternRAG(
            base_directory="./northeastern_docs",
            model_name="llama3.2",
            embedding_model="nomic-embed-text",
            persist_directory="./northeastern_unified_db"
        )
        
        # Check if database exists
        db_status = rag.check_database_status()
        if not db_status["exists"]:
            st.info("Building vector database from documents... This may take a few minutes on first run.")
            rag.process_all_folders()
        else:
            st.success("Loading existing vector database...")
            rag.load_vector_database()
        
        return rag

# Main app header
st.markdown("""
<div class="main-header">
    <h1>🎓 Northeastern University International Student Assistant</h1>
    <p style="font-size: 1.1rem; margin-top: 0.5rem;">Get answers to your questions about visa, employment, enrollment, and more!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for category selection
with st.sidebar:
    st.markdown("## 📁 Select a Category")
    st.markdown("Choose a category to ask questions about:")
    
    # Display categories as buttons
    for cat_key, cat_info in CATEGORIES.items():
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown(f"<div style='font-size: 1.5rem; text-align: center; margin-top: 0.5rem;'>{cat_info['icon']}</div>", unsafe_allow_html=True)
        with col2:
            if st.button(
                f"{cat_info['name']}", 
                key=f"cat_{cat_key}",
                help=cat_info['description'],
                disabled=st.session_state.category_locked
            ):
                if st.session_state.rag_system:
                    if st.session_state.rag_system.set_category(cat_key):
                        st.session_state.current_category = cat_key
                        st.session_state.messages.append({
                            "role": "system",
                            "content": f"Category changed to: {cat_info['name']}"
                        })
                        st.rerun()
    
    st.markdown("---")
    
    # Show current category
    if st.session_state.current_category:
        current = CATEGORIES[st.session_state.current_category]
        st.markdown(f"""
        <div class="info-box">
            <strong>Current Category:</strong><br>
            {current['icon']} {current['name']}
        </div>
        """, unsafe_allow_html=True)
        
        # Lock/Unlock category during conversation
        lock_label = "🔓 Unlock Category" if st.session_state.category_locked else "🔒 Lock Category"
        if st.button(lock_label):
            st.session_state.category_locked = not st.session_state.category_locked
            st.rerun()
    else:
        st.markdown("""
        <div class="warning-box">
            ⚠️ Please select a category to start asking questions
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.category_locked = False
        st.rerun()
    
    # System status
    if st.session_state.rag_system:
        with st.expander("📊 System Status"):
            stats = st.session_state.rag_system.get_statistics()
            st.write(f"**Database:** {'✅ Loaded' if stats['database_exists'] else '❌ Not loaded'}")
            st.write(f"**Documents:** {stats.get('total_documents', 'N/A')}")
            st.write(f"**Model:** llama2")

# Initialize RAG system
if st.session_state.rag_system is None:
    st.session_state.rag_system = initialize_rag_system()
    st.session_state.loading_complete = True

# Main chat interface
main_container = st.container()

with main_container:
    # Instructions or welcome message
    if len(st.session_state.messages) == 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="category-card">
                <h3>📍 Step 1</h3>
                <p>Select a category from the sidebar that matches your question topic</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="category-card">
                <h3>💬 Step 2</h3>
                <p>Type your question in the chat box below</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="category-card">
                <h3>📖 Step 3</h3>
                <p>Get comprehensive answers from official documents</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Example questions
        st.markdown("### 💡 Example Questions You Can Ask:")
        
        example_cols = st.columns(2)
        with example_cols[0]:
            st.markdown("""
            <div class="category-card">
            <strong>Visa & Immigration:</strong><br>
            • What documents do I need for F-1 visa?<br>
            • How do I pay the SEVIS fee?<br>
            • When should I arrive at the university?<br><br>
            
            <strong>Employment:</strong><br>
            • How many hours can I work on campus?<br>
            • What is CPT and how do I apply?<br>
            • Can I work during summer break?
            </div>
            """, unsafe_allow_html=True)
        
        with example_cols[1]:
            st.markdown("""
            <div class="category-card">
            <strong>Billing & Payments:</strong><br>
            • What payment methods are accepted?<br>
            • When is tuition due?<br>
            • How do I access my billing statement?<br><br>
            
            <strong>Scholarships:</strong><br>
            • What scholarships are available?<br>
            • How do I apply for financial aid?<br>
            • What is the Double Husky Scholarship?
            </div>
            """, unsafe_allow_html=True)

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "system":
        st.info(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources with download links
            if "sources" in message and message["sources"]:
                st.markdown("**📚 Sources:**")
                source_cols = st.columns(min(len(message["sources"]), 3))
                for idx, source in enumerate(message["sources"]):
                    col_idx = idx % len(source_cols)
                    with source_cols[col_idx]:
                        # Create download link for the source PDF
                        if "category" in message:
                            source_path = find_source_file(source, message.get("category"))
                            if source_path and os.path.exists(source_path):
                                with open(source_path, "rb") as file:
                                    st.download_button(
                                        label=f"📄 {source}",
                                        data=file.read(),
                                        file_name=source,
                                        mime="application/pdf",
                                        key=f"download_history_{i}_{idx}"
                                    )
                            else:
                                st.markdown(f"<span class='source-badge'>📄 {source}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span class='source-badge'>📄 {source}</span>", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask your question here...", disabled=not st.session_state.current_category):
    if not st.session_state.current_category:
        st.error("⚠️ Please select a category from the sidebar first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Show typing indicator
            with st.spinner("Thinking..."):
                try:
                    # Query the RAG system
                    result = st.session_state.rag_system.query(prompt)
                    
                    # Extract the answer
                    if "error" in result:
                        response = f"❌ {result['error']}"
                        sources = []
                    else:
                        # Clean the response to fix formatting issues
                        response = clean_response_text(result['answer'])
                        sources = [doc['filename'] for doc in result.get('source_documents', [])]
                    
                    # Display response with typing effect (optional)
                    message_placeholder.markdown(response)
                    
                    # Show sources with download links
                    if sources:
                        st.markdown("**📚 Sources:**")
                        source_cols = st.columns(min(len(sources), 3))
                        for idx, source in enumerate(sources):
                            col_idx = idx % len(source_cols)
                            with source_cols[col_idx]:
                                # Create download link for the source PDF
                                source_path = find_source_file(source, st.session_state.current_category)
                                if source_path and os.path.exists(source_path):
                                    with open(source_path, "rb") as file:
                                        st.download_button(
                                            label=f"📄 {source}",
                                            data=file.read(),
                                            file_name=source,
                                            mime="application/pdf",
                                            key=f"download_{prompt}_{idx}"
                                        )
                                else:
                                    st.markdown(f"<span class='source-badge'>📄 {source}</span>", unsafe_allow_html=True)
                    
                    # Add to message history with category info
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources,
                        "category": st.session_state.current_category
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: {'#999' if st.session_state.dark_mode else '#666'};'>
    <p>💡 Tip: Lock the category to prevent accidental changes during your conversation</p>
    <p>Powered by RAG System with Ollama & ChromaDB | Northeastern University Documentation</p>
</div>
""", unsafe_allow_html=True)