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

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .category-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }
    .category-card:hover {
        border-color: #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .selected-category {
        background: #667eea;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background: white;
        margin-right: 20%;
    }
    .source-badge {
        background: #f0f0f0;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
    div[data-testid="stSidebar"] {
        background-color: white;
        border-right: 2px solid #e0e0e0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system (cached to avoid reloading)"""
    with st.spinner("🚀 Initializing Northeastern University RAG System..."):
        rag = NortheasternRAG(
            base_directory="./northeastern_docs",
            model_name="llama2",
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
    <p>Get answers to your questions about visa, employment, enrollment, and more!</p>
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
            st.markdown(f"<div style='font-size: 1.5rem; text-align: center;'>{cat_info['icon']}</div>", unsafe_allow_html=True)
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
            **Visa & Immigration:**
            - What documents do I need for F-1 visa?
            - How do I pay the SEVIS fee?
            - When should I arrive at the university?
            
            **Employment:**
            - How many hours can I work on campus?
            - What is CPT and how do I apply?
            - Can I work during summer break?
            """)
        
        with example_cols[1]:
            st.markdown("""
            **Billing & Payments:**
            - What payment methods are accepted?
            - When is tuition due?
            - How do I access my billing statement?
            
            **Scholarships:**
            - What scholarships are available?
            - How do I apply for financial aid?
            - What is the Double Husky Scholarship?
            """)

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "system":
        st.info(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                st.markdown("**📚 Sources:**")
                for source in message["sources"]:
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
                        response = result['answer']
                        sources = [doc['filename'] for doc in result.get('source_documents', [])]
                    
                    # Display response with typing effect (optional)
                    message_placeholder.markdown(response)
                    
                    # Show sources
                    if sources:
                        st.markdown("**📚 Sources:**")
                        for source in sources:
                            st.markdown(f"<span class='source-badge'>📄 {source}</span>", unsafe_allow_html=True)
                    
                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
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
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💡 Tip: Lock the category to prevent accidental changes during your conversation</p>
    <p>Powered by RAG System with Ollama & ChromaDB | Northeastern University Documentation</p>
</div>
""", unsafe_allow_html=True)