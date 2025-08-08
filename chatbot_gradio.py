"""
Northeastern University International Student Chatbot
Category-based Q&A Interface
"""

import chatbot_gradio as gr
import time
from typing import List, Tuple, Optional

# Import the RAG system
from rag import NortheasternRAG

# Category information
CATEGORIES = {
    "arrival": {
        "name": "✈️ Arrival & Immigration",
        "description": "Questions about arriving in the US, I-94, customs, port of entry",
        "examples": ["What documents do I need at the port of entry?", "How do I get my I-94?", "What happens at customs?"]
    },
    "billing": {
        "name": "💳 Billing & Payments",
        "description": "Questions about paying tuition, billing, payment methods",
        "examples": ["How do I pay my tuition?", "What payment methods are accepted?", "When is the payment deadline?"]
    },
    "course_registration": {
        "name": "📚 Course Registration",
        "description": "Questions about registering for courses, enrollment, schedules",
        "examples": ["How do I register for classes?", "What are the registration deadlines?", "How do I drop a course?"]
    },
    "employment": {
        "name": "💼 Employment",
        "description": "Questions about work authorization, CPT, OPT, on-campus employment",
        "examples": ["Can I work on campus?", "What is CPT?", "How many hours can I work?"]
    },
    "forms": {
        "name": "📋 Forms & Documents",
        "description": "Questions about I-20, DS-2019, and other immigration documents",
        "examples": ["How do I get a new I-20?", "What forms do I need for travel?", "How do I request a letter?"]
    },
    "scholarships": {
        "name": "🎓 Scholarships",
        "description": "Questions about financial aid, scholarships, funding opportunities",
        "examples": ["What scholarships are available?", "How do I apply for financial aid?", "Are there graduate scholarships?"]
    },
    "tuition&fees": {
        "name": "💰 Tuition & Fees",
        "description": "Questions about tuition costs, fees, charges",
        "examples": ["How much is tuition?", "What fees do I need to pay?", "Are there additional charges?"]
    },
    "visa": {
        "name": "📄 Visa",
        "description": "Questions about F-1/J-1 visa application, renewal, requirements",
        "examples": ["How do I apply for an F-1 visa?", "What documents do I need for visa interview?", "How do I pay SEVIS fee?"]
    }
}

class NortheasternChatbot:
    def __init__(self):
        """Initialize the chatbot"""
        self.rag_system = None
        self.load_rag_system()
    
    def load_rag_system(self):
        """Load the RAG system"""
        print("Loading RAG system...")
        self.rag_system = NortheasternRAG(
            base_directory="./northeastern_docs",
            model_name="llama2",
            embedding_model="nomic-embed-text",
            persist_directory="./northeastern_unified_db_v2"
        )
        
        # Load existing vector database
        self.rag_system.load_vector_database()
        
        # Setup QA chain
        retriever_kwargs = {
            "search_type": "similarity",
            "search_kwargs": {"k": 6}
        }
        self.rag_system.setup_qa_chain(retriever_kwargs=retriever_kwargs)
        print("RAG system loaded successfully!")
    
    def chat_with_category(self, message: str, category: str, history: List[Tuple[str, str]]):
        """Process chat messages with specific category"""
        if not self.rag_system:
            return history + [(message, "⚠️ System is still loading. Please wait a moment and try again.")]
        
        if not category:
            return history + [(message, "❌ Please select a category first before asking your question.")]
        
        # Get category key from display name
        category_key = None
        for key, info in CATEGORIES.items():
            if info["name"] == category:
                category_key = key
                break
        
        if not category_key:
            return history + [(message, "❌ Invalid category selected.")]
        
        # Start timing
        start_time = time.time()
        
        # Query the RAG system with category filter
        try:
            result = self.rag_system.query(message, filter_category=category_key)
            response_time = time.time() - start_time
            
            # Format response
            answer = result["answer"]
            sources = result["source_documents"]
            
            # Add metadata footer
            footer = f"\n\n---\n📍 Category: {category}\n⏱️ Response time: {response_time:.2f}s\n📚 Sources: {len(sources)} documents"
            
            full_response = answer + footer
            
            return history + [(message, full_response)]
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}\n\nPlease try rephrasing your question."
            return history + [(message, error_msg)]

# Initialize chatbot
chatbot = NortheasternChatbot()

# Create Gradio interface
with gr.Blocks(title="Northeastern OGS Assistant", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 🎓 Northeastern University International Student Assistant
        
        ### How to use:
        1. **Select a category** that best matches your question
        2. **Type your question** in the text box
        3. **Click Send** to get your answer
        
        All answers will be filtered based on your selected category for more accurate responses.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Select Category")
            
            category_radio = gr.Radio(
                choices=[info["name"] for info in CATEGORIES.values()],
                label="Choose the topic area for your question:",
                value=None,
                type="value"
            )
            
            # Category description
            category_desc = gr.Markdown("")
            example_questions = gr.Markdown("")
            
            def update_category_info(category_name):
                if not category_name:
                    return "", ""
                
                for key, info in CATEGORIES.items():
                    if info["name"] == category_name:
                        desc = f"**Description:** {info['description']}"
                        examples = "**Example questions:**\n" + "\n".join([f"• {ex}" for ex in info['examples']])
                        return desc, examples
                return "", ""
            
            category_radio.change(
                update_category_info,
                inputs=[category_radio],
                outputs=[category_desc, example_questions]
            )
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")
            
            chatbot_interface = gr.Chatbot(
                label="Conversation",
                height=500,
                bubble_full_width=False,
                avatar_images=("🧑‍🎓", "🤖")
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="First select a category, then type your question here...",
                    lines=2,
                    scale=4
                )
                
                submit = gr.Button("🚀 Send", variant="primary", scale=1)
            
            with gr.Row():
                clear = gr.Button("🗑️ Clear Chat")
                reset = gr.Button("🔄 Reset Category")
    
    # Example usage section
    gr.Markdown(
        """
        ---
        ### 💡 Tips for best results:
        - Select the most relevant category for your question
        - Be specific in your questions
        - If you're unsure which category to choose, think about which office would handle your question
        - You can ask multiple questions within the same category
        """
    )
    
    # Event handlers
    def respond(message, category, history):
        if not message.strip():
            return "", history
        
        new_history = chatbot.chat_with_category(message, category, history)
        return "", new_history
    
    def clear_chat():
        return []
    
    def reset_category():
        return None, [], "", ""
    
    # Connect events
    msg.submit(respond, [msg, category_radio, chatbot_interface], [msg, chatbot_interface])
    submit.click(respond, [msg, category_radio, chatbot_interface], [msg, chatbot_interface])
    clear.click(clear_chat, outputs=[chatbot_interface])
    reset.click(reset_category, outputs=[category_radio, chatbot_interface, category_desc, example_questions])
    
    # Footer
    gr.Markdown(
        """
        ---
        📚 *This AI assistant provides information based on Northeastern University's international student documentation. 
        Always verify important information with the Office of Global Services.*
        """
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=True, server_name="0.0.0.0", server_port=7860)