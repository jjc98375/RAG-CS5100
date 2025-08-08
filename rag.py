"""
RAG System for Northeastern University International Student Documentation
Using LangChain, Ollama, and ChromaDB for vector storage
Single category focused system - user must select a category to query
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
import warnings
import logging

# Suppress ChromaDB telemetry warnings
warnings.filterwarnings("ignore", message="Failed to send telemetry")
logging.getLogger('chromadb.telemetry').setLevel(logging.ERROR)

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    # Try to use the new langchain-ollama package
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
except ImportError:
    # Fall back to community version if not available
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama as OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

class NortheasternRAG:
    """RAG system for Northeastern University international student documentation - Category focused"""
    
    # Define available categories
    AVAILABLE_CATEGORIES = {
        "arrival": "Arrival",
        "billing": "Billing", 
        "course": "Course Registration",
        "employment": "Employment",
        "forms": "forms",
        "scholarships": "Scholarships",
        "tuition": "Tuition&Fees",
        "visa": "Visa"
    }
    
    def __init__(
        self,
        base_directory: str = "./northeastern_docs",
        model_name: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        persist_directory: str = "./northeastern_chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the RAG system
        
        Args:
            base_directory: Base directory containing folders with PDFs
            model_name: Ollama model name for generation
            embedding_model: Ollama model name for embeddings
            persist_directory: Directory to persist vector database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.base_directory = base_directory
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.llm = OllamaLLM(model=model_name, temperature=0.2)
        self.vectorstore = None
        self.qa_chain = None
        
        # Track processed files
        self.processed_files = []
        
        # Current selected category
        self.current_category = None
    
    def check_database_status(self) -> Dict:
        """
        Check if the vector database exists and get its status
        
        Returns:
            Dictionary with database status information
        """
        abs_persist_dir = os.path.abspath(self.persist_directory)
        exists = os.path.exists(abs_persist_dir)
        
        status = {
            "exists": exists,
            "path": abs_persist_dir,
            "size_mb": 0,
            "files": []
        }
        
        if exists:
            # Calculate total size
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(abs_persist_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
                    status["files"].append(filename)
            
            status["size_mb"] = total_size / (1024 * 1024)  # Convert to MB
            print(f"📊 Database found at: {abs_persist_dir}")
            print(f"   Size: {status['size_mb']:.2f} MB")
            print(f"   Files: {len(status['files'])}")
        else:
            print(f"❌ No database found at: {abs_persist_dir}")
        
        return status
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the vector database
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "database_exists": False,
            "total_documents": 0,
            "persist_directory": self.persist_directory,
            "available_categories": list(self.AVAILABLE_CATEGORIES.keys()),
            "current_category": self.current_category
        }
        
        if self.vectorstore:
            try:
                # Try to get collection stats
                collection = self.vectorstore._collection
                stats["database_exists"] = True
                stats["total_documents"] = collection.count()
            except Exception as e:
                print(f"Could not retrieve statistics: {e}")
        
        return stats
    
    def list_categories(self) -> None:
        """Display available categories to the user"""
        print("\n📚 Available Categories:")
        print("-" * 40)
        for key, folder in self.AVAILABLE_CATEGORIES.items():
            status = "✓" if self.current_category == key else " "
            print(f"[{status}] {key:15} - {folder}")
        print("-" * 40)
        if self.current_category:
            print(f"Currently selected: {self.current_category}")
        else:
            print("No category selected")
        
    def load_pdfs_from_folders(self, folders: Optional[List[str]] = None) -> List[Document]:
        """
        Load all PDFs from specified folders
        
        Args:
            folders: List of folder names to process. If None, process all folders
            
        Returns:
            List of all documents
        """
        all_documents = []
        base_path = Path(self.base_directory)
        
        # Define folders to process
        if folders is None:
            folders = list(self.AVAILABLE_CATEGORIES.values())
        
        print(f"Processing PDFs from {len(folders)} folders...")
        
        for folder in folders:
            folder_path = base_path / folder
            
            if not folder_path.exists():
                print(f"Warning: Folder '{folder}' not found at {folder_path}")
                continue
            
            # Find all PDFs in the folder (including subdirectories)
            pdf_files = list(folder_path.glob("**/*.pdf"))
            
            print(f"\nProcessing folder: {folder}")
            print(f"Found {len(pdf_files)} PDF files")
            
            for pdf_file in pdf_files:
                try:
                    # Load PDF
                    loader = PyPDFLoader(str(pdf_file))
                    documents = loader.load()
                    
                    # Add metadata to each document
                    # Find the category key for this folder
                    category_key = None
                    for key, val in self.AVAILABLE_CATEGORIES.items():
                        if val == folder:
                            category_key = key
                            break
                    
                    for doc in documents:
                        doc.metadata.update({
                            "source": str(pdf_file),
                            "folder": folder,
                            "filename": pdf_file.name,
                            "category": category_key or folder.lower().replace(" ", "_")
                        })
                    
                    all_documents.extend(documents)
                    self.processed_files.append(str(pdf_file))
                    print(f"  ✓ Loaded: {pdf_file.name}")
                    
                except Exception as e:
                    print(f"  ✗ Error loading {pdf_file.name}: {str(e)}")
        
        print(f"\nTotal documents loaded: {len(all_documents)}")
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            is_separator_regex=False
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} document chunks from {len(documents)} documents")
        
        return chunks
    
    def create_or_update_vector_database(self, documents: List[Document], update: bool = False) -> Chroma:
        """
        Create new or update existing vector database
        
        Args:
            documents: List of document chunks
            update: If True, add to existing database; if False, create new
            
        Returns:
            Chroma vector store
        """
        import chromadb
        from chromadb.config import Settings
        import shutil
        
        # Use absolute path
        abs_persist_dir = os.path.abspath(self.persist_directory)
        print(f"\n🗄️ Database path (absolute): {abs_persist_dir}")
        
        # Configure ChromaDB with telemetry disabled
        client_settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            persist_directory=abs_persist_dir
        )
        
        if update and os.path.exists(abs_persist_dir):
            print("Updating existing database...")
            # Load existing database
            self.vectorstore = Chroma(
                persist_directory=abs_persist_dir,
                embedding_function=self.embeddings,
                client_settings=client_settings
            )
            # Add new documents
            self.vectorstore.add_documents(documents)
        else:
            # Delete existing directory if it exists
            if os.path.exists(abs_persist_dir):
                print(f"Removing existing directory: {abs_persist_dir}")
                shutil.rmtree(abs_persist_dir)
            
            # Create directory
            os.makedirs(abs_persist_dir, exist_ok=True)
            print(f"✓ Directory created: {abs_persist_dir}")
            
            print("Creating new database...")
            print("This may take a few minutes depending on document count...")
            
            # Create new database with progress tracking
            batch_size = 50
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            # Create ChromaDB client directly
            client = chromadb.PersistentClient(
                path=abs_persist_dir,
                settings=client_settings
            )
            
            # Create collection
            try:
                collection = client.get_collection("langchain")
                client.delete_collection("langchain")
            except:
                pass
            
            # Create vectorstore with first batch
            first_batch = documents[:batch_size]
            print(f"  Processing batch 1/{total_batches} ({len(first_batch)} documents)...")
            
            self.vectorstore = Chroma.from_documents(
                documents=first_batch,
                embedding=self.embeddings,
                persist_directory=abs_persist_dir,
                client_settings=client_settings
            )
            
            # Add remaining batches
            for i in range(batch_size, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                current_batch = i // batch_size + 1
                print(f"  Processing batch {current_batch}/{total_batches} ({len(batch)} documents)...")
                self.vectorstore.add_documents(batch)
        
        print(f"\n✓ Vector database saved at: {abs_persist_dir}")
        
        return self.vectorstore
    
    def load_vector_database(self) -> Chroma:
        """Load existing vector database"""
        if not os.path.exists(self.persist_directory):
            raise ValueError(f"No vector database found at {self.persist_directory}")
            
        print(f"Loading vector database from {self.persist_directory}...")
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        print("Vector database loaded successfully")
        return self.vectorstore
    
    def set_category(self, category: str) -> bool:
        """
        Set the current category for queries
        
        Args:
            category: Category key (e.g., 'visa', 'billing')
            
        Returns:
            True if category was set successfully, False otherwise
        """
        if category not in self.AVAILABLE_CATEGORIES:
            print(f"❌ Invalid category: '{category}'")
            print(f"Available categories: {', '.join(self.AVAILABLE_CATEGORIES.keys())}")
            return False
        
        self.current_category = category
        print(f"✓ Category set to: {category} ({self.AVAILABLE_CATEGORIES[category]})")
        
        # Setup QA chain for this category
        if self.vectorstore:
            self.setup_qa_chain_for_category(category)
        
        return True
    
    def setup_qa_chain_for_category(self, category: str):
        """
        Setup the QA chain for a specific category
        
        Args:
            category: Category to filter by
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        if category not in self.AVAILABLE_CATEGORIES:
            raise ValueError(f"Invalid category: {category}")
        
        # Create retriever filtered by category
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 6,  # Retrieve more documents for better context
                "filter": {"category": category}
            }
        )
        
        # Custom prompt template for category-specific queries
        prompt_template = f"""You are a helpful assistant for Northeastern University international students, 
        specifically answering questions about {self.AVAILABLE_CATEGORIES[category]}.
        
        Use ALL the relevant information from the provided context to give a comprehensive answer.
        The context comes from the {self.AVAILABLE_CATEGORIES[category]} documentation.
        
        Important instructions:
        - Provide detailed, comprehensive answers using ALL relevant information from the context
        - Include specific policies, procedures, requirements, dates, and amounts when mentioned
        - If multiple documents contain relevant information, synthesize them into a complete answer
        - Be specific and thorough - don't leave out important details
        - If the context doesn't contain relevant information, say "I don't have that specific information in the {self.AVAILABLE_CATEGORIES[category]} documentation"
        
        Context: {{context}}
        
        Question: {{question}}
        
        Comprehensive Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print(f"QA chain setup completed for category: {category}")
    
    def query(self, question: str) -> Dict:
        """
        Query the RAG system using the currently selected category
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.current_category:
            return {
                "error": "No category selected. Please select a category first using set_category()",
                "available_categories": list(self.AVAILABLE_CATEGORIES.keys())
            }
        
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Run set_category() first.")
        
        print(f"\n🔍 Searching in category: {self.current_category} ({self.AVAILABLE_CATEGORIES[self.current_category]})")
        
        # Query the QA chain
        result = self.qa_chain({"query": question})
        
        # Format the response
        response = {
            "category": self.current_category,
            "category_name": self.AVAILABLE_CATEGORIES[self.current_category],
            "question": question,
            "answer": result["result"],
            "source_documents": []
        }
        
        # Add source information with unique documents
        seen_files = set()
        for doc in result["source_documents"]:
            filename = doc.metadata.get('filename', 'Unknown')
            if filename not in seen_files:
                seen_files.add(filename)
                response["source_documents"].append({
                    "filename": filename,
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                })
        
        return response
    
    def process_all_folders(self):
        """
        Complete pipeline to process all folders and create vector database
        """
        # Load all PDFs from all categories
        documents = self.load_pdfs_from_folders()
        
        if not documents:
            print("No documents found to process!")
            return
        
        # Split documents
        chunks = self.split_documents(documents)
        
        # Create vector database
        self.create_or_update_vector_database(chunks, update=False)
        
        print(f"\n✓ RAG system initialized successfully!")
        print(f"  - Processed {len(self.processed_files)} PDF files")
        print(f"  - Created {len(chunks)} chunks")
        print(f"  - Vector database ready at: {self.persist_directory}")
        print(f"\n⚠️  Remember to select a category before querying!")
        self.list_categories()


def interactive_mode(rag: NortheasternRAG):
    """
    Interactive mode for querying the RAG system
    
    Args:
        rag: Initialized NortheasternRAG instance
    """
    print("\n" + "="*60)
    print("INTERACTIVE QUERY MODE")
    print("="*60)
    print("\nCommands:")
    print("  'list' - Show available categories")
    print("  'set <category>' - Select a category (e.g., 'set visa')")
    print("  'status' - Show current status")
    print("  'quit' or 'exit' - Exit the program")
    print("\nAfter selecting a category, type your questions directly.")
    print("="*60)
    
    while True:
        print()
        if rag.current_category:
            prompt = f"[{rag.current_category}] > "
        else:
            prompt = "[no category] > "
        
        user_input = input(prompt).strip()
        
        if not user_input:
            continue
        
        # Check for commands
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        elif user_input.lower() == 'list':
            rag.list_categories()
        
        elif user_input.lower() == 'status':
            stats = rag.get_statistics()
            print("\n📊 System Status:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        elif user_input.lower().startswith('set '):
            category = user_input[4:].strip().lower()
            if rag.set_category(category):
                print(f"You can now ask questions about {rag.AVAILABLE_CATEGORIES[category]}")
        
        else:
            # Treat as a question
            if not rag.current_category:
                print("❌ Please select a category first using 'set <category>'")
                print("   Available categories:", ', '.join(rag.AVAILABLE_CATEGORIES.keys()))
            else:
                print("\n🔍 Processing your question...")
                try:
                    result = rag.query(user_input)
                    
                    print(f"\n📁 Category: {result['category_name']}")
                    print(f"❓ Question: {result['question']}")
                    print(f"\n💡 Answer:\n{'-'*50}")
                    print(result['answer'])
                    
                    if result['source_documents']:
                        print(f"\n📚 Sources ({len(result['source_documents'])} documents):")
                        for i, doc in enumerate(result['source_documents'], 1):
                            print(f"  [{i}] {doc['filename']}")
                    
                except Exception as e:
                    print(f"❌ Error processing query: {str(e)}")


def main():
    """Main function with interactive mode"""
    
    # Initialize RAG system
    rag = NortheasternRAG(
        base_directory="./northeastern_docs",
        model_name="llama2",  # or "llama3.2"
        embedding_model="nomic-embed-text",
        persist_directory="./northeastern_unified_db"
    )
    
    # Check if database exists, if not create it
    db_status = rag.check_database_status()
    if not db_status["exists"]:
        print("="*60)
        print("Building Vector Database from All Categories")
        print("="*60)
        rag.process_all_folders()
    else:
        print("="*60)
        print("Loading Existing Vector Database")
        print("="*60)
        rag.load_vector_database()
    
    # Show available categories
    print("\n" + "="*60)
    print("NORTHEASTERN UNIVERSITY RAG SYSTEM")
    print("Category-Based Question Answering")
    print("="*60)
    
    rag.list_categories()
    
    # Example demonstration
    print("\n" + "="*60)
    print("EXAMPLE DEMONSTRATION")
    print("="*60)
    
    # Demonstrate with a few categories
    demo_queries = [
        ("visa", "What documents do I need for F-1 visa application?"),
        ("employment", "How many hours can international students work on campus?"),
        ("scholarships", "What scholarships are available for international students?")
    ]
    
    for category, question in demo_queries:
        print(f"\n{'='*40}")
        print(f"Setting category to: {category}")
        rag.set_category(category)
        
        print(f"\nQuestion: {question}")
        result = rag.query(question)
        
        print(f"\nAnswer from {result['category_name']} documents:")
        print(result['answer'][:500] + "..." if len(result['answer']) > 500 else result['answer'])
        
        if result['source_documents']:
            print(f"\nSources used:")
            for doc in result['source_documents'][:3]:
                print(f"  - {doc['filename']}")
    
    # Enter interactive mode
    print("\n" + "="*60)
    interactive_mode(rag)


if __name__ == "__main__":
    main()