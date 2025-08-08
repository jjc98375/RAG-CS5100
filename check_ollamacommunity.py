import sys

# Check if langchain-ollama is available
try:
    import langchain_ollama
    print("✅ Using: langchain-ollama (dedicated package)")
except ImportError:
    print("📦 Using: langchain-community (community version)")

# Check what's actually imported in your system
try:
    from langchain_community.llms import Ollama
    print("✓ langchain-community is available")
except:
    print("✗ langchain-community not found")