#!/usr/bin/env python3
"""Test script to verify configuration loading."""
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.common.config import CONFIG

def test_config_loading():
    """Test that configuration is loaded correctly."""
    print("Testing configuration loading...")
    
    # Test vector store config
    print(f"Vector store index path: {CONFIG['vector_store']['index_path']}")
    print(f"Vector store metadata path: {CONFIG['vector_store']['metadata_path']}")
    
    # Test embeddings config
    print(f"Azure embeddings dimension: {CONFIG['embeddings']['azure']['dimension']}")
    print(f"Gemini embeddings dimension: {CONFIG['embeddings']['gemini']['dimension']}")
    
    # Test LLM config
    print(f"Azure LLM temperature: {CONFIG['llm']['azure']['temperature']}")
    print(f"Gemini LLM temperature: {CONFIG['llm']['gemini']['temperature']}")
    
    # Test document loader config
    print(f"Document loader chunk size: {CONFIG['document_loader']['chunk_size']}")
    
    # Test AI service config
    print(f"AI service system prompt: {CONFIG['ai_service']['system_prompt']}")
    
    # Test providers config
    print(f"LLM provider: {CONFIG['providers']['llm']}")
    print(f"Embeddings provider: {CONFIG['providers']['embeddings']}")
    
    print("Configuration loading test completed successfully!")

if __name__ == "__main__":
    test_config_loading() 