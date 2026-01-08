#!/usr/bin/env python3
"""
Command-line interface for RAG Customer Service Chatbot
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vector_store import load_vector_store
from rag_pipeline import create_pipeline
from config import APP_NAME
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 70)
    print(f"  {APP_NAME}".center(70))
    print("=" * 70)
    print("\nWelcome! I'm here to help with product info, policies, and support.")
    print("\nCommands:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit' or 'exit' to leave")
    print("  - Type 'clear' to clear conversation history")
    print("  - Type 'stats' to see system statistics")
    print("=" * 70 + "\n")


def print_separator():
    """Print separator line"""
    print("-" * 70)


def main():
    """Main CLI application"""
    
    # Load pipeline
    print("\nLoading RAG pipeline...")
    try:
        vector_store = load_vector_store()
        pipeline = create_pipeline(vector_store)
        print("✓ Pipeline loaded successfully!\n")
    except Exception as e:
        print(f"\n✗ Error loading pipeline: {e}")
        print("\nPlease run 'python build_index.py' first to create the vector index.")
        sys.exit(1)
    
    # Print banner
    print_banner()
    
    # Main loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for commands
            if user_input.lower() in ['quit', 'exit']:
                print("\nThank you for using our service. Goodbye! 👋\n")
                break
            
            if user_input.lower() == 'clear':
                pipeline.clear_history()
                print("\n✓ Conversation history cleared.\n")
                continue
            
            if user_input.lower() == 'stats':
                stats = pipeline.get_stats()
                print("\n📊 System Statistics:")
                print_separator()
                print(f"Total vectors: {stats['vector_store_stats']['total_vectors']}")
                print(f"Total chunks: {stats['vector_store_stats']['total_chunks']}")
                print(f"Vector dimension: {stats['vector_store_stats']['dimension']}")
                print(f"Conversation length: {stats['conversation_length']}")
                print_separator()
                print()
                continue
            
            if not user_input:
                continue
            
            # Process query
            print("\n🤖 Assistant: ", end="", flush=True)
            
            # Get response
            result = pipeline.query(user_input)
            
            # Print response
            print(result['response'])
            print()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.\n")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\n✗ An error occurred: {e}\n")


if __name__ == "__main__":
    main()