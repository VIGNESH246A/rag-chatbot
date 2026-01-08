#!/usr/bin/env python3
"""
Streamlit interface for RAG Customer Service Chatbot
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from vector_store import load_vector_store
from rag_pipeline import create_pipeline
from config import APP_NAME
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
}
.assistant-message {
    background-color: #f5f5f5;
    border-left: 4px solid #4caf50;
}
.context-box {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffc107;
    margin-top: 1rem;
}
.stats-box {
    background-color: #e8f5e9;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load RAG pipeline (cached)"""
    try:
        vector_store = load_vector_store()
        pipeline = create_pipeline(vector_store)
        return pipeline
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        st.info("Please run 'python build_index.py' first to create the vector index.")
        return None


def display_message(role, content):
    """Display chat message"""
    if role == "user":
        st.markdown(f'<div class="chat-message user-message"><b>👤 You:</b><br>{content}</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message assistant-message"><b>🤖 Assistant:</b><br>{content}</div>', 
                   unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Header
    st.markdown(f'<div class="main-header">🤖 {APP_NAME}</div>', unsafe_allow_html=True)
    
    # Load pipeline
    pipeline = load_pipeline()
    
    if pipeline is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Show stats
        if st.button("📊 Show Statistics"):
            stats = pipeline.get_stats()
            st.markdown('<div class="stats-box">', unsafe_allow_html=True)
            st.write("**Vector Store Stats:**")
            st.write(f"- Total vectors: {stats['vector_store_stats']['total_vectors']}")
            st.write(f"- Total chunks: {stats['vector_store_stats']['total_chunks']}")
            st.write(f"- Dimension: {stats['vector_store_stats']['dimension']}")
            st.write(f"- Conversation length: {stats['conversation_length']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            pipeline.clear_history()
            st.session_state.messages = []
            st.success("Conversation cleared!")
            st.rerun()
        
        # Options
        st.subheader("Options")
        show_context = st.checkbox("Show Retrieved Context", value=False)
        show_scores = st.checkbox("Show Relevance Scores", value=False)
        
        st.markdown("---")
        st.markdown("### 💡 Sample Questions")
        st.markdown("""
        - What is the return policy?
        - How do I set up the Smart Thermostat?
        - Tell me about the Smart Refrigerator
        - My security camera won't connect
        - What payment methods do you accept?
        """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(message["role"], message["content"])
        
        # Show context if enabled
        if show_context and "context" in message:
            with st.expander("📚 Retrieved Context"):
                st.markdown('<div class="context-box">', unsafe_allow_html=True)
                for idx, chunk in enumerate(message["context"], 1):
                    st.markdown(f"**Context {idx}** (Section: {chunk.get('section', 'N/A')})")
                    st.text(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
                    if show_scores and "scores" in message:
                        st.write(f"Relevance Score: {message['scores'][idx-1]:.4f}")
                    st.markdown("---")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about our products or policies..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)
        
        # Get response
        with st.spinner("🤔 Thinking..."):
            if show_scores:
                result = pipeline.query_with_scores(prompt)
            else:
                result = pipeline.query(prompt)
        
        # Add assistant message
        message_data = {
            "role": "assistant",
            "content": result['response'],
            "context": result['context_chunks']
        }
        
        if show_scores and 'relevance_scores' in result:
            message_data["scores"] = result['relevance_scores']
        
        st.session_state.messages.append(message_data)
        
        # Rerun to display new messages
        st.rerun()


if __name__ == "__main__":
    main()