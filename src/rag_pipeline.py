import logging
from typing import Dict, List, Optional
from retriever import ContextRetriever
from llm_gemini import GeminiLLM, ConversationManager
from prompt_templates import create_rag_prompt, create_followup_prompt
from vector_store import FAISSVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline"""
    
    def __init__(self, vector_store: FAISSVectorStore, top_k: int = 3):
        self.retriever = ContextRetriever(vector_store, top_k)
        self.llm = GeminiLLM()
        self.conversation_manager = ConversationManager()
        logger.info("RAG Pipeline initialized")
    
    def query(self, user_query: str, use_history: bool = True) -> Dict:
        """Process user query through RAG pipeline"""
        try:
            logger.info(f"Processing query: {user_query}")
            
            # Step 1: Retrieve relevant context
            chunks = self.retriever.retrieve(user_query)
            context = self.retriever.format_context(chunks)
            
            # Step 2: Create prompt
            if use_history and self.conversation_manager.history:
                history = self.conversation_manager.get_history_string()
                prompt = create_followup_prompt(user_query, context, history)
            else:
                prompt = create_rag_prompt(user_query, context)
            
            # Step 3: Generate response
            response = self.llm.generate_response(prompt)
            
            # Step 4: Update conversation history
            self.conversation_manager.add_user_message(user_query)
            self.conversation_manager.add_assistant_message(response)
            
            # Return structured result
            return {
                'query': user_query,
                'response': response,
                'context_chunks': chunks,
                'num_chunks_retrieved': len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                'query': user_query,
                'response': "I apologize, but I encountered an error processing your request.",
                'context_chunks': [],
                'num_chunks_retrieved': 0,
                'error': str(e)
            }
    
    def query_with_scores(self, user_query: str) -> Dict:
        """Process query and return with relevance scores"""
        try:
            logger.info(f"Processing query with scores: {user_query}")
            
            # Retrieve with scores
            results = self.retriever.retrieve_with_scores(user_query)
            chunks = [chunk for chunk, score in results]
            scores = [score for chunk, score in results]
            
            # Format context
            context = self.retriever.format_context(chunks)
            
            # Create prompt and generate response
            prompt = create_rag_prompt(user_query, context)
            response = self.llm.generate_response(prompt)
            
            return {
                'query': user_query,
                'response': response,
                'context_chunks': chunks,
                'relevance_scores': scores,
                'num_chunks_retrieved': len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                'query': user_query,
                'response': "I apologize, but I encountered an error processing your request.",
                'error': str(e)
            }
    
    def stream_response(self, user_query: str):
        """Stream response for real-time display"""
        try:
            # Retrieve context
            chunks = self.retriever.retrieve(user_query)
            context = self.retriever.format_context(chunks)
            
            # Create prompt
            prompt = create_rag_prompt(user_query, context)
            
            # Stream response
            for chunk in self.llm.generate_streaming_response(prompt):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield "I apologize, but I encountered an error."
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_manager.clear()
        logger.info("Conversation history cleared")
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            'vector_store_stats': self.retriever.vector_store.get_stats(),
            'conversation_length': len(self.conversation_manager.history)
        }


def create_pipeline(vector_store: FAISSVectorStore) -> RAGPipeline:
    """Convenience function to create RAG pipeline"""
    return RAGPipeline(vector_store)