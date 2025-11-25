"""
RAG Helper for LiveKit Voice Agent.

Connects the voice agent to the existing RAG pipeline to retrieve
relevant context from the knowledge base in real-time.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.vector_store import FAISSVectorStore
from src.retriever.query_engine import QueryEngine

logger = logging.getLogger(__name__)


class RAGHelper:
    """
    Helper class to integrate RAG with the voice agent.
    
    Handles real-time retrieval of relevant context from the knowledge base.
    """
    
    def __init__(self, config: Config):
        """
        Initialize RAG helper.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.query_engine: Optional[QueryEngine] = None
        self.embedder: Optional[EmbeddingGenerator] = None
        self.vector_store: Optional[FAISSVectorStore] = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAG components (embedder, vector store, query engine)."""
        try:
            logger.info("Initializing RAG components...")
            
            # Initialize embedder
            self.embedder = EmbeddingGenerator(
                model_name=self.config.EMBEDDING_MODEL
            )
            logger.info("Embedder initialized")
            
            # Initialize vector store
            self.vector_store = FAISSVectorStore(
                dimension=self.config.EMBEDDING_DIMENSION,
                index_path=self.config.VECTOR_STORE_PATH,
                index_type=self.config.FAISS_INDEX_TYPE
            )
            
            # Load existing index
            if self.vector_store.index_exists():
                self.vector_store.load()
                logger.info(f"Loaded vector store with {self.vector_store.index.ntotal} vectors")
            else:
                logger.warning("No vector store found. Agent will work without RAG context.")
                return
            
            # Initialize query engine
            self.query_engine = QueryEngine(
                vector_store=self.vector_store,
                embedder=self.embedder,
                top_k=self.config.TOP_K_RESULTS,
                score_threshold=self.config.SIMILARITY_THRESHOLD
            )
            logger.info("Query engine initialized")
            
            logger.info("RAG components ready")
            
        except Exception as e:
            logger.error(f"Error initializing RAG components: {e}", exc_info=True)
            raise
    
    async def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_context_length: int = 1500
    ) -> str:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve (uses config default if None)
            max_context_length: Maximum total characters in context
            
        Returns:
            Formatted context string ready for Gemini
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return ""
        
        if self.query_engine is None:
            logger.warning("Query engine not initialized, no RAG context available")
            return ""
        
        try:
            logger.info(f"Retrieving context for: {query[:100]}...")
            
            # Retrieve documents with context
            result = self.query_engine.retrieve_with_context(
                query=query,
                max_context_length=max_context_length
            )
            
            if not result['context']:
                logger.info("No relevant context found")
                return "No relevant information found in the knowledge base."
            
            # Format context for voice
            formatted_context = self._format_context_for_voice(
                result['documents'],
                query
            )
            
            logger.info(f"Retrieved {result['num_documents']} documents, {len(formatted_context)} chars")
            
            return formatted_context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            return f"Error retrieving context: {str(e)}"
    
    def _format_context_for_voice(
        self,
        documents: List[Dict],
        query: str
    ) -> str:
        """
        Format retrieved documents for voice consumption.
        
        Makes context more natural for voice responses by:
        - Adding conversational framing
        - Numbering sources
        - Keeping it concise
        
        Args:
            documents: Retrieved documents
            query: Original query
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        context_parts = [
            f"Based on the knowledge base, here's what I found about '{query}':\n"
        ]
        
        for i, doc in enumerate(documents, 1):
            text = doc['text']
            score = doc.get('score', 0)
            
            # Add source number and text
            context_parts.append(f"Source {i} (relevance: {score:.2f}):")
            context_parts.append(text)
            context_parts.append("")  # Empty line between sources
        
        return "\n".join(context_parts)
    
    async def retrieve_structured(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Dict:
        """
        Retrieve context with structured information.
        
        Returns full document details including metadata and scores.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with structured results
        """
        if self.query_engine is None:
            return {
                'query': query,
                'documents': [],
                'success': False,
                'error': 'Query engine not initialized'
            }
        
        try:
            # Override top_k if provided
            if top_k:
                self.query_engine.update_parameters(top_k=top_k)
            
            # Retrieve documents
            documents = self.query_engine.retrieve(query)
            
            return {
                'query': query,
                'documents': documents,
                'num_results': len(documents),
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error in structured retrieval: {e}", exc_info=True)
            return {
                'query': query,
                'documents': [],
                'success': False,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the RAG system.
        
        Returns:
            Dictionary with RAG statistics
        """
        if self.query_engine is None:
            return {
                'status': 'not_initialized',
                'num_documents': 0
            }
        
        stats = self.query_engine.get_stats()
        return {
            'status': 'ready',
            'num_documents': stats['vector_store']['num_vectors'],
            'embedding_model': stats['embedder']['model_name'],
            'top_k': stats['top_k'],
            'threshold': stats['score_threshold']
        }
    
    def is_ready(self) -> bool:
        """
        Check if RAG system is ready.
        
        Returns:
            True if ready, False otherwise
        """
        return (
            self.query_engine is not None and
            self.vector_store is not None and
            self.vector_store.index.ntotal > 0
        )


if __name__ == "__main__":
    """Test the RAG helper."""
    import asyncio
    from dotenv import load_dotenv
    
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("RAG HELPER TEST")
    print("="*80)
    
    # Load config
    load_dotenv()
    config = Config()
    
    # Initialize helper
    try:
        helper = RAGHelper(config)
        
        # Check status
        stats = helper.get_stats()
        print(f"\nRAG Status: {stats['status']}")
        print(f"Documents in index: {stats.get('num_documents', 0)}")
        print(f"Ready: {helper.is_ready()}")
        
        if helper.is_ready():
            # Test retrieval
            print("\n" + "="*80)
            print("Testing retrieval...")
            print("="*80)
            
            test_query = "What is two-factor authentication?"
            
            async def test():
                context = await helper.retrieve_context(test_query)
                print(f"\nQuery: {test_query}")
                print(f"\nContext:\n{context}")
            
            asyncio.run(test())
            
            print("\n" + "="*80)
            print("Test completed successfully!")
            print("="*80)
        else:
            print("\nRAG not ready. Please ingest documents first:")
            print("python main.py --ingest --data-dir ./data/raw")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()