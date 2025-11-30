"""
Optimized RAG Helper for LiveKit Voice Agent.

Key optimizations:
- Smart query filtering (skip RAG for greetings)
- Cached embeddings for common queries
- Reduced context length
- Batch processing optimization
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict
from functools import lru_cache

sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.vector_store import FAISSVectorStore
from src.retriever.query_engine import QueryEngine

logger = logging.getLogger(__name__)

# Common greetings that don't need RAG
GREETINGS = {
    'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
    'good evening', 'howdy', 'sup', 'yo', 'hiya', 'whats up'
}


class RAGHelper:
    """
    Optimized helper class to integrate RAG with the voice agent.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.query_engine: Optional[QueryEngine] = None
        self.embedder: Optional[EmbeddingGenerator] = None
        self.vector_store: Optional[FAISSVectorStore] = None
        
        # Cache for recent queries (LRU with max 100 items)
        self._embedding_cache = {}
        self._max_cache_size = 100
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAG components with minimal logging."""
        try:
            # Initialize embedder
            self.embedder = EmbeddingGenerator(
                model_name=self.config.EMBEDDING_MODEL
            )
            
            # Initialize vector store
            self.vector_store = FAISSVectorStore(
                dimension=self.config.EMBEDDING_DIMENSION,
                index_path=self.config.VECTOR_STORE_PATH,
                index_type=self.config.FAISS_INDEX_TYPE
            )
            
            # Load existing index
            if self.vector_store.index_exists():
                self.vector_store.load()
                logger.info(f"RAG ready: {self.vector_store.index.ntotal} docs")
            else:
                logger.warning("No vector store found")
                return
            
            # Initialize query engine with optimized settings
            self.query_engine = QueryEngine(
                vector_store=self.vector_store,
                embedder=self.embedder,
                top_k=3,  # Reduced from 5 to 3
                score_threshold=self.config.SIMILARITY_THRESHOLD
            )
            
            logger.info("RAG components ready")
            
        except Exception as e:
            logger.error(f"Error initializing RAG: {e}")
            raise
    
    def _is_greeting(self, query: str) -> bool:
        """Check if query is a simple greeting."""
        normalized = query.lower().strip()
        return normalized in GREETINGS or len(normalized.split()) <= 2
    
    @lru_cache(maxsize=100)
    def _get_cached_embedding(self, query: str):
        """Get cached embedding or generate new one."""
        return self.embedder.embed_query(query)
    
    async def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_context_length: int = 1000  # Reduced from 1500
    ) -> str:
        """
        Retrieve relevant context with optimizations.
        """
        if not query or not query.strip():
            return ""
        
        # Skip RAG for simple greetings
        if self._is_greeting(query):
            logger.info("Skipping RAG for greeting")
            return "No knowledge base query needed for greetings."
        
        if self.query_engine is None:
            logger.warning("Query engine not initialized")
            return ""
        
        try:
            # Retrieve with reduced context
            result = self.query_engine.retrieve_with_context(
                query=query,
                max_context_length=max_context_length
            )
            
            if not result['context']:
                return "No relevant information found in the knowledge base."
            
            # Simplified formatting for voice
            docs = result['documents']
            if not docs:
                return "No relevant information found."
            
            # Format concisely
            context_parts = []
            for i, doc in enumerate(docs[:2], 1):  # Limit to top 2 docs
                text = doc['text'][:300]  # Truncate long docs
                context_parts.append(f"Source {i}: {text}")
            
            formatted = "\n\n".join(context_parts)
            logger.info(f"Retrieved {len(docs)} docs, {len(formatted)} chars")
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return f"Error retrieving context: {str(e)}"
    
    async def retrieve_structured(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Dict:
        """Retrieve with structured information."""
        if self.query_engine is None:
            return {
                'query': query,
                'documents': [],
                'success': False,
                'error': 'Query engine not initialized'
            }
        
        try:
            if top_k:
                self.query_engine.update_parameters(top_k=top_k)
            
            documents = self.query_engine.retrieve(query)
            
            return {
                'query': query,
                'documents': documents,
                'num_results': len(documents),
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error in structured retrieval: {e}")
            return {
                'query': query,
                'documents': [],
                'success': False,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict:
        """Get RAG system statistics."""
        if self.query_engine is None:
            return {'status': 'not_initialized', 'num_documents': 0}
        
        stats = self.query_engine.get_stats()
        return {
            'status': 'ready',
            'num_documents': stats['vector_store']['num_vectors'],
            'embedding_model': stats['embedder']['model_name'],
            'top_k': stats['top_k'],
            'threshold': stats['score_threshold']
        }
    
    def is_ready(self) -> bool:
        """Check if RAG system is ready."""
        return (
            self.query_engine is not None and
            self.vector_store is not None and
            self.vector_store.index.ntotal > 0
        )


if __name__ == "__main__":
    """Test the optimized RAG helper."""
    import asyncio
    from dotenv import load_dotenv
    
    logging.basicConfig(level=logging.INFO)
    
    load_dotenv()
    config = Config()
    
    try:
        helper = RAGHelper(config)
        
        stats = helper.get_stats()
        print(f"\nRAG Status: {stats['status']}")
        print(f"Documents: {stats.get('num_documents', 0)}")
        
        if helper.is_ready():
            async def test():
                # Test greeting (should skip RAG)
                print("\n--- Test 1: Greeting ---")
                context = await helper.retrieve_context("hi")
                print(f"Context: {context}")
                
                # Test real query
                print("\n--- Test 2: Real Query ---")
                context = await helper.retrieve_context("What is 2FA?")
                print(f"Context: {context[:200]}...")
            
            asyncio.run(test())
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()