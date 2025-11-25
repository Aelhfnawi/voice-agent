"""
Query engine module for the Technical RAG Assistant.

This module handles the retrieval component of RAG:
- Query embedding generation
- Similarity search in vector store
- Result ranking and filtering
- Context assembly for LLM

Combines embedder and vector store into a unified retrieval interface.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import from ingestion modules
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Handles query processing and document retrieval.
    
    Orchestrates embedding generation and vector search to retrieve
    relevant context for answering queries.
    """
    
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: EmbeddingGenerator,
        top_k: int = 5,
        score_threshold: float = 0.5,
        rerank: bool = False
    ):
        """
        Initialize the QueryEngine.
        
        Args:
            vector_store: FAISS vector store instance
            embedder: Embedding generator instance
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score (0-1)
            rerank: Whether to rerank results (not implemented yet)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.rerank = rerank
        
        logger.info(f"QueryEngine initialized: top_k={top_k}, threshold={score_threshold}")
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            
        Returns:
            List of dictionaries containing:
                - text: Document text
                - metadata: Document metadata
                - score: Relevance score (0-1)
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        logger.info(f"Retrieving documents for query: {query[:100]}...")
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_query(query)
            
            # Search in vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=self.top_k,
                score_threshold=self.score_threshold
            )
            
            # Format results
            formatted_results = []
            for text, metadata, score in results:
                formatted_results.append({
                    'text': text,
                    'metadata': metadata,
                    'score': score
                })
            
            logger.info(f"Retrieved {len(formatted_results)} documents")
            
            # Optionally rerank results
            if self.rerank and len(formatted_results) > 0:
                formatted_results = self._rerank_results(query, formatted_results)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            return []
    
    def retrieve_with_context(self, query: str, max_context_length: int = 4096) -> Dict:
        """
        Retrieve documents and assemble context for LLM.
        
        Args:
            query: User query string
            max_context_length: Maximum total characters in context
            
        Returns:
            Dictionary containing:
                - query: Original query
                - documents: List of retrieved documents
                - context: Assembled context string for LLM
                - num_documents: Number of documents included
        """
        # Retrieve documents
        documents = self.retrieve(query)
        
        if not documents:
            return {
                'query': query,
                'documents': [],
                'context': '',
                'num_documents': 0
            }
        
        # Assemble context
        context_parts = []
        total_length = 0
        included_docs = []
        
        for i, doc in enumerate(documents):
            doc_text = doc['text']
            doc_length = len(doc_text)
            
            # Check if adding this document would exceed limit
            if total_length + doc_length > max_context_length:
                logger.info(f"Context limit reached, including {i} of {len(documents)} documents")
                break
            
            # Add document with formatting
            context_parts.append(f"[Document {i+1}]\n{doc_text}")
            total_length += doc_length
            included_docs.append(doc)
        
        context = "\n\n".join(context_parts)
        
        logger.info(f"Assembled context: {len(context)} characters from {len(included_docs)} documents")
        
        return {
            'query': query,
            'documents': included_docs,
            'context': context,
            'num_documents': len(included_docs)
        }
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rerank results using more sophisticated scoring.
        
        Args:
            query: Original query
            results: Initial retrieval results
            
        Returns:
            Reranked results
        """
        # Placeholder for reranking logic
        # Could implement cross-encoder reranking, BM25, or other methods
        logger.warning("Reranking not yet implemented, returning original results")
        return results
    
    def batch_retrieve(self, queries: List[str]) -> List[List[Dict]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of result lists, one per query
        """
        all_results = []
        
        for query in queries:
            results = self.retrieve(query)
            all_results.append(results)
        
        return all_results
    
    def get_similar_documents(self, document_text: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Find documents similar to a given document.
        
        Args:
            document_text: Text of the document to find similar documents for
            top_k: Number of results (uses default if None)
            
        Returns:
            List of similar documents
        """
        if top_k is None:
            top_k = self.top_k
        
        # Generate embedding for the document
        doc_embedding = self.embedder.embed(document_text)
        
        # Search (use first embedding if batch returned)
        if doc_embedding.ndim > 1:
            doc_embedding = doc_embedding[0]
        
        results = self.vector_store.search(
            query_embedding=doc_embedding,
            top_k=top_k + 1,  # +1 because the document itself might be in results
            score_threshold=self.score_threshold
        )
        
        # Filter out the exact same document (if present)
        filtered_results = []
        for text, metadata, score in results:
            if text != document_text:  # Skip exact match
                filtered_results.append({
                    'text': text,
                    'metadata': metadata,
                    'score': score
                })
        
        return filtered_results[:top_k]
    
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze a query and provide insights.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with query analysis
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Get retrieval results
        results = self.retrieve(query)
        
        # Calculate statistics
        scores = [doc['score'] for doc in results]
        
        analysis = {
            'query': query,
            'query_length': len(query),
            'num_results': len(results),
            'avg_score': np.mean(scores) if scores else 0.0,
            'max_score': max(scores) if scores else 0.0,
            'min_score': min(scores) if scores else 0.0,
            'has_relevant_results': len(results) > 0 and max(scores) > self.score_threshold if scores else False
        }
        
        return analysis
    
    def update_parameters(
        self,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ):
        """
        Update query engine parameters.
        
        Args:
            top_k: New top_k value
            score_threshold: New score threshold
        """
        if top_k is not None:
            self.top_k = top_k
            logger.info(f"Updated top_k to {top_k}")
        
        if score_threshold is not None:
            self.score_threshold = score_threshold
            logger.info(f"Updated score_threshold to {score_threshold}")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the query engine.
        
        Returns:
            Dictionary with statistics
        """
        vector_store_stats = self.vector_store.get_stats()
        embedder_info = self.embedder.get_model_info()
        
        return {
            'top_k': self.top_k,
            'score_threshold': self.score_threshold,
            'rerank_enabled': self.rerank,
            'vector_store': vector_store_stats,
            'embedder': embedder_info
        }


if __name__ == "__main__":
    # Test the query engine
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("QUERY ENGINE TEST")
    print("="*80)
    
    try:
        # Create test data
        print("\nInitializing components...")
        
        # Initialize embedder
        embedder = EmbeddingGenerator(
            model_name="BAAI/bge-large-en-v1.5"
        )
        
        # Create test documents
        test_docs = [
            "Python is a high-level programming language known for its simplicity.",
            "Machine learning enables computers to learn from data without explicit programming.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "Deep learning is a subset of machine learning using multi-layered neural networks.",
            "Natural language processing helps computers understand and generate human language.",
        ]
        
        print(f"Created {len(test_docs)} test documents")
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        embeddings = embedder.embed_documents(test_docs, show_progress=True)
        
        # Create vector store
        print("\nCreating vector store...")
        vector_store = FAISSVectorStore(
            dimension=embedder.embedding_dimension,
            index_type="Flat"
        )
        
        # Add documents
        metadatas = [{'doc_id': i, 'category': 'tech'} for i in range(len(test_docs))]
        vector_store.add(embeddings, test_docs, metadatas)
        
        # Create query engine
        print("\nInitializing query engine...")
        query_engine = QueryEngine(
            vector_store=vector_store,
            embedder=embedder,
            top_k=3,
            score_threshold=0.3
        )
        
        # Display stats
        print("\nQuery Engine Stats:")
        print("-"*80)
        stats = query_engine.get_stats()
        print(f"Top K: {stats['top_k']}")
        print(f"Score Threshold: {stats['score_threshold']}")
        print(f"Vector Store Size: {stats['vector_store']['num_vectors']}")
        
        # Test queries
        test_queries = [
            "What is machine learning?",
            "Tell me about neural networks",
            "How does Python work?"
        ]
        
        for query in test_queries:
            print("\n" + "="*80)
            print(f"QUERY: {query}")
            print("="*80)
            
            # Retrieve documents
            results = query_engine.retrieve(query)
            
            print(f"\nFound {len(results)} relevant documents:")
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. Score: {doc['score']:.4f}")
                print(f"   Text: {doc['text']}")
            
            # Analyze query
            analysis = query_engine.analyze_query(query)
            print(f"\nQuery Analysis:")
            print(f"  Avg Score: {analysis['avg_score']:.4f}")
            print(f"  Max Score: {analysis['max_score']:.4f}")
            print(f"  Has Relevant Results: {analysis['has_relevant_results']}")
        
        # Test context assembly
        print("\n" + "="*80)
        print("TESTING CONTEXT ASSEMBLY")
        print("="*80)
        
        context_result = query_engine.retrieve_with_context(
            "What is deep learning?",
            max_context_length=500
        )
        
        print(f"\nQuery: {context_result['query']}")
        print(f"Documents included: {context_result['num_documents']}")
        print(f"Context length: {len(context_result['context'])} characters")
        print(f"\nAssembled Context:\n{context_result['context'][:300]}...")
        
        print("\n" + "="*80)
        print("Test completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)