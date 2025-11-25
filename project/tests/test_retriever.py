"""
Unit tests for the retriever components.

Tests:
- Query engine initialization
- Document retrieval
- Context assembly
- Query analysis
- Similarity search
"""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.retriever.query_engine import QueryEngine
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.vector_store import FAISSVectorStore


class TestQueryEngine:
    """Tests for QueryEngine."""
    
    @pytest.fixture
    def setup_components(self):
        """Set up embedder, vector store, and query engine for testing."""
        # Initialize embedder
        embedder = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")
        
        # Create test documents
        test_docs = [
            "Python is a high-level programming language known for its simplicity.",
            "Machine learning enables computers to learn from data without explicit programming.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "Deep learning uses multi-layered neural networks for complex pattern recognition.",
            "Natural language processing helps computers understand and generate human language."
        ]
        
        # Generate embeddings
        embeddings = embedder.embed_documents(test_docs, show_progress=False)
        
        # Create vector store
        vector_store = FAISSVectorStore(
            dimension=embedder.embedding_dimension,
            index_type="Flat"
        )
        
        # Add documents
        metadatas = [{'doc_id': i, 'source': 'test'} for i in range(len(test_docs))]
        vector_store.add(embeddings, test_docs, metadatas)
        
        # Create query engine
        query_engine = QueryEngine(
            vector_store=vector_store,
            embedder=embedder,
            top_k=3,
            score_threshold=0.3
        )
        
        return {
            'query_engine': query_engine,
            'vector_store': vector_store,
            'embedder': embedder,
            'test_docs': test_docs
        }
    
    def test_query_engine_initialization(self, setup_components):
        """Test that query engine initializes correctly."""
        query_engine = setup_components['query_engine']
        assert query_engine is not None
        assert query_engine.top_k == 3
        assert query_engine.score_threshold == 0.3
    
    def test_retrieve_documents(self, setup_components):
        """Test basic document retrieval."""
        query_engine = setup_components['query_engine']
        
        results = query_engine.retrieve("What is machine learning?")
        
        assert len(results) > 0
        assert len(results) <= 3  # top_k=3
        assert all('text' in doc for doc in results)
        assert all('score' in doc for doc in results)
        assert all('metadata' in doc for doc in results)
    
    def test_retrieve_empty_query(self, setup_components):
        """Test retrieval with empty query."""
        query_engine = setup_components['query_engine']
        
        results = query_engine.retrieve("")
        assert results == []
    
    def test_retrieve_with_context(self, setup_components):
        """Test context assembly."""
        query_engine = setup_components['query_engine']
        
        result = query_engine.retrieve_with_context(
            "What is Python?",
            max_context_length=500
        )
        
        assert 'query' in result
        assert 'documents' in result
        assert 'context' in result
        assert 'num_documents' in result
        assert len(result['context']) > 0
        assert result['num_documents'] > 0
    
    def test_retrieve_with_context_limit(self, setup_components):
        """Test that context respects length limit."""
        query_engine = setup_components['query_engine']
        
        result = query_engine.retrieve_with_context(
            "Tell me about AI",
            max_context_length=100
        )
        
        assert len(result['context']) <= 100
    
    def test_relevance_scores(self, setup_components):
        """Test that relevance scores are within valid range."""
        query_engine = setup_components['query_engine']
        
        results = query_engine.retrieve("neural networks")
        
        for doc in results:
            assert 0 <= doc['score'] <= 1
    
    def test_top_k_parameter(self, setup_components):
        """Test that top_k parameter works correctly."""
        query_engine = setup_components['query_engine']
        
        # Test with different top_k values
        query_engine.update_parameters(top_k=2)
        results = query_engine.retrieve("machine learning")
        assert len(results) <= 2
        
        query_engine.update_parameters(top_k=5)
        results = query_engine.retrieve("machine learning")
        assert len(results) <= 5
    
    def test_score_threshold(self, setup_components):
        """Test that score threshold filters results."""
        query_engine = setup_components['query_engine']
        
        # Set high threshold
        query_engine.update_parameters(score_threshold=0.9)
        results = query_engine.retrieve("completely unrelated query xyz")
        
        # Should have fewer or no results with high threshold
        assert all(doc['score'] >= 0.9 for doc in results)
    
    def test_similar_documents(self, setup_components):
        """Test finding similar documents."""
        query_engine = setup_components['query_engine']
        test_docs = setup_components['test_docs']
        
        # Find documents similar to first test doc
        results = query_engine.get_similar_documents(test_docs[0], top_k=2)
        
        assert len(results) > 0
        assert len(results) <= 2
        # Should not return the exact same document
        assert all(doc['text'] != test_docs[0] for doc in results)
    
    def test_analyze_query(self, setup_components):
        """Test query analysis."""
        query_engine = setup_components['query_engine']
        
        analysis = query_engine.analyze_query("What is deep learning?")
        
        assert 'query' in analysis
        assert 'num_results' in analysis
        assert 'avg_score' in analysis
        assert 'max_score' in analysis
        assert 'has_relevant_results' in analysis
        assert isinstance(analysis['has_relevant_results'], bool)
    
    def test_batch_retrieve(self, setup_components):
        """Test batch retrieval."""
        query_engine = setup_components['query_engine']
        
        queries = [
            "What is Python?",
            "Explain machine learning",
            "What are neural networks?"
        ]
        
        results = query_engine.batch_retrieve(queries)
        
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)
    
    def test_update_parameters(self, setup_components):
        """Test parameter updates."""
        query_engine = setup_components['query_engine']
        
        # Update parameters
        query_engine.update_parameters(top_k=5, score_threshold=0.7)
        
        assert query_engine.top_k == 5
        assert query_engine.score_threshold == 0.7
    
    def test_get_stats(self, setup_components):
        """Test getting query engine statistics."""
        query_engine = setup_components['query_engine']
        
        stats = query_engine.get_stats()
        
        assert 'top_k' in stats
        assert 'score_threshold' in stats
        assert 'vector_store' in stats
        assert 'embedder' in stats
        assert stats['vector_store']['num_vectors'] == 5
    
    def test_retrieve_ordering(self, setup_components):
        """Test that results are ordered by relevance."""
        query_engine = setup_components['query_engine']
        
        results = query_engine.retrieve("machine learning algorithms")
        
        # Check that scores are in descending order
        scores = [doc['score'] for doc in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_metadata_preservation(self, setup_components):
        """Test that metadata is preserved in results."""
        query_engine = setup_components['query_engine']
        
        results = query_engine.retrieve("Python programming")
        
        for doc in results:
            assert 'metadata' in doc
            assert 'doc_id' in doc['metadata']
            assert 'source' in doc['metadata']
    
    def test_context_formatting(self, setup_components):
        """Test that context is properly formatted."""
        query_engine = setup_components['query_engine']
        
        result = query_engine.retrieve_with_context("machine learning")
        
        context = result['context']
        # Check that documents are numbered
        assert '[Document 1]' in context or 'Document 1' in context or len(context) > 0
    
    def test_empty_vector_store(self):
        """Test query engine with empty vector store."""
        embedder = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")
        vector_store = FAISSVectorStore(dimension=1024, index_type="Flat")
        
        query_engine = QueryEngine(
            vector_store=vector_store,
            embedder=embedder,
            top_k=3
        )
        
        results = query_engine.retrieve("test query")
        assert results == []


class TestQueryEngineEdgeCases:
    """Test edge cases for QueryEngine."""
    
    @pytest.fixture
    def minimal_setup(self):
        """Set up minimal test environment."""
        embedder = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")
        vector_store = FAISSVectorStore(dimension=1024, index_type="Flat")
        
        # Add single document
        text = "This is a single test document."
        embedding = embedder.embed(text)
        vector_store.add(embedding, [text], [{'id': 0}])
        
        query_engine = QueryEngine(
            vector_store=vector_store,
            embedder=embedder,
            top_k=5
        )
        
        return query_engine
    
    def test_top_k_exceeds_documents(self, minimal_setup):
        """Test when top_k is larger than number of documents."""
        query_engine = minimal_setup
        
        # Request more documents than available
        results = query_engine.retrieve("test", )
        
        # Should return only available document
        assert len(results) == 1
    
    def test_very_long_query(self, minimal_setup):
        """Test with very long query."""
        query_engine = minimal_setup
        
        long_query = "test " * 1000  # Very long query
        results = query_engine.retrieve(long_query)
        
        # Should still work
        assert isinstance(results, list)
    
    def test_special_characters_query(self, minimal_setup):
        """Test query with special characters."""
        query_engine = minimal_setup
        
        special_query = "test @#$% special !@# characters"
        results = query_engine.retrieve(special_query)
        
        # Should handle gracefully
        assert isinstance(results, list)
    
    def test_unicode_query(self, minimal_setup):
        """Test query with unicode characters."""
        query_engine = minimal_setup
        
        unicode_query = "test 你好 мир 🌍"
        results = query_engine.retrieve(unicode_query)
        
        # Should handle gracefully
        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])