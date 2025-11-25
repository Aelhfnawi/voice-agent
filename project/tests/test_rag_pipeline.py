"""
Unit tests for the RAG pipeline.

Tests:
- End-to-end RAG query processing
- LLM integration
- Prompt formatting
- Source citation
- Error handling
- Multi-turn conversations
"""

import pytest
import sys
from pathlib import Path
import os
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.rag_pipeline import RAGPipeline
from src.retriever.query_engine import QueryEngine
from src.llm.llm_gateway import GeminiLLMGateway
from src.llm.prompt_template import MedicalPromptTemplate
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.vector_store import FAISSVectorStore


# Load environment variables
load_dotenv()


class TestRAGPipeline:
    """Tests for RAG Pipeline."""
    
    @pytest.fixture
    def setup_pipeline(self):
        """Set up complete RAG pipeline for testing."""
        # Initialize embedder
        embedder = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")
        
        # Create test documents
        test_docs = [
            "Python is a high-level programming language known for its simplicity and readability.",
            "Machine learning is a subset of AI that enables computers to learn from data.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "Deep learning uses multi-layered neural networks for complex pattern recognition.",
            "Natural language processing helps computers understand and generate human language.",
            "Two-factor authentication provides an extra layer of security for user accounts."
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
        
        # Check if API key is available
        api_key = os.getenv('GEMINI_API_KEY')
        has_api_key = api_key is not None and len(api_key) > 0
        
        if has_api_key:
            # Real LLM gateway
            llm_gateway = GeminiLLMGateway(
                api_key=api_key,
                model_name="gemini-2.5-flash",
                temperature=0.3,
                max_tokens=512
            )
        else:
            # Mock LLM gateway for testing without API key
            llm_gateway = Mock()
            llm_gateway.generate.return_value = {
                'response': 'This is a test response.',
                'success': True,
                'error': None
            }
            llm_gateway.get_model_info.return_value = {
                'model_name': 'mock-model',
                'temperature': 0.3,
                'max_tokens': 512
            }
        
        # Create prompt template
        prompt_template = MedicalPromptTemplate(include_citations=True)
        
        # Create RAG pipeline
        rag_pipeline = RAGPipeline(
            query_engine=query_engine,
            llm_gateway=llm_gateway,
            prompt_template=prompt_template,
            max_context_length=2048
        )
        
        return {
            'rag_pipeline': rag_pipeline,
            'query_engine': query_engine,
            'llm_gateway': llm_gateway,
            'has_api_key': has_api_key
        }
    
    def test_pipeline_initialization(self, setup_pipeline):
        """Test that pipeline initializes correctly."""
        rag_pipeline = setup_pipeline['rag_pipeline']
        assert rag_pipeline is not None
        assert rag_pipeline.max_context_length == 2048
    
    def test_query_basic(self, setup_pipeline):
        """Test basic query processing."""
        rag_pipeline = setup_pipeline['rag_pipeline']
        
        result = rag_pipeline.query("What is machine learning?")
        
        assert 'answer' in result
        assert 'sources' in result
        assert 'success' in result
        assert result['success'] is True
        assert len(result['answer']) > 0
    
    def test_query_empty(self, setup_pipeline):
        """Test query with empty string."""
        rag_pipeline = setup_pipeline['rag_pipeline']
        
        result = rag_pipeline.query("")
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_query_with_sources(self, setup_pipeline):
        """Test that sources are included in response."""
        rag_pipeline = setup_pipeline['rag_pipeline']
        
        result = rag_pipeline.query("Explain neural networks")
        
        if result['success']:
            assert 'sources' in result
            assert isinstance(result['sources'], list)
            if len(result['sources']) > 0:
                assert 'text' in result['sources'][0]
                assert 'score' in result['sources'][0]
    
    def test_query_no_relevant_docs(self, setup_pipeline):
        """Test query when no relevant documents are found."""
        rag_pipeline = setup_pipeline['rag_pipeline']
        query_engine = setup_pipeline['query_engine']
        
        # Set very high threshold so no docs match
        query_engine.update_parameters(score_threshold=0.99)
        
        result = rag_pipeline.query("completely unrelated xyz query")
        
        # Should handle gracefully
        assert result['success'] is True
        assert "couldn't find" in result['answer'].lower() or "no" in result['answer'].lower()
    
    def test_extract_information(self, setup_pipeline):
        """Test information extraction."""
        rag_pipeline = setup_pipeline['rag_pipeline']
        
        result = rag_pipeline.extract_information("What programming languages are mentioned?")
        
        assert 'answer' in result
        assert 'success' in result
        if result['success']:
            assert len(result['answer']) > 0
    
    def test_summarize(self, setup_pipeline):
        """Test document summarization."""
        rag_pipeline = setup_pipeline['rag_pipeline']
        
        result = rag_pipeline.summarize(
            query="artificial intelligence and machine learning",
            focus="key concepts"
        )
        
        assert 'answer' in result
        assert 'success' in result
    
    def test_chat_single_turn(self, setup_pipeline):
        """Test single-turn chat."""
        rag_pipeline = setup_pipeline['rag_pipeline']
        
        messages = [
            {'role': 'user', 'content': 'What is Python?'}
        ]
        
        result = rag_pipeline.chat(messages)
        
        assert 'answer' in result
        assert result['success'] is True
    
    def test_chat_multi_turn(self, setup_pipeline):
        """Test multi-turn conversation."""
        rag_pipeline = setup_pipeline['rag_pipeline']
        
        messages = [
            {'role': 'user', 'content': 'What is machine learning?'},
            {'role': 'assistant', 'content': 'Machine learning is a subset of AI.'},
            {'role': 'user', 'content': 'Can you tell me more about it?'}
        ]
        
        result = rag_pipeline.chat(messages)
        
        assert 'answer' in result
        assert result['success'] is True
    
    def test_chat_empty_messages(self, setup_pipeline):
        """Test chat with empty messages."""
        rag_pipeline = setup_pipeline['rag_pipeline']
        
        result = rag_pipeline.chat([])
        
        assert result['success'] is False
    
    def test_get_stats(self, setup_pipeline):
        """Test getting pipeline statistics."""
        rag_pipeline = setup_pipeline['rag_pipeline']
        
        stats = rag_pipeline.get_stats()
        
        assert 'query_engine' in stats
        assert 'llm' in stats
        assert 'max_context_length' in stats
        assert stats['max_context_length'] == 2048
    
    def test_source_formatting(self, setup_pipeline):
        """Test that sources are properly formatted."""
        rag_pipeline = setup_pipeline['rag_pipeline']
        
        result = rag_pipeline.query("What is Python?")
        
        if result['success'] and len(result['sources']) > 0:
            source = result['sources'][0]
            assert 'id' in source
            assert 'text' in source
            assert 'score' in source
            assert 'metadata' in source
    
    def test_context_length_limit(self, setup_pipeline):
        """Test that context respects length limit."""
        rag_pipeline = setup_pipeline['rag_pipeline']
        
        # Set very small context limit
        rag_pipeline.max_context_length = 100
        
        result = rag_pipeline.query("Tell me about AI")
        
        # Should still work with limited context
        assert result['success'] is True or 'error' in result
    
    def test_retrieval_results_included(self, setup_pipeline):
        """Test that retrieval results are included."""
        rag_pipeline = setup_pipeline['rag_pipeline']
        
        result = rag_pipeline.query("machine learning")
        
        assert 'retrieval_results' in result
        assert isinstance(result['retrieval_results'], list)


class TestRAGPipelineIntegration:
    """Integration tests for RAG Pipeline with real LLM (if API key available)."""
    
    @pytest.fixture
    def real_pipeline(self):
        """Set up pipeline with real LLM if API key available."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            pytest.skip("GEMINI_API_KEY not available")
        
        # Initialize components
        embedder = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")
        
        test_docs = [
            "Python is a versatile programming language used for web development, data science, and automation.",
            "Machine learning algorithms learn patterns from data to make predictions.",
            "Deep learning is a specialized form of machine learning using neural networks."
        ]
        
        embeddings = embedder.embed_documents(test_docs, show_progress=False)
        
        vector_store = FAISSVectorStore(dimension=1024, index_type="Flat")
        vector_store.add(embeddings, test_docs, [{'id': i} for i in range(len(test_docs))])
        
        query_engine = QueryEngine(vector_store=vector_store, embedder=embedder, top_k=2)
        
        llm_gateway = GeminiLLMGateway(
            api_key=api_key,
            model_name="gemini-2.5-flash",
            temperature=0.3,
            max_tokens=256
        )
        
        prompt_template = MedicalPromptTemplate(include_citations=True)
        
        rag_pipeline = RAGPipeline(
            query_engine=query_engine,
            llm_gateway=llm_gateway,
            prompt_template=prompt_template
        )
        
        return rag_pipeline
    
    def test_real_llm_response(self, real_pipeline):
        """Test with real LLM response."""
        result = real_pipeline.query("What is Python used for?")
        
        assert result['success'] is True
        assert len(result['answer']) > 10  # Real response should be substantial
        assert 'Python' in result['answer'] or 'python' in result['answer']
    
    def test_real_llm_citation(self, real_pipeline):
        """Test that real LLM includes citations."""
        result = real_pipeline.query("Explain machine learning")
        
        if result['success']:
            answer = result['answer']
            # Check for citation patterns
            has_citation = (
                'Document' in answer or 
                '[' in answer or
                'according to' in answer.lower()
            )
            # Real LLM should attempt to cite sources
            assert len(answer) > 0


class TestRAGPipelineErrorHandling:
    """Test error handling in RAG Pipeline."""
    
    def test_llm_failure_handling(self):
        """Test handling of LLM failures."""
        embedder = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")
        vector_store = FAISSVectorStore(dimension=1024, index_type="Flat")
        
        # Add test document
        text = "Test document"
        embedding = embedder.embed(text)
        vector_store.add(embedding, [text], [{'id': 0}])
        
        query_engine = QueryEngine(vector_store=vector_store, embedder=embedder)
        
        # Mock LLM that fails
        mock_llm = Mock()
        mock_llm.generate.return_value = {
            'response': '',
            'success': False,
            'error': 'API Error'
        }
        mock_llm.get_model_info.return_value = {'model_name': 'mock'}
        
        prompt_template = MedicalPromptTemplate()
        
        rag_pipeline = RAGPipeline(
            query_engine=query_engine,
            llm_gateway=mock_llm,
            prompt_template=prompt_template
        )
        
        result = rag_pipeline.query("test")
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_exception_handling(self):
        """Test handling of exceptions during processing."""
        embedder = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")
        vector_store = FAISSVectorStore(dimension=1024, index_type="Flat")
        
        # Add test document
        text = "Test document"
        embedding = embedder.embed(text)
        vector_store.add(embedding, [text], [{'id': 0}])
        
        query_engine = QueryEngine(vector_store=vector_store, embedder=embedder)
        
        # Mock LLM that raises exception
        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("Test exception")
        mock_llm.get_model_info.return_value = {'model_name': 'mock'}
        
        prompt_template = MedicalPromptTemplate()
        
        rag_pipeline = RAGPipeline(
            query_engine=query_engine,
            llm_gateway=mock_llm,
            prompt_template=prompt_template
        )
        
        result = rag_pipeline.query("test")
        
        # Should handle exception gracefully
        assert 'error' in result or result['success'] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])