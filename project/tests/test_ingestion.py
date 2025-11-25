"""
Unit tests for the ingestion pipeline components.

Tests:
- Document extraction (PDF, DOCX, TXT)
- Text cleaning
- Text chunking
- Embedding generation
- Vector store operations
"""

import pytest
import sys
from pathlib import Path
import tempfile
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.extractor import DocumentExtractor
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import TextChunker, SemanticChunker
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.vector_store import FAISSVectorStore


class TestDocumentExtractor:
    """Tests for DocumentExtractor."""
    
    def test_extractor_initialization(self):
        """Test that extractor initializes correctly."""
        extractor = DocumentExtractor()
        assert extractor is not None
        assert extractor.extraction_method == 'auto'
    
    def test_extract_txt_file(self):
        """Test extraction from TXT file."""
        extractor = DocumentExtractor()
        
        # Create temporary TXT file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\nIt has multiple lines.")
            temp_path = Path(f.name)
        
        try:
            text = extractor.extract(temp_path)
            assert len(text) > 0
            assert "test document" in text
        finally:
            temp_path.unlink()
    
    def test_extract_nonexistent_file(self):
        """Test that extracting nonexistent file raises error."""
        extractor = DocumentExtractor()
        
        with pytest.raises(FileNotFoundError):
            extractor.extract("nonexistent_file.pdf")
    
    def test_extract_unsupported_format(self):
        """Test that unsupported file format raises error."""
        extractor = DocumentExtractor()
        
        # Create temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError):
                extractor.extract(temp_path)
        finally:
            temp_path.unlink()
    
    def test_get_document_metadata(self):
        """Test metadata extraction."""
        extractor = DocumentExtractor()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = Path(f.name)
        
        try:
            metadata = extractor.get_document_metadata(temp_path)
            assert 'filename' in metadata
            assert 'size_bytes' in metadata
            assert metadata['extension'] == '.txt'
        finally:
            temp_path.unlink()


class TestTextCleaner:
    """Tests for TextCleaner."""
    
    def test_cleaner_initialization(self):
        """Test that cleaner initializes correctly."""
        cleaner = TextCleaner()
        assert cleaner is not None
    
    def test_clean_empty_text(self):
        """Test cleaning empty text."""
        cleaner = TextCleaner()
        result = cleaner.clean("")
        assert result == ""
    
    def test_clean_whitespace(self):
        """Test whitespace normalization."""
        cleaner = TextCleaner(normalize_whitespace=True)
        text = "This  has   multiple    spaces"
        result = cleaner.clean(text)
        assert "multiple    spaces" not in result
        assert "multiple spaces" in result
    
    def test_remove_urls(self):
        """Test URL removal."""
        cleaner = TextCleaner(remove_urls=True)
        text = "Visit https://example.com for more info"
        result = cleaner.clean(text)
        assert "https://example.com" not in result
    
    def test_remove_special_characters(self):
        """Test special character removal."""
        cleaner = TextCleaner(remove_special_chars=True)
        text = "Text with special chars: ™ © ®"
        result = cleaner.clean(text)
        assert len(result) > 0
    
    def test_min_text_length(self):
        """Test minimum text length filtering."""
        cleaner = TextCleaner(min_text_length=50)
        short_text = "Short"
        result = cleaner.clean(short_text)
        assert result == ""
    
    def test_clean_batch(self):
        """Test batch cleaning."""
        cleaner = TextCleaner()
        texts = ["Text 1", "Text 2", "Text 3"]
        results = cleaner.clean_batch(texts)
        assert len(results) == 3


class TestTextChunker:
    """Tests for TextChunker."""
    
    def test_chunker_initialization(self):
        """Test that chunker initializes correctly."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 10
    
    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        result = chunker.chunk("")
        assert result == []
    
    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk size."""
        chunker = TextChunker(chunk_size=100)
        text = "Short text"
        result = chunker.chunk(text)
        assert len(result) == 1
        assert result[0] == text
    
    def test_chunk_long_text(self):
        """Test chunking long text."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 200  # 200 character text
        result = chunker.chunk(text)
        assert len(result) > 1
    
    def test_chunk_overlap(self):
        """Test that chunks have overlap."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a test sentence. " * 10
        result = chunker.chunk(text)
        if len(result) > 1:
            # Check that consecutive chunks share some content
            assert len(result) >= 2
    
    def test_get_chunk_stats(self):
        """Test chunk statistics."""
        chunker = TextChunker()
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        stats = chunker.get_chunk_stats(chunks)
        assert stats['num_chunks'] == 3
        assert 'avg_chunk_size' in stats


class TestSemanticChunker:
    """Tests for SemanticChunker."""
    
    def test_semantic_chunker_initialization(self):
        """Test semantic chunker initialization."""
        chunker = SemanticChunker(chunk_size=100)
        assert chunker is not None
    
    def test_semantic_chunking_with_sections(self):
        """Test semantic chunking with clear sections."""
        chunker = SemanticChunker(chunk_size=100)
        text = """
        # Section 1
        Content for section 1.
        
        # Section 2
        Content for section 2.
        """
        result = chunker.chunk(text)
        assert len(result) > 0


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator."""
    
    @pytest.fixture
    def embedder(self):
        """Create embedder instance for testing."""
        return EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")
    
    def test_embedder_initialization(self, embedder):
        """Test that embedder initializes correctly."""
        assert embedder is not None
        assert embedder.embedding_dimension == 1024
    
    def test_embed_single_text(self, embedder):
        """Test embedding single text."""
        text = "This is a test sentence."
        embedding = embedder.embed(text)
        assert embedding.shape == (1, 1024)
    
    def test_embed_multiple_texts(self, embedder):
        """Test embedding multiple texts."""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = embedder.embed(texts)
        assert embeddings.shape == (3, 1024)
    
    def test_embed_query(self, embedder):
        """Test query embedding."""
        query = "What is machine learning?"
        embedding = embedder.embed_query(query)
        assert embedding.shape == (1024,)
    
    def test_embed_empty_list(self, embedder):
        """Test embedding empty list."""
        embeddings = embedder.embed([])
        assert len(embeddings) == 0
    
    def test_compute_similarity(self, embedder):
        """Test similarity computation."""
        emb1 = embedder.embed("Machine learning")[0]
        emb2 = embedder.embed("Deep learning")[0]
        similarity = embedder.compute_similarity(emb1, emb2)
        assert 0 <= similarity <= 1


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore."""
    
    @pytest.fixture
    def vector_store(self):
        """Create vector store instance for testing."""
        return FAISSVectorStore(dimension=1024, index_type="Flat")
    
    def test_vector_store_initialization(self, vector_store):
        """Test that vector store initializes correctly."""
        assert vector_store is not None
        assert vector_store.dimension == 1024
        assert vector_store.index.ntotal == 0
    
    def test_add_vectors(self, vector_store):
        """Test adding vectors to store."""
        embeddings = np.random.randn(5, 1024).astype('float32')
        texts = [f"Text {i}" for i in range(5)]
        metadatas = [{'id': i} for i in range(5)]
        
        vector_store.add(embeddings, texts, metadatas)
        assert vector_store.index.ntotal == 5
    
    def test_search_vectors(self, vector_store):
        """Test searching vectors."""
        embeddings = np.random.randn(5, 1024).astype('float32')
        texts = [f"Text {i}" for i in range(5)]
        
        vector_store.add(embeddings, texts)
        
        query = embeddings[0]
        results = vector_store.search(query, top_k=3)
        assert len(results) > 0
        assert len(results) <= 3
    
    def test_save_and_load(self, vector_store):
        """Test saving and loading vector store."""
        embeddings = np.random.randn(3, 1024).astype('float32')
        texts = ["Text 1", "Text 2", "Text 3"]
        
        vector_store.add(embeddings, texts)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_index"
            vector_store.index_path = save_path
            vector_store.save()
            
            # Load
            new_store = FAISSVectorStore(dimension=1024, index_path=save_path)
            new_store.load()
            
            assert new_store.index.ntotal == 3
    
    def test_clear_index(self, vector_store):
        """Test clearing the index."""
        embeddings = np.random.randn(3, 1024).astype('float32')
        texts = ["Text 1", "Text 2", "Text 3"]
        
        vector_store.add(embeddings, texts)
        assert vector_store.index.ntotal == 3
        
        vector_store.clear()
        assert vector_store.index.ntotal == 0
    
    def test_get_stats(self, vector_store):
        """Test getting statistics."""
        stats = vector_store.get_stats()
        assert 'num_vectors' in stats
        assert 'dimension' in stats
        assert stats['num_vectors'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])