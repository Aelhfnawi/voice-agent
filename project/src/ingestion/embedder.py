"""
Embedding generation module for the Technical RAG Assistant.

This module handles generation of embeddings using BAAI/bge-large-en-v1.5:
- Efficient batch processing
- GPU acceleration (if available)
- Caching support
- Normalization

BAAI/bge-large-en-v1.5 produces 1024-dimensional embeddings optimized for retrieval.
"""

import logging
import numpy as np
from typing import List, Union, Optional
import torch

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text using BAAI/bge-large-en-v1.5.
    
    Features:
    - Batch processing for efficiency
    - GPU acceleration when available
    - Automatic normalization
    - Progress tracking for large batches
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True
    ):
        """
        Initialize the EmbeddingGenerator.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for embedding generation. "
                "Install it with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def embed(self, texts: Union[str, List[str]], show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings (shape: [n_texts, embedding_dim])
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.
        
        For BGE models, queries should be prefixed with "Represent this sentence for searching relevant passages:"
        for better retrieval performance.
        
        Args:
            query: Query text
            
        Returns:
            NumPy array of embedding (shape: [embedding_dim])
        """
        # Add instruction prefix for BGE models (improves retrieval)
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"
        
        embedding = self.embed(query, show_progress=False)
        
        return embedding[0]  # Return first (and only) embedding
    
    def embed_documents(self, documents: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for documents.
        
        Args:
            documents: List of document texts
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings (shape: [n_docs, embedding_dim])
        """
        return self.embed(documents, show_progress=show_progress)
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings in batches (useful for very large datasets).
        
        Args:
            texts: List of texts
            batch_size: Override default batch size
            show_progress: Whether to show progress
            
        Returns:
            NumPy array of embeddings
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if show_progress:
                logger.info(f"Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
            
            batch_embeddings = self.embed(batch, show_progress=False)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Ensure embeddings are 1D
        if embedding1.ndim > 1:
            embedding1 = embedding1.flatten()
        if embedding2.ndim > 1:
            embedding2 = embedding2.flatten()
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        return float(similarity)
    
    def compute_similarities(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarities between a query and multiple documents.
        
        Args:
            query_embedding: Query embedding (shape: [embedding_dim])
            doc_embeddings: Document embeddings (shape: [n_docs, embedding_dim])
            
        Returns:
            Array of similarity scores (shape: [n_docs])
        """
        # Ensure query embedding is 1D
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()
        
        # Compute cosine similarities
        similarities = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        return similarities
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'device': self.device,
            'batch_size': self.batch_size,
            'normalize': self.normalize,
            'max_seq_length': self.model.max_seq_length
        }


if __name__ == "__main__":
    # Test the embedder
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("EMBEDDING GENERATOR TEST")
    print("="*80)
    
    # Initialize embedder
    try:
        embedder = EmbeddingGenerator(
            model_name="BAAI/bge-large-en-v1.5",
            batch_size=2
        )
        
        # Display model info
        print("\nModel Information:")
        print("-"*80)
        info = embedder.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Test texts
        test_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing enables computers to understand text.",
        ]
        
        test_query = "What is machine learning?"
        
        # Generate embeddings
        print("\n" + "="*80)
        print("Generating Document Embeddings...")
        print("="*80)
        
        doc_embeddings = embedder.embed_documents(test_texts, show_progress=True)
        print(f"\nDocument embeddings shape: {doc_embeddings.shape}")
        print(f"First embedding preview: {doc_embeddings[0][:5]}...")
        
        # Generate query embedding
        print("\n" + "="*80)
        print("Generating Query Embedding...")
        print("="*80)
        print(f"Query: {test_query}")
        
        query_embedding = embedder.embed_query(test_query)
        print(f"Query embedding shape: {query_embedding.shape}")
        print(f"Query embedding preview: {query_embedding[:5]}...")
        
        # Compute similarities
        print("\n" + "="*80)
        print("Computing Similarities...")
        print("="*80)
        
        similarities = embedder.compute_similarities(query_embedding, doc_embeddings)
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        print(f"\nQuery: {test_query}")
        print("\nRanked Results:")
        for i, idx in enumerate(sorted_indices, 1):
            print(f"\n{i}. Similarity: {similarities[idx]:.4f}")
            print(f"   Text: {test_texts[idx]}")
        
        print("\n" + "="*80)
        print("Test completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during test: {e}")
        print("\nMake sure you have installed: pip install sentence-transformers")
        sys.exit(1)