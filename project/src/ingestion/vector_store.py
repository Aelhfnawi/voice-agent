"""
FAISS vector store module for the Technical RAG Assistant.

This module handles vector storage and retrieval using FAISS:
- Multiple index types (Flat, IVFFlat, HNSW)
- Efficient similarity search
- Persistence (save/load)
- Metadata management

FAISS provides fast similarity search for large-scale vector databases.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("faiss not installed. Install with: pip install faiss-cpu or faiss-gpu")

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Vector store using FAISS for efficient similarity search.
    
    Supports multiple index types and metadata storage.
    """
    
    def __init__(
        self,
        dimension: int = 1024,
        index_type: str = "Flat",
        index_path: Optional[Union[str, Path]] = None,
        nlist: int = 100,
        nprobe: int = 10
    ):
        """
        Initialize the FAISS vector store.
        
        Args:
            dimension: Dimension of embeddings (1024 for bge-large-en-v1.5)
            index_type: Type of FAISS index ('Flat', 'IVFFlat', 'HNSW')
            index_path: Path to save/load index
            nlist: Number of clusters for IVFFlat (only used if index_type='IVFFlat')
            nprobe: Number of clusters to search (only used if index_type='IVFFlat')
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss is required for vector storage. "
                "Install it with: pip install faiss-cpu (or faiss-gpu for GPU support)"
            )
        
        self.dimension = dimension
        self.index_type = index_type
        self.index_path = Path(index_path) if index_path else None
        self.nlist = nlist
        self.nprobe = nprobe
        
        # Initialize index
        self.index = None
        self.texts: List[str] = []
        self.metadatas: List[dict] = []
        
        self._create_index()
        
        logger.info(f"FAISSVectorStore initialized: type={index_type}, dimension={dimension}")
    
    def _create_index(self):
        """Create a new FAISS index based on index_type."""
        if self.index_type == "Flat":
            # Simple flat index (exact search, no training needed)
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("Created Flat index (exact search)")
            
        elif self.index_type == "IVFFlat":
            # IVF (Inverted File) index (approximate search, requires training)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            self.index.nprobe = self.nprobe
            logger.info(f"Created IVFFlat index (nlist={self.nlist}, nprobe={self.nprobe})")
            
        elif self.index_type == "HNSW":
            # HNSW (Hierarchical Navigable Small World) index
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            logger.info("Created HNSW index")
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def add(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadatas: Optional[List[dict]] = None
    ):
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings: NumPy array of embeddings (shape: [n, dimension])
            texts: List of corresponding text chunks
            metadatas: List of metadata dictionaries for each text
        """
        if embeddings.shape[0] != len(texts):
            raise ValueError("Number of embeddings must match number of texts")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Train index if needed (IVFFlat requires training)
        if self.index_type == "IVFFlat" and not self.index.is_trained:
            logger.info("Training IVFFlat index...")
            # Use at least nlist embeddings for training
            if embeddings.shape[0] >= self.nlist:
                self.index.train(embeddings)
            else:
                logger.warning(
                    f"Not enough embeddings to train IVFFlat index "
                    f"(have {embeddings.shape[0]}, need {self.nlist}). "
                    f"Using all available embeddings for training."
                )
                self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store texts and metadata
        self.texts.extend(texts)
        
        if metadatas is None:
            metadatas = [{}] * len(texts)
        self.metadatas.extend(metadatas)
        
        logger.info(f"Added {len(texts)} vectors to index. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[str, dict, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding (shape: [dimension])
            top_k: Number of results to return
            score_threshold: Minimum similarity score (None for no threshold)
            
        Returns:
            List of tuples: (text, metadata, distance)
            Distance is L2 distance (lower is more similar)
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, no results to return")
            return []
        
        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Search
        top_k = min(top_k, self.index.ntotal)  # Can't return more than we have
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Flatten results (since query is single vector)
        distances = distances[0]
        indices = indices[0]
        
        # Convert to list of results
        results = []
        for distance, idx in zip(distances, indices):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            # Convert L2 distance to similarity score (inverse)
            # Normalized embeddings: similarity = 1 - (distance^2 / 4)
            similarity = 1.0 - (distance / 4.0)
            
            # Apply threshold if specified
            if score_threshold is not None and similarity < score_threshold:
                continue
            
            results.append((
                self.texts[idx],
                self.metadatas[idx],
                float(similarity)
            ))
        
        logger.debug(f"Search returned {len(results)} results")
        return results
    
    def search_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[List[Tuple[str, dict, float]]]:
        """
        Search for multiple queries.
        
        Args:
            query_embeddings: Query embeddings (shape: [n_queries, dimension])
            top_k: Number of results per query
            score_threshold: Minimum similarity score
            
        Returns:
            List of result lists, one per query
        """
        all_results = []
        
        for query_embedding in query_embeddings:
            results = self.search(query_embedding, top_k, score_threshold)
            all_results.append(results)
        
        return all_results
    
    def save(self, path: Optional[Union[str, Path]] = None):
        """
        Save the index and metadata to disk.
        
        Args:
            path: Path to save to (uses self.index_path if not specified)
        """
        if path is None:
            if self.index_path is None:
                raise ValueError("No save path specified")
            path = self.index_path
        else:
            path = Path(path)
        
        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = str(path) + ".index"
        faiss.write_index(self.index, index_file)
        
        # Save texts and metadata
        data_file = str(path) + ".pkl"
        with open(data_file, 'wb') as f:
            pickle.dump({
                'texts': self.texts,
                'metadatas': self.metadatas,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'nlist': self.nlist,
                'nprobe': self.nprobe
            }, f)
        
        logger.info(f"Saved index to {path}")
    
    def load(self, path: Optional[Union[str, Path]] = None):
        """
        Load the index and metadata from disk.
        
        Args:
            path: Path to load from (uses self.index_path if not specified)
        """
        if path is None:
            if self.index_path is None:
                raise ValueError("No load path specified")
            path = self.index_path
        else:
            path = Path(path)
        
        # Load FAISS index
        index_file = str(path) + ".index"
        if not Path(index_file).exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        self.index = faiss.read_index(index_file)
        
        # Load texts and metadata
        data_file = str(path) + ".pkl"
        if not Path(data_file).exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            self.texts = data['texts']
            self.metadatas = data['metadatas']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
            self.nlist = data.get('nlist', 100)
            self.nprobe = data.get('nprobe', 10)
        
        # Set nprobe for IVFFlat index
        if self.index_type == "IVFFlat":
            self.index.nprobe = self.nprobe
        
        logger.info(f"Loaded index from {path} ({self.index.ntotal} vectors)")
    
    def index_exists(self, path: Optional[Union[str, Path]] = None) -> bool:
        """
        Check if an index exists at the given path.
        
        Args:
            path: Path to check (uses self.index_path if not specified)
            
        Returns:
            True if index exists, False otherwise
        """
        if path is None:
            if self.index_path is None:
                return False
            path = self.index_path
        else:
            path = Path(path)
        
        index_file = Path(str(path) + ".index")
        data_file = Path(str(path) + ".pkl")
        
        return index_file.exists() and data_file.exists()
    
    def clear(self):
        """Clear the index and all stored data."""
        self._create_index()
        self.texts = []
        self.metadatas = []
        logger.info("Index cleared")
    
    def get_stats(self) -> dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'num_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            'num_texts': len(self.texts),
            'num_metadatas': len(self.metadatas)
        }


if __name__ == "__main__":
    # Test the vector store
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("FAISS VECTOR STORE TEST")
    print("="*80)
    
    try:
        # Create test embeddings
        dimension = 1024
        num_vectors = 10
        
        np.random.seed(42)
        test_embeddings = np.random.randn(num_vectors, dimension).astype('float32')
        test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)  # Normalize
        
        test_texts = [f"This is test document {i}" for i in range(num_vectors)]
        test_metadatas = [{'doc_id': i, 'category': f'category_{i % 3}'} for i in range(num_vectors)]
        
        # Create vector store
        print("\nCreating vector store...")
        vector_store = FAISSVectorStore(
            dimension=dimension,
            index_type="Flat",
            index_path="test_index"
        )
        
        # Add vectors
        print("\nAdding vectors...")
        vector_store.add(test_embeddings, test_texts, test_metadatas)
        
        # Get stats
        print("\nVector Store Stats:")
        print("-"*80)
        stats = vector_store.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Search
        print("\n" + "="*80)
        print("Testing Search...")
        print("="*80)
        
        query_embedding = test_embeddings[0]  # Use first embedding as query
        results = vector_store.search(query_embedding, top_k=3)
        
        print(f"\nQuery: {test_texts[0]}")
        print("\nTop 3 Results:")
        for i, (text, metadata, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.4f}")
            print(f"   Text: {text}")
            print(f"   Metadata: {metadata}")
        
        # Save and load
        print("\n" + "="*80)
        print("Testing Save/Load...")
        print("="*80)
        
        print("\nSaving index...")
        vector_store.save()
        
        print("Creating new vector store and loading index...")
        new_store = FAISSVectorStore(
            dimension=dimension,
            index_type="Flat",
            index_path="test_index"
        )
        new_store.load()
        
        print("\nLoaded Vector Store Stats:")
        print("-"*80)
        loaded_stats = new_store.get_stats()
        for key, value in loaded_stats.items():
            print(f"{key}: {value}")
        
        # Search with loaded index
        print("\nSearching with loaded index...")
        loaded_results = new_store.search(query_embedding, top_k=3)
        print(f"Found {len(loaded_results)} results")
        
        # Clean up
        import os
        if Path("test_index.index").exists():
            os.remove("test_index.index")
        if Path("test_index.pkl").exists():
            os.remove("test_index.pkl")
        print("\nTest files cleaned up")
        
        print("\n" + "="*80)
        print("Test completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during test: {e}")
        print("\nMake sure you have installed: pip install faiss-cpu")
        import traceback
        traceback.print_exc()
        sys.exit(1)