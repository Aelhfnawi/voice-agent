"""
Configuration module for the Medical RAG Assistant.

This module contains all configuration settings including:
- API keys and credentials
- Model parameters
- Chunking settings
- Vector store configuration
- File paths

Environment variables can override default settings.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Configuration class for the Medical RAG Assistant.
    
    All settings can be overridden using environment variables.
    """
    
    # ==================== API KEYS ====================
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found. Please set it in .env file or environment variables."
        )
    
    # ==================== LIVEKIT SETTINGS ====================
    LIVEKIT_URL: str = os.getenv('LIVEKIT_URL', 'ws://localhost:7880')
    LIVEKIT_API_KEY: str = os.getenv('LIVEKIT_API_KEY', '')
    LIVEKIT_API_SECRET: str = os.getenv('LIVEKIT_API_SECRET', '')
    
    # ==================== MODEL SETTINGS ====================
    # Gemini model configuration
    GEMINI_MODEL: str = os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')
    TEMPERATURE: float = float(os.getenv('TEMPERATURE', '0.3'))
    MAX_TOKENS: int = int(os.getenv('MAX_TOKENS', '2048'))
    
    # Embedding model configuration
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-large-en-v1.5')
    EMBEDDING_DIMENSION: int = int(os.getenv('EMBEDDING_DIMENSION', '1024'))
    # Note: BAAI/bge-large-en-v1.5 produces 1024-dimensional embeddings
    
    # ==================== CHUNKING SETTINGS ====================
    CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', '512'))
    CHUNK_OVERLAP: int = int(os.getenv('CHUNK_OVERLAP', '50'))
    
    # Minimum chunk size (discard smaller chunks)
    MIN_CHUNK_SIZE: int = int(os.getenv('MIN_CHUNK_SIZE', '100'))
    
    # ==================== RETRIEVAL SETTINGS ====================
    # Number of top documents to retrieve
    TOP_K_RESULTS: int = int(os.getenv('TOP_K_RESULTS', '5'))
    
    # Similarity threshold (0-1, higher = more strict)
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.5'))
    
    # ==================== VECTOR STORE SETTINGS ====================
    # FAISS index type: 'Flat', 'IVFFlat', 'HNSW'
    FAISS_INDEX_TYPE: str = os.getenv('FAISS_INDEX_TYPE', 'Flat')
    
    # Number of clusters for IVFFlat (only used if index type is IVFFlat)
    FAISS_NLIST: int = int(os.getenv('FAISS_NLIST', '100'))
    
    # Number of probes for search (only used if index type is IVFFlat)
    FAISS_NPROBE: int = int(os.getenv('FAISS_NPROBE', '10'))
    
    # ==================== FILE PATHS ====================
    # Base project directory
    BASE_DIR: Path = Path(__file__).parent.resolve()
    
    # Data directories
    DATA_DIR: Path = BASE_DIR / 'data'
    RAW_DATA_DIR: Path = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR: Path = DATA_DIR / 'processed'
    
    # Vector store path
    VECTOR_STORE_PATH: Path = DATA_DIR / 'vector_store' / 'faiss_index'
    
    # Logs directory
    LOGS_DIR: Path = BASE_DIR / 'logs'
    
    # ==================== DOCUMENT PROCESSING ====================
    # Supported file extensions
    SUPPORTED_EXTENSIONS: list = ['.pdf', '.docx', '.txt']
    
    # Maximum file size in MB
    MAX_FILE_SIZE_MB: int = int(os.getenv('MAX_FILE_SIZE_MB', '50'))
    
    # Text extraction settings
    PDF_EXTRACTION_METHOD: str = os.getenv('PDF_EXTRACTION_METHOD', 'pypdf')
    # Options: 'pypdf', 'pdfplumber', 'pymupdf'
    
    # ==================== TEXT CLEANING ====================
    # Whether to remove special characters
    REMOVE_SPECIAL_CHARS: bool = os.getenv('REMOVE_SPECIAL_CHARS', 'true').lower() == 'true'
    
    # Whether to normalize whitespace
    NORMALIZE_WHITESPACE: bool = os.getenv('NORMALIZE_WHITESPACE', 'true').lower() == 'true'
    
    # Whether to remove URLs
    REMOVE_URLS: bool = os.getenv('REMOVE_URLS', 'true').lower() == 'true'
    
    # Whether to remove email addresses
    REMOVE_EMAILS: bool = os.getenv('REMOVE_EMAILS', 'false').lower() == 'true'
    
    # Minimum text length to keep (characters)
    MIN_TEXT_LENGTH: int = int(os.getenv('MIN_TEXT_LENGTH', '50'))
    
    # ==================== MEDICAL SAFETY SETTINGS ====================
    # Whether to add medical disclaimer to responses
    ADD_MEDICAL_DISCLAIMER: bool = os.getenv('ADD_MEDICAL_DISCLAIMER', 'true').lower() == 'true'
    
    # Medical disclaimer text
    MEDICAL_DISCLAIMER: str = (
        "\n\n⚕️ Medical Disclaimer: This information is for educational purposes only "
        "and should not replace professional medical advice. Please consult with a "
        "qualified healthcare provider for medical decisions."
    )
    
    # Whether to filter sensitive queries
    FILTER_SENSITIVE_QUERIES: bool = os.getenv('FILTER_SENSITIVE_QUERIES', 'true').lower() == 'true'
    
    # ==================== PERFORMANCE SETTINGS ====================
    # Batch size for embedding generation
    EMBEDDING_BATCH_SIZE: int = int(os.getenv('EMBEDDING_BATCH_SIZE', '32'))
    
    # Number of workers for parallel processing
    NUM_WORKERS: int = int(os.getenv('NUM_WORKERS', '4'))
    
    # Cache embeddings to disk
    CACHE_EMBEDDINGS: bool = os.getenv('CACHE_EMBEDDINGS', 'true').lower() == 'true'
    
    # ==================== LOGGING SETTINGS ====================
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE: Path = LOGS_DIR / 'medical_rag.log'
    
    # ==================== ADVANCED RAG SETTINGS ====================
    # Whether to use hybrid search (keyword + semantic)
    USE_HYBRID_SEARCH: bool = os.getenv('USE_HYBRID_SEARCH', 'false').lower() == 'true'
    
    # Weight for semantic search in hybrid mode (0-1)
    SEMANTIC_WEIGHT: float = float(os.getenv('SEMANTIC_WEIGHT', '0.7'))
    
    # Whether to rerank retrieved results
    USE_RERANKING: bool = os.getenv('USE_RERANKING', 'false').lower() == 'true'
    
    # Reranking model (if enabled)
    RERANKING_MODEL: str = os.getenv('RERANKING_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Maximum context length to send to LLM (in tokens)
    MAX_CONTEXT_LENGTH: int = int(os.getenv('MAX_CONTEXT_LENGTH', '4096'))
    
    # ==================== PROMPT SETTINGS ====================
    # System prompt prefix
    SYSTEM_PROMPT_PREFIX: str = os.getenv(
        'SYSTEM_PROMPT_PREFIX',
        'You are a helpful medical assistant. Provide accurate, evidence-based information.'
    )
    
    # Whether to include source citations in responses
    INCLUDE_CITATIONS: bool = os.getenv('INCLUDE_CITATIONS', 'true').lower() == 'true'
    
    # Citation format: 'inline', 'endnotes', 'none'
    CITATION_FORMAT: str = os.getenv('CITATION_FORMAT', 'inline')
    
    
    @classmethod
    def create_directories(cls):
        """
        Create necessary directories if they don't exist.
        """
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.VECTOR_STORE_PATH.parent,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate(cls):
        """
        Validate configuration settings.
        
        Raises:
            ValueError: If any configuration is invalid
        """
        # Validate API key
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")
        
        # Validate numeric ranges
        if not 0 <= cls.TEMPERATURE <= 2:
            raise ValueError("TEMPERATURE must be between 0 and 2")
        
        if cls.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        
        if cls.TOP_K_RESULTS <= 0:
            raise ValueError("TOP_K_RESULTS must be positive")
        
        if not 0 <= cls.SIMILARITY_THRESHOLD <= 1:
            raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")
        
        if not 0 <= cls.SEMANTIC_WEIGHT <= 1:
            raise ValueError("SEMANTIC_WEIGHT must be between 0 and 1")
        
        # Validate index type
        valid_index_types = ['Flat', 'IVFFlat', 'HNSW']
        if cls.FAISS_INDEX_TYPE not in valid_index_types:
            raise ValueError(f"FAISS_INDEX_TYPE must be one of {valid_index_types}")
        
        # Validate citation format
        valid_citation_formats = ['inline', 'endnotes', 'none']
        if cls.CITATION_FORMAT not in valid_citation_formats:
            raise ValueError(f"CITATION_FORMAT must be one of {valid_citation_formats}")
    
    @classmethod
    def display(cls):
        """
        Display current configuration settings.
        """
        print("="*80)
        print("MEDICAL RAG ASSISTANT - CONFIGURATION")
        print("="*80)
        print(f"\n📊 MODEL SETTINGS:")
        print(f"  Gemini Model: {cls.GEMINI_MODEL}")
        print(f"  Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"  Temperature: {cls.TEMPERATURE}")
        print(f"  Max Tokens: {cls.MAX_TOKENS}")
        
        print(f"\n📄 DOCUMENT PROCESSING:")
        print(f"  Chunk Size: {cls.CHUNK_SIZE}")
        print(f"  Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"  Supported Extensions: {', '.join(cls.SUPPORTED_EXTENSIONS)}")
        
        print(f"\n🔍 RETRIEVAL SETTINGS:")
        print(f"  Top K Results: {cls.TOP_K_RESULTS}")
        print(f"  Similarity Threshold: {cls.SIMILARITY_THRESHOLD}")
        print(f"  FAISS Index Type: {cls.FAISS_INDEX_TYPE}")
        
        print(f"\n📁 FILE PATHS:")
        print(f"  Raw Data: {cls.RAW_DATA_DIR}")
        print(f"  Vector Store: {cls.VECTOR_STORE_PATH}")
        print(f"  Logs: {cls.LOGS_DIR}")
        
        print(f"\n⚕️ MEDICAL SAFETY:")
        print(f"  Add Disclaimer: {cls.ADD_MEDICAL_DISCLAIMER}")
        print(f"  Filter Sensitive Queries: {cls.FILTER_SENSITIVE_QUERIES}")
        print(f"  Include Citations: {cls.INCLUDE_CITATIONS}")
        
        print("="*80 + "\n")


# Create directories on import
Config.create_directories()

# Validate configuration
Config.validate()


if __name__ == "__main__":
    # Display configuration when run directly
    Config.display()