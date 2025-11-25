"""
Main entry point for the Medical RAG Assistant.

This module orchestrates the complete RAG pipeline:
- Configuration loading
- Document ingestion (PDF, DOCX, TXT)
- Vector store initialization with FAISS
- LLM integration with Gemini
- RAG query interface

Usage:
    python main.py --query "What are the symptoms of diabetes?"
    python main.py --ingest --data-dir ./data/raw
    python main.py --rebuild-index
"""

import logging
import argparse
import sys
from pathlib import Path
from typing import Optional

# Project imports
from config import Config
from src.ingestion.extractor import DocumentExtractor
from src.ingestion.cleaner import TextCleaner
from src.ingestion.chunker import TextChunker
from src.ingestion.embedder import EmbeddingGenerator
from src.ingestion.vector_store import FAISSVectorStore
from src.retriever.query_engine import QueryEngine
from src.llm.llm_gateway import GeminiLLMGateway
from src.llm.prompt_template import MedicalPromptTemplate
from src.pipeline.rag_pipeline import RAGPipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medical_rag.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MedicalRAGAssistant:
    """
    Main application class for the Medical RAG Assistant.
    
    Handles initialization, ingestion, and query workflows.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the Medical RAG Assistant.
        
        Args:
            config: Configuration object containing all settings
        """
        self.config = config
        self.vector_store: Optional[FAISSVectorStore] = None
        self.query_engine: Optional[QueryEngine] = None
        self.llm_gateway: Optional[GeminiLLMGateway] = None
        self.rag_pipeline: Optional[RAGPipeline] = None
        
        logger.info("Initializing Medical RAG Assistant")
        
    def setup_ingestion_pipeline(self):
        """
        Set up the document ingestion pipeline.
        
        Returns:
            Tuple of (extractor, cleaner, chunker, embedder, vector_store)
        """
        logger.info("Setting up ingestion pipeline...")
        
        # Initialize components
        extractor = DocumentExtractor()
        cleaner = TextCleaner()
        chunker = TextChunker(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        embedder = EmbeddingGenerator(
            model_name=self.config.EMBEDDING_MODEL
        )
        vector_store = FAISSVectorStore(
            dimension=self.config.EMBEDDING_DIMENSION,
            index_path=self.config.VECTOR_STORE_PATH
        )
        
        logger.info("Ingestion pipeline ready")
        return extractor, cleaner, chunker, embedder, vector_store
    
    def ingest_documents(self, data_dir: Path, rebuild: bool = False):
        """
        Ingest documents from the data directory.
        
        Args:
            data_dir: Path to directory containing documents
            rebuild: If True, rebuild the entire index from scratch
        """
        logger.info(f"Starting document ingestion from {data_dir}")
        
        try:
            # Setup pipeline
            extractor, cleaner, chunker, embedder, vector_store = self.setup_ingestion_pipeline()
            
            # Load existing index or create new
            if not rebuild and vector_store.index_exists():
                logger.info("Loading existing vector store...")
                vector_store.load()
            else:
                logger.info("Creating new vector store...")
            
            # Find all supported documents
            supported_extensions = ['.pdf', '.docx', '.txt']
            documents = []
            for ext in supported_extensions:
                documents.extend(data_dir.glob(f'**/*{ext}'))
            
            if not documents:
                logger.warning(f"No documents found in {data_dir}")
                return
            
            logger.info(f"Found {len(documents)} documents to process")
            
            # Process each document
            all_chunks = []
            all_metadatas = []
            
            for doc_path in documents:
                try:
                    logger.info(f"Processing: {doc_path.name}")
                    
                    # Extract text
                    text = extractor.extract(doc_path)
                    
                    # Clean text
                    cleaned_text = cleaner.clean(text)
                    
                    # Chunk text
                    chunks = chunker.chunk(cleaned_text)
                    
                    # Create metadata for each chunk
                    metadatas = [
                        {
                            'source': str(doc_path),
                            'filename': doc_path.name,
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        }
                        for i in range(len(chunks))
                    ]
                    
                    all_chunks.extend(chunks)
                    all_metadatas.extend(metadatas)
                    
                    logger.info(f"  -> Created {len(chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"Error processing {doc_path.name}: {e}")
                    continue
            
            # Generate embeddings and add to vector store
            if all_chunks:
                logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
                embeddings = embedder.embed(all_chunks)
                
                logger.info("Adding to vector store...")
                vector_store.add(embeddings, all_chunks, all_metadatas)
                
                # Save the index
                vector_store.save()
                logger.info(f"Vector store saved to {self.config.VECTOR_STORE_PATH}")
            
            self.vector_store = vector_store
            logger.info("Document ingestion completed successfully")
            
        except Exception as e:
            logger.error(f"Error during ingestion: {e}", exc_info=True)
            raise
    
    def initialize_rag_pipeline(self):
        """
        Initialize the RAG pipeline with all components.
        """
        logger.info("Initializing RAG pipeline...")
        
        try:
            # Load vector store if not already loaded
            if self.vector_store is None:
                logger.info("Loading vector store...")
                embedder = EmbeddingGenerator(
                    model_name=self.config.EMBEDDING_MODEL
                )
                self.vector_store = FAISSVectorStore(
                    dimension=self.config.EMBEDDING_DIMENSION,
                    index_path=self.config.VECTOR_STORE_PATH
                )
                
                if not self.vector_store.index_exists():
                    logger.error("Vector store not found. Please run ingestion first.")
                    raise FileNotFoundError("Vector store index not found")
                
                self.vector_store.load()
            
            # Initialize query engine
            embedder = EmbeddingGenerator(
                model_name=self.config.EMBEDDING_MODEL
            )
            self.query_engine = QueryEngine(
                vector_store=self.vector_store,
                embedder=embedder,
                top_k=self.config.TOP_K_RESULTS
            )
            
            # Initialize LLM gateway
            self.llm_gateway = GeminiLLMGateway(
                api_key=self.config.GEMINI_API_KEY,
                model_name=self.config.GEMINI_MODEL,
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS
            )
            
            # Initialize prompt template
            prompt_template = MedicalPromptTemplate()
            
            # Create RAG pipeline
            self.rag_pipeline = RAGPipeline(
                query_engine=self.query_engine,
                llm_gateway=self.llm_gateway,
                prompt_template=prompt_template
            )
            
            logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {e}", exc_info=True)
            raise
    
    def ask(self, query: str) -> dict:
        """
        Ask a medical question and get a RAG-based response.
        
        Args:
            query: The medical question to ask
            
        Returns:
            Dictionary containing:
                - answer: The generated answer
                - sources: List of source documents used
                - confidence: Confidence score (if available)
        """
        logger.info(f"Processing query: {query}")
        
        try:
            if self.rag_pipeline is None:
                self.initialize_rag_pipeline()
            
            # Run the RAG pipeline
            result = self.rag_pipeline.query(query)
            
            logger.info("Query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'error': str(e)
            }


def main():
    """
    Main entry point for the Medical RAG Assistant CLI.
    """
    parser = argparse.ArgumentParser(
        description="Medical RAG Assistant - Ask questions based on medical documents"
    )
    
    # Mutually exclusive group for operations
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--query', '-q',
        type=str,
        help='Ask a medical question'
    )
    group.add_argument(
        '--ingest', '-i',
        action='store_true',
        help='Ingest documents from data directory'
    )
    
    # Optional arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/raw',
        help='Directory containing documents to ingest (default: ./data/raw)'
    )
    parser.add_argument(
        '--rebuild-index',
        action='store_true',
        help='Rebuild the vector store index from scratch'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.py',
        help='Path to configuration file (default: config.py)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = Config()
        logger.info("Configuration loaded successfully")
        
        # Initialize assistant
        assistant = MedicalRAGAssistant(config)
        
        # Handle ingestion
        if args.ingest:
            data_dir = Path(args.data_dir)
            if not data_dir.exists():
                logger.error(f"Data directory not found: {data_dir}")
                sys.exit(1)
            
            assistant.ingest_documents(data_dir, rebuild=args.rebuild_index)
            logger.info("Ingestion complete!")
        
        # Handle query
        elif args.query:
            result = assistant.ask(args.query)
            
            print("\n" + "="*80)
            print("QUERY:", args.query)
            print("="*80)
            print("\nANSWER:")
            print(result.get('answer', 'No answer generated'))
            print("\n" + "-"*80)
            print("SOURCES:")
            for i, source in enumerate(result.get('sources', []), 1):
                print(f"\n{i}. {source.get('filename', 'Unknown')}")
                print(f"   Relevance: {source.get('score', 'N/A')}")
            print("="*80 + "\n")
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()