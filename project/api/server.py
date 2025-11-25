"""
FastAPI server for the Technical RAG Assistant.

This module provides REST API endpoints for:
- Querying the RAG system
- Document ingestion
- Health checks
- System statistics

Run with: uvicorn api.server:app --reload
"""

import logging
import sys
import os
import time
from pathlib import Path
from typing import List, Optional, Dict
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# LiveKit imports
try:
    from livekit import api
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    logger.warning("livekit-api not installed. Install with: pip install livekit-api")

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Technical RAG Assistant API",
    description="REST API for document-based question answering using RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for components
config: Optional[Config] = None
rag_pipeline: Optional[RAGPipeline] = None
query_engine: Optional[QueryEngine] = None
embedder: Optional[EmbeddingGenerator] = None
vector_store: Optional[FAISSVectorStore] = None


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""
    query: str = Field(..., description="The question to ask", min_length=1, max_length=1000)
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    include_sources: Optional[bool] = Field(True, description="Include source documents in response")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict] = Field(default_factory=list, description="Source documents")
    success: bool = Field(..., description="Whether the query was successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    components: Dict[str, bool] = Field(..., description="Component health status")


class StatsResponse(BaseModel):
    """Response model for system statistics."""
    num_documents: int = Field(..., description="Number of documents in index")
    vector_store_type: str = Field(..., description="Type of vector store")
    embedding_model: str = Field(..., description="Embedding model name")
    llm_model: str = Field(..., description="LLM model name")


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool = Field(..., description="Whether ingestion was successful")
    message: str = Field(..., description="Status message")
    num_documents: int = Field(0, description="Number of documents ingested")
    num_chunks: int = Field(0, description="Number of chunks created")


class ExtractionRequest(BaseModel):
    """Request model for information extraction."""
    extraction_query: str = Field(..., description="What to extract", min_length=1)


class SummaryRequest(BaseModel):
    """Request model for summarization."""
    query: str = Field(..., description="Query to retrieve documents for summarization")
    focus: Optional[str] = Field(None, description="Focus area for the summary")


class LiveKitTokenRequest(BaseModel):
    """Request model for LiveKit token generation."""
    room_name: str = Field(..., description="Name of the LiveKit room")
    participant_name: str = Field(..., description="Name of the participant")
    metadata: Optional[str] = Field(None, description="Optional participant metadata")


class LiveKitTokenResponse(BaseModel):
    """Response model for LiveKit token generation."""
    token: str = Field(..., description="JWT token for LiveKit")
    url: str = Field(..., description="LiveKit server URL")
    room_name: str = Field(..., description="Room name")


class CreateRoomRequest(BaseModel):
    """Request model for creating a LiveKit room."""
    room_name: Optional[str] = Field(None, description="Custom room name (auto-generated if not provided)")


class CreateRoomResponse(BaseModel):
    """Response model for room creation."""
    room_name: str = Field(..., description="Name of the created room")
    url: str = Field(..., description="LiveKit server URL")


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global config, rag_pipeline, query_engine, embedder, vector_store
    
    logger.info("Starting Technical RAG Assistant API...")
    
    try:
        # Load configuration
        config = Config()
        logger.info("Configuration loaded")
        
        # Initialize embedder
        embedder = EmbeddingGenerator(
            model_name=config.EMBEDDING_MODEL
        )
        logger.info("Embedder initialized")
        
        # Initialize vector store
        vector_store = FAISSVectorStore(
            dimension=config.EMBEDDING_DIMENSION,
            index_path=config.VECTOR_STORE_PATH,
            index_type=config.FAISS_INDEX_TYPE
        )
        
        # Load existing index if available
        if vector_store.index_exists():
            vector_store.load()
            logger.info(f"Loaded existing vector store with {vector_store.index.ntotal} vectors")
        else:
            logger.warning("No existing vector store found. Please ingest documents first.")
        
        # Initialize query engine
        query_engine = QueryEngine(
            vector_store=vector_store,
            embedder=embedder,
            top_k=config.TOP_K_RESULTS,
            score_threshold=config.SIMILARITY_THRESHOLD
        )
        logger.info("Query engine initialized")
        
        # Initialize LLM gateway
        llm_gateway = GeminiLLMGateway(
            api_key=config.GEMINI_API_KEY,
            model_name=config.GEMINI_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
        logger.info("LLM gateway initialized")
        
        # Initialize prompt template
        prompt_template = MedicalPromptTemplate(
            include_citations=config.INCLUDE_CITATIONS,
            citation_format=config.CITATION_FORMAT
        )
        logger.info("Prompt template initialized")
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            query_engine=query_engine,
            llm_gateway=llm_gateway,
            prompt_template=prompt_template,
            max_context_length=config.MAX_CONTEXT_LENGTH
        )
        logger.info("RAG pipeline initialized")
        
        logger.info("API startup complete!")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Technical RAG Assistant API...")


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Technical RAG Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the health status of all components.
    """
    components = {
        "rag_pipeline": rag_pipeline is not None,
        "query_engine": query_engine is not None,
        "vector_store": vector_store is not None and vector_store.index.ntotal > 0,
        "embedder": embedder is not None,
    }
    
    all_healthy = all(components.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        components=components
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(request: QueryRequest):
    """
    Query the RAG system.
    
    Processes a question and returns an answer with sources.
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    if vector_store is None or vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents in knowledge base. Please ingest documents first."
        )
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Update query engine parameters if provided
        if request.top_k:
            query_engine.update_parameters(top_k=request.top_k)
        
        # Process query
        result = rag_pipeline.query(request.query)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Filter sources if not requested
        sources = result['sources'] if request.include_sources else []
        
        return QueryResponse(
            answer=result['answer'],
            sources=sources,
            success=result['success'],
            error=result.get('error'),
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract", response_model=QueryResponse, tags=["Query"])
async def extract_information(request: ExtractionRequest):
    """
    Extract specific information from documents.
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        result = rag_pipeline.extract_information(request.extraction_query)
        
        return QueryResponse(
            answer=result['answer'],
            sources=result.get('sources', []),
            success=result['success'],
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"Error extracting information: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize", response_model=QueryResponse, tags=["Query"])
async def summarize_documents(request: SummaryRequest):
    """
    Summarize documents.
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        result = rag_pipeline.summarize(
            query=request.query,
            focus=request.focus
        )
        
        return QueryResponse(
            answer=result['answer'],
            sources=result.get('sources', []),
            success=result['success'],
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"Error summarizing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """
    Get system statistics.
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        stats = rag_pipeline.get_stats()
        
        return StatsResponse(
            num_documents=stats['query_engine']['vector_store']['num_vectors'],
            vector_store_type=stats['query_engine']['vector_store']['index_type'],
            embedding_model=stats['query_engine']['embedder']['model_name'],
            llm_model=stats['llm']['model_name']
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Ingest a single document file.
    
    Accepts PDF, DOCX, or TXT files.
    """
    if vector_store is None or embedder is None:
        raise HTTPException(status_code=503, detail="Components not initialized")
    
    # Check file type
    allowed_extensions = ['.pdf', '.docx', '.txt']
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file temporarily
        temp_path = Path("data/raw") / file.filename
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process in background
        background_tasks.add_task(process_document, temp_path)
        
        return IngestResponse(
            success=True,
            message=f"Document '{file.filename}' queued for processing",
            num_documents=1,
            num_chunks=0  # Will be updated after processing
        )
        
    except Exception as e:
        logger.error(f"Error ingesting file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== LIVEKIT ENDPOINTS ====================

@app.post("/livekit/token", response_model=LiveKitTokenResponse, tags=["LiveKit"])
async def generate_livekit_token(request: LiveKitTokenRequest):
    """
    Generate a LiveKit access token for a participant.
    
    This token allows a user or agent to join a LiveKit room.
    """
    if not LIVEKIT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="LiveKit not available. Install with: pip install livekit-api"
        )
    
    # Get LiveKit credentials from config
    livekit_url = os.getenv('LIVEKIT_URL')
    livekit_api_key = os.getenv('LIVEKIT_API_KEY')
    livekit_api_secret = os.getenv('LIVEKIT_API_SECRET')
    
    if not all([livekit_url, livekit_api_key, livekit_api_secret]):
        raise HTTPException(
            status_code=500,
            detail="LiveKit credentials not configured. Check LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET in .env"
        )
    
    try:
        # Create access token
        token = api.AccessToken(livekit_api_key, livekit_api_secret)
        token.with_identity(request.participant_name)
        token.with_name(request.participant_name)
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=request.room_name,
        ))
        
        if request.metadata:
            token.with_metadata(request.metadata)
        
        jwt_token = token.to_jwt()
        
        logger.info(f"Generated token for {request.participant_name} in room {request.room_name}")
        
        return LiveKitTokenResponse(
            token=jwt_token,
            url=livekit_url,
            room_name=request.room_name
        )
        
    except Exception as e:
        logger.error(f"Error generating LiveKit token: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/livekit/room", response_model=CreateRoomResponse, tags=["LiveKit"])
async def create_livekit_room(request: CreateRoomRequest):
    """
    Create a new LiveKit room.
    
    If room_name is not provided, generates a unique room name.
    """
    if not LIVEKIT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="LiveKit not available. Install with: pip install livekit-api"
        )
    
    livekit_url = os.getenv('LIVEKIT_URL')
    livekit_api_key = os.getenv('LIVEKIT_API_KEY')
    livekit_api_secret = os.getenv('LIVEKIT_API_SECRET')
    
    if not all([livekit_url, livekit_api_key, livekit_api_secret]):
        raise HTTPException(
            status_code=500,
            detail="LiveKit credentials not configured"
        )
    
    try:
        # Generate room name if not provided
        room_name = request.room_name or f"room-{int(time.time())}"
        
        # Create room using LiveKit API
        room_service = api.RoomService()
        room_service.url = livekit_url
        room_service.api_key = livekit_api_key
        room_service.api_secret = livekit_api_secret
        
        # Create or get room
        await room_service.create_room(api.CreateRoomRequest(name=room_name))
        
        logger.info(f"Created/Retrieved LiveKit room: {room_name}")
        
        return CreateRoomResponse(
            room_name=room_name,
            url=livekit_url
        )
        
    except Exception as e:
        logger.error(f"Error creating room: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/livekit/rooms", tags=["LiveKit"])
async def list_livekit_rooms():
    """
    List all active LiveKit rooms.
    """
    if not LIVEKIT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="LiveKit not available"
        )
    
    livekit_url = os.getenv('LIVEKIT_URL')
    livekit_api_key = os.getenv('LIVEKIT_API_KEY')
    livekit_api_secret = os.getenv('LIVEKIT_API_SECRET')
    
    if not all([livekit_url, livekit_api_key, livekit_api_secret]):
        raise HTTPException(
            status_code=500,
            detail="LiveKit credentials not configured"
        )
    
    try:
        room_service = api.RoomService()
        room_service.url = livekit_url
        room_service.api_key = livekit_api_key
        room_service.api_secret = livekit_api_secret
        
        rooms = await room_service.list_rooms(api.ListRoomsRequest())
        
        return {
            "rooms": [
                {
                    "name": room.name,
                    "num_participants": room.num_participants,
                    "creation_time": room.creation_time
                }
                for room in rooms.rooms
            ]
        }
        
    except Exception as e:
        logger.error(f"Error listing rooms: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def process_document(file_path: Path):
    """Background task to process uploaded document."""
    try:
        logger.info(f"Processing document: {file_path.name}")
        
        # Extract
        extractor = DocumentExtractor()
        text = extractor.extract(file_path)
        
        # Clean
        cleaner = TextCleaner()
        cleaned_text = cleaner.clean(text)
        
        # Chunk
        chunker = TextChunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        chunks = chunker.chunk(cleaned_text)
        
        # Embed
        embeddings = embedder.embed(chunks)
        
        # Add to vector store
        metadatas = [
            {'filename': file_path.name, 'chunk_index': i}
            for i in range(len(chunks))
        ]
        vector_store.add(embeddings, chunks, metadatas)
        vector_store.save()
        
        logger.info(f"Successfully processed {file_path.name}: {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}", exc_info=True)


# Import for time module
import time


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )