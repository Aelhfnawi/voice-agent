# Technical RAG Assistant with Voice Agent

A production-ready Retrieval-Augmented Generation (RAG) system with LiveKit voice integration for technical documentation and knowledge base question answering. Built with FAISS vector store, BAAI embeddings, Google Gemini API, and LiveKit for real-time voice interaction.

## 🎯 Features

- **Multi-format Document Processing**: PDF, DOCX, and TXT files
- **Intelligent Text Chunking**: Semantic chunking with configurable overlap
- **Advanced Embeddings**: BAAI/bge-large-en-v1.5 (1024-dimensional)
- **Fast Vector Search**: FAISS-based similarity search
- **LLM Integration**: Google Gemini 2.5 API
- **Voice Agent**: LiveKit + Gemini Live API for real-time voice interaction
- **Source Citations**: Automatic citation of source documents
- **REST API**: FastAPI backend with LiveKit integration
- **Web UI**: React frontend with LiveKit SDK
- **CLI Interface**: Easy command-line usage
- **Modular Architecture**: Clean, maintainable codebase

## 📁 Project Structure

```
project/
├── data/
│   ├── raw/                     # Original documents (PDF, DOCX, TXT)
│   ├── processed/               # Cleaned and processed data
│   └── vector_store/            # FAISS index files
│
├── src/
│   ├── ingestion/
│   │   ├── extractor.py         # Document extraction
│   │   ├── cleaner.py           # Text cleaning
│   │   ├── chunker.py           # Text chunking
│   │   ├── embedder.py          # Embedding generation
│   │   └── vector_store.py      # FAISS vector store
│   │
│   ├── retriever/
│   │   └── query_engine.py      # Retrieval logic
│   │
│   ├── llm/
│   │   ├── llm_gateway.py       # Gemini API integration
│   │   └── prompt_template.py   # Prompt templates
│   │
│   └── pipeline/
│       └── rag_pipeline.py      # Complete RAG pipeline
│
├── agent/
│   ├── livekit_agent.py         # LiveKit voice agent
│   ├── rag_helper.py            # RAG integration for agent
│   └── __init__.py
│
├── api/
│   └── server.py                # FastAPI backend with LiveKit
│
├── frontend/                     # React web UI
│   ├── src/
│   │   ├── App.js              # Main React component
│   │   ├── App.css             # Styles
│   │   └── index.js
│   ├── public/
│   └── package.json
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retriever.py
│   └── test_rag_pipeline.py
│
├── main.py                      # CLI entry point
├── config.py                    # Configuration
├── requirements.txt             # Python dependencies
└── .env                         # Environment variables
```

## 🚀 Quick Start Guide

### Complete Setup and Run

```bash
# 1. Setup Python environment
cd project
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Ingest documents
python main.py --ingest --data-dir ./data/raw

# 5. Test CLI query
python main.py --query "What is two-factor authentication?"

# 6. Start API (Terminal 1)
uvicorn api.server:app --reload

# 7. Start Voice Agent (Terminal 2)
python agent/livekit_agent.py dev

# 8. Setup and start Frontend (Terminal 3)
cd ../frontend
npm install
npm start

# 9. Open browser at http://localhost:3000
```

### Quick Test
```bash
# Single command to test the RAG system
python main.py --query "your question here"
```

```env
# Required - Gemini API
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Required - LiveKit Credentials
LIVEKIT_URL=wss://your-livekit-server.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# Optional (defaults shown)
TEMPERATURE=0.3
MAX_TOKENS=2048

EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DIMENSION=1024

CHUNK_SIZE=512
CHUNK_OVERLAP=50

TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.5

FAISS_INDEX_TYPE=Flat
```

### 4. Ingest Documents

Place your documents in `data/raw/` folder, then run:

```bash
python main.py --ingest --data-dir ./data/raw
```

This will:
- Extract text from all PDF, DOCX, and TXT files
- Clean and chunk the text
- Generate embeddings
- Build FAISS vector index

### 5. Ask Questions

```bash
python main.py --query "What is two-factor authentication?"
```

## 📖 Usage Examples

### Basic Query
```bash
python main.py --query "How do I set up 2FA?"
```

### Rebuild Index
```bash
python main.py --ingest --rebuild-index
```

### Custom Data Directory
```bash
python main.py --ingest --data-dir /path/to/documents
```

## 🔧 Configuration Options

### Embedding Models
- `BAAI/bge-large-en-v1.5` (default) - 1024 dimensions, optimized for retrieval
- `sentence-transformers/all-MiniLM-L6-v2` - Lightweight alternative
- `text-embedding-3-small` - OpenAI embeddings (requires API key)

### FAISS Index Types
- `Flat` (default) - Exact search, best for <100k vectors
- `IVFFlat` - Approximate search, faster for large datasets
- `HNSW` - Graph-based, good balance of speed/accuracy

### LLM Models
- `gemini-2.5-flash` - Fast and cost-effective
- `gemini-2.5-pro` - Higher quality responses
- `gemini-2.0-flash-exp` - Experimental features

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suites
```bash
# Test ingestion pipeline
pytest tests/test_ingestion.py -v

# Test retriever
pytest tests/test_retriever.py -v

# Test RAG pipeline
pytest tests/test_rag_pipeline.py -v
```

### Test Individual Components
```bash
# Test document extraction
python src/ingestion/extractor.py path/to/document.pdf

# Test text cleaning
python src/ingestion/cleaner.py

# Test chunking
python src/ingestion/chunker.py

# Test embeddings
python src/ingestion/embedder.py

# Test vector store
python src/ingestion/vector_store.py

# Test query engine
python -m src.retriever.query_engine

# Test RAG pipeline
python -m src.pipeline.rag_pipeline

# Test RAG helper (for voice agent)
python agent/rag_helper.py
```

### Test Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

Test Results: **73 tests passing** ✅

## 🏗️ Architecture

### Ingestion Pipeline
1. **Extraction**: Multi-format document parsing
2. **Cleaning**: Text normalization and cleaning
3. **Chunking**: Semantic text splitting with overlap
4. **Embedding**: Generate dense vector representations
5. **Indexing**: Store in FAISS vector database

### Query Pipeline
1. **Query Embedding**: Convert query to vector
2. **Retrieval**: Find similar documents via FAISS
3. **Context Assembly**: Format retrieved documents
4. **Generation**: Generate answer using Gemini
5. **Response**: Return answer with sources

## 📊 Performance

- **Embedding Speed**: ~50-100 docs/second (CPU)
- **Search Latency**: <100ms for 10k vectors
- **End-to-end Query**: 2-5 seconds
- **Supported Document Formats**: PDF, DOCX, TXT
- **Maximum Document Size**: 50MB per file

## 🔐 Security

- API keys stored in `.env` (never commit to git)
- Add `.env` to `.gitignore`
- Validate and sanitize all inputs
- Rate limiting recommended for production

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure you're in the project directory
cd project

# Create __init__.py files
echo. > src\__init__.py
echo. > src\ingestion\__init__.py
echo. > src\retriever\__init__.py
echo. > src\llm\__init__.py
echo. > src\pipeline\__init__.py
```

**Model Not Found**
```bash
# List available Gemini models
python -c "import google.generativeai as genai; import os; from dotenv import load_dotenv; load_dotenv(); genai.configure(api_key=os.getenv('GEMINI_API_KEY')); [print(m.name) for m in genai.list_models()]"
```

**GPU Not Detected**
```bash
# Install GPU version of FAISS
pip uninstall faiss-cpu
pip install faiss-gpu

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## 📝 API Usage (Optional)

Start the FastAPI server:
```bash
uvicorn api.server:app --reload
```

API endpoints:
- `POST /query` - Ask a question
- `POST /ingest` - Ingest new documents
- `GET /health` - Health check

## 🛠️ Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to all functions
- Keep functions small and focused

### Adding New Features
1. Create feature branch
2. Implement with tests
3. Update documentation
4. Submit pull request

## 📚 Resources

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [Google Gemini API](https://ai.google.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- BAAI for the BGE embedding models
- Facebook Research for FAISS
- Google for Gemini API
- LiveKit for real-time voice infrastructure
- HuggingFace for Sentence Transformers
- **Claude AI for assistance in development**

---

**Need help?** Open an issue or contact the maintainers.