"""
RAG Pipeline module for the Technical RAG Assistant.

This module orchestrates the complete RAG workflow:
1. Query processing
2. Document retrieval
3. Context assembly
4. LLM generation
5. Response formatting

Combines all components into a unified interface.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.retriever.query_engine import QueryEngine
from src.llm.llm_gateway import GeminiLLMGateway
from src.llm.prompt_template import MedicalPromptTemplate

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline orchestrating retrieval and generation.
    
    Combines query engine, LLM, and prompt templates into a
    unified question-answering system.
    """
    
    def __init__(
        self,
        query_engine: QueryEngine,
        llm_gateway: GeminiLLMGateway,
        prompt_template: MedicalPromptTemplate,
        max_context_length: int = 4096,
        include_sources: bool = True
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            query_engine: Query engine for retrieval
            llm_gateway: LLM gateway for generation
            prompt_template: Prompt template manager
            max_context_length: Maximum context length in characters
            include_sources: Whether to include source documents in response
        """
        self.query_engine = query_engine
        self.llm_gateway = llm_gateway
        self.prompt_template = prompt_template
        self.max_context_length = max_context_length
        self.include_sources = include_sources
        
        logger.info("RAGPipeline initialized")
    
    def query(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            query: User query string
            conversation_history: Optional conversation history
            
        Returns:
            Dictionary containing:
                - answer: Generated answer
                - sources: List of source documents
                - retrieval_results: Raw retrieval results
                - success: Whether generation succeeded
                - error: Error message (if failed)
        """
        if not query or not query.strip():
            return {
                'answer': 'Please provide a valid question.',
                'sources': [],
                'retrieval_results': [],
                'success': False,
                'error': 'Empty query'
            }
        
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Step 1: Retrieve relevant documents
            logger.info("Step 1: Retrieving documents...")
            retrieval_result = self.query_engine.retrieve_with_context(
                query=query,
                max_context_length=self.max_context_length
            )
            
            context = retrieval_result['context']
            documents = retrieval_result['documents']
            
            if not context:
                logger.warning("No relevant context found")
                return {
                    'answer': "I couldn't find relevant information in the knowledge base to answer your question. Please try rephrasing or ask a different question.",
                    'sources': [],
                    'retrieval_results': [],
                    'success': True,
                    'error': None
                }
            
            logger.info(f"Retrieved {len(documents)} documents, context length: {len(context)}")
            
            # Step 2: Format prompt
            logger.info("Step 2: Formatting prompt...")
            prompt = self.prompt_template.format_rag_prompt(
                query=query,
                context=context,
                conversation_history=conversation_history
            )
            
            system_instruction = self.prompt_template.get_system_instruction()
            
            # Step 3: Generate response
            logger.info("Step 3: Generating response...")
            generation_result = self.llm_gateway.generate(
                prompt=prompt,
                system_instruction=system_instruction
            )
            
            if not generation_result['success']:
                logger.error(f"Generation failed: {generation_result['error']}")
                return {
                    'answer': f"Error generating response: {generation_result['error']}",
                    'sources': self._format_sources(documents) if self.include_sources else [],
                    'retrieval_results': documents,
                    'success': False,
                    'error': generation_result['error']
                }
            
            answer = generation_result['response']
            
            # Step 4: Format response
            logger.info("Step 4: Formatting response...")
            
            # Add disclaimer if enabled
            answer = self.prompt_template.add_disclaimer(answer)
            
            # Format sources
            sources = self._format_sources(documents) if self.include_sources else []
            
            logger.info("Query processing completed successfully")
            
            return {
                'answer': answer,
                'sources': sources,
                'retrieval_results': documents,
                'success': True,
                'error': None,
                'usage': generation_result.get('usage', {})
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}", exc_info=True)
            return {
                'answer': f"An error occurred while processing your question: {str(e)}",
                'sources': [],
                'retrieval_results': [],
                'success': False,
                'error': str(e)
            }
    
    def _format_sources(self, documents: List[Dict]) -> List[Dict]:
        """
        Format source documents for output.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            List of formatted source dictionaries
        """
        sources = []
        
        for i, doc in enumerate(documents, 1):
            source = {
                'id': i,
                'text': doc['text'][:200] + '...' if len(doc['text']) > 200 else doc['text'],
                'score': round(doc['score'], 4),
                'metadata': doc.get('metadata', {})
            }
            
            # Add filename if available
            if 'filename' in doc.get('metadata', {}):
                source['filename'] = doc['metadata']['filename']
            
            sources.append(source)
        
        return sources
    
    def chat(self, messages: List[Dict[str, str]]) -> Dict:
        """
        Have a multi-turn conversation with RAG.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Response dictionary
        """
        if not messages:
            return {
                'answer': 'No messages provided.',
                'sources': [],
                'success': False,
                'error': 'Empty messages'
            }
        
        # Get the last user message
        last_message = messages[-1].get('content', '')
        
        # Get conversation history (all but last)
        history = messages[:-1] if len(messages) > 1 else None
        
        # Process through RAG pipeline
        return self.query(last_message, conversation_history=history)
    
    def extract_information(self, extraction_query: str) -> Dict:
        """
        Extract specific information from documents.
        
        Args:
            extraction_query: What information to extract
            
        Returns:
            Response dictionary with extracted information
        """
        logger.info(f"Extracting information: {extraction_query[:100]}...")
        
        try:
            # Retrieve relevant documents
            retrieval_result = self.query_engine.retrieve_with_context(
                query=extraction_query,
                max_context_length=self.max_context_length
            )
            
            context = retrieval_result['context']
            documents = retrieval_result['documents']
            
            if not context:
                return {
                    'answer': "No relevant information found.",
                    'sources': [],
                    'success': True,
                    'error': None
                }
            
            # Format extraction prompt
            prompt = self.prompt_template.format_extraction_prompt(
                query=extraction_query,
                context=context
            )
            
            system_instruction = self.prompt_template.get_system_instruction()
            
            # Generate response
            generation_result = self.llm_gateway.generate(
                prompt=prompt,
                system_instruction=system_instruction
            )
            
            if not generation_result['success']:
                return {
                    'answer': f"Error: {generation_result['error']}",
                    'sources': [],
                    'success': False,
                    'error': generation_result['error']
                }
            
            return {
                'answer': generation_result['response'],
                'sources': self._format_sources(documents) if self.include_sources else [],
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error in extraction: {e}", exc_info=True)
            return {
                'answer': f"Error: {str(e)}",
                'sources': [],
                'success': False,
                'error': str(e)
            }
    
    def summarize(self, focus: Optional[str] = None, query: Optional[str] = None) -> Dict:
        """
        Summarize documents.
        
        Args:
            focus: Optional focus area for summary
            query: Optional query to retrieve specific documents
            
        Returns:
            Response dictionary with summary
        """
        logger.info("Generating summary...")
        
        try:
            # If query provided, retrieve specific documents
            if query:
                retrieval_result = self.query_engine.retrieve_with_context(
                    query=query,
                    max_context_length=self.max_context_length
                )
            else:
                # Otherwise, could implement retrieving all docs or top docs
                # For now, require a query
                return {
                    'answer': "Please provide a query to specify which documents to summarize.",
                    'sources': [],
                    'success': False,
                    'error': 'No query provided'
                }
            
            context = retrieval_result['context']
            documents = retrieval_result['documents']
            
            if not context:
                return {
                    'answer': "No documents found to summarize.",
                    'sources': [],
                    'success': True,
                    'error': None
                }
            
            # Format summary prompt
            prompt = self.prompt_template.format_summary_prompt(
                context=context,
                focus=focus
            )
            
            system_instruction = self.prompt_template.get_system_instruction()
            
            # Generate response
            generation_result = self.llm_gateway.generate(
                prompt=prompt,
                system_instruction=system_instruction
            )
            
            if not generation_result['success']:
                return {
                    'answer': f"Error: {generation_result['error']}",
                    'sources': [],
                    'success': False,
                    'error': generation_result['error']
                }
            
            return {
                'answer': generation_result['response'],
                'sources': self._format_sources(documents) if self.include_sources else [],
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error in summarization: {e}", exc_info=True)
            return {
                'answer': f"Error: {str(e)}",
                'sources': [],
                'success': False,
                'error': str(e)
            }
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        return {
            'query_engine': self.query_engine.get_stats(),
            'llm': self.llm_gateway.get_model_info(),
            'max_context_length': self.max_context_length,
            'include_sources': self.include_sources
        }


if __name__ == "__main__":
    # Test the RAG pipeline
    import os
    from dotenv import load_dotenv
    from src.ingestion.embedder import EmbeddingGenerator
    from src.ingestion.vector_store import FAISSVectorStore
    
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("RAG PIPELINE TEST")
    print("="*80)
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("\nError: GEMINI_API_KEY not found in environment variables.")
        sys.exit(1)
    
    try:
        # Step 1: Create test documents and vector store
        print("\nStep 1: Setting up test knowledge base...")
        
        test_docs = [
            "Python is a high-level programming language known for its simplicity and readability.",
            "Machine learning is a subset of AI that enables computers to learn from data.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "Deep learning uses multi-layered neural networks for complex pattern recognition.",
            "Natural language processing helps computers understand and generate human language.",
        ]
        
        # Initialize components
        embedder = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")
        embeddings = embedder.embed_documents(test_docs, show_progress=False)
        
        vector_store = FAISSVectorStore(
            dimension=embedder.embedding_dimension,
            index_type="Flat"
        )
        metadatas = [{'doc_id': i, 'source': 'test'} for i in range(len(test_docs))]
        vector_store.add(embeddings, test_docs, metadatas)
        
        # Step 2: Initialize RAG components
        print("\nStep 2: Initializing RAG pipeline...")
        
        query_engine = QueryEngine(
            vector_store=vector_store,
            embedder=embedder,
            top_k=3
        )
        
        llm_gateway = GeminiLLMGateway(
            api_key=api_key,
            model_name="gemini-2.5-flash",
            temperature=0.3
        )
        
        prompt_template = MedicalPromptTemplate(include_citations=True)
        
        rag_pipeline = RAGPipeline(
            query_engine=query_engine,
            llm_gateway=llm_gateway,
            prompt_template=prompt_template
        )
        
        # Step 3: Test queries
        print("\n" + "="*80)
        print("Test 1: Simple Query")
        print("="*80)
        
        result = rag_pipeline.query("What is machine learning?")
        
        print(f"\nQuery: What is machine learning?")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources ({len(result['sources'])}):")
        for source in result['sources']:
            print(f"  - Score: {source['score']}: {source['text']}")
        
        # Test 2: Extraction
        print("\n" + "="*80)
        print("Test 2: Information Extraction")
        print("="*80)
        
        result = rag_pipeline.extract_information(
            "What programming languages are mentioned?"
        )
        
        print(f"\nExtracted Information:\n{result['answer']}")
        
        # Test 3: Summary
        print("\n" + "="*80)
        print("Test 3: Summarization")
        print("="*80)
        
        result = rag_pipeline.summarize(
            query="artificial intelligence and machine learning",
            focus="key concepts"
        )
        
        print(f"\nSummary:\n{result['answer']}")
        
        # Display stats
        print("\n" + "="*80)
        print("Pipeline Statistics")
        print("="*80)
        stats = rag_pipeline.get_stats()
        print(f"Vector Store: {stats['query_engine']['vector_store']['num_vectors']} documents")
        print(f"LLM Model: {stats['llm']['model_name']}")
        print(f"Max Context: {stats['max_context_length']} characters")
        
        print("\n" + "="*80)
        print("All tests completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)