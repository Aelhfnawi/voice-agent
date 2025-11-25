"""
LLM Gateway module for the Technical RAG Assistant.

This module handles integration with Google Gemini API:
- API authentication and connection
- Prompt formatting and sending
- Response parsing
- Error handling and retries
- Token usage tracking

Supports Gemini 1.5 Pro and other Gemini models.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logging.warning("google-generativeai not installed. Install with: pip install google-generativeai")

logger = logging.getLogger(__name__)


class GeminiLLMGateway:
    """
    Gateway for interacting with Google Gemini API.
    
    Handles all LLM operations including prompt generation,
    API calls, and response parsing.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-pro",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 40
    ):
        """
        Initialize the Gemini LLM Gateway.
        
        Args:
            api_key: Google API key for Gemini
            model_name: Name of the Gemini model to use
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai is required for Gemini integration. "
                "Install it with: pip install google-generativeai"
            )
        
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        
        # Configure API
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.generation_config = genai.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_tokens
        )
        
        try:
            self.model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=self.generation_config
            )
            logger.info(f"GeminiLLMGateway initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        retry_count: int = 3,
        retry_delay: float = 1.0
    ) -> Dict:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt/query
            system_instruction: Optional system instruction
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            Dictionary containing:
                - response: Generated text
                - success: Whether generation succeeded
                - error: Error message (if failed)
                - usage: Token usage information (if available)
        """
        if not prompt or not prompt.strip():
            return {
                'response': '',
                'success': False,
                'error': 'Empty prompt provided'
            }
        
        logger.info(f"Generating response for prompt (length: {len(prompt)})")
        
        for attempt in range(retry_count):
            try:
                # Create model with system instruction if provided
                if system_instruction:
                    model = genai.GenerativeModel(
                        model_name=self.model_name,
                        generation_config=self.generation_config,
                        system_instruction=system_instruction
                    )
                else:
                    model = self.model
                
                # Generate response
                response = model.generate_content(prompt)
                
                # Extract text
                if response.parts:
                    response_text = response.text
                else:
                    response_text = ""
                
                logger.info(f"Generated response (length: {len(response_text)})")
                
                # Try to get usage metadata (may not be available)
                usage_info = {}
                if hasattr(response, 'usage_metadata'):
                    usage_info = {
                        'prompt_tokens': getattr(response.usage_metadata, 'prompt_token_count', None),
                        'response_tokens': getattr(response.usage_metadata, 'candidates_token_count', None),
                        'total_tokens': getattr(response.usage_metadata, 'total_token_count', None)
                    }
                
                return {
                    'response': response_text,
                    'success': True,
                    'error': None,
                    'usage': usage_info
                }
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{retry_count} failed: {e}")
                
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All attempts failed: {e}")
                    return {
                        'response': '',
                        'success': False,
                        'error': str(e)
                    }
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_instruction: Optional[str] = None
    ) -> Dict:
        """
        Generate response with provided context (for RAG).
        
        Args:
            query: User query
            context: Retrieved context from documents
            system_instruction: Optional system instruction
            
        Returns:
            Response dictionary
        """
        # Combine query and context
        prompt = f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        return self.generate(prompt, system_instruction=system_instruction)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None
    ) -> Dict:
        """
        Have a multi-turn conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     Role can be 'user' or 'model'
            system_instruction: Optional system instruction
            
        Returns:
            Response dictionary
        """
        try:
            # Create chat session
            if system_instruction:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=self.generation_config,
                    system_instruction=system_instruction
                )
            else:
                model = self.model
            
            chat = model.start_chat(history=[])
            
            # Add conversation history (all but last message)
            for msg in messages[:-1]:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                # Gemini uses 'user' and 'model' roles
                if role == 'assistant':
                    role = 'model'
                
                chat.history.append({
                    'role': role,
                    'parts': [content]
                })
            
            # Send last message
            last_message = messages[-1].get('content', '')
            response = chat.send_message(last_message)
            
            return {
                'response': response.text,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {
                'response': '',
                'success': False,
                'error': str(e)
            }
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        try:
            result = self.model.count_tokens(text)
            return result.total_tokens
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Rough estimate: ~4 characters per token
            return len(text) // 4
    
    def test_connection(self) -> bool:
        """
        Test if the API connection is working.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            result = self.generate("Hello", retry_count=1)
            return result['success']
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'top_k': self.top_k
        }


if __name__ == "__main__":
    # Test the LLM gateway
    import os
    from dotenv import load_dotenv
    
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("GEMINI LLM GATEWAY TEST")
    print("="*80)
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("\nError: GEMINI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        sys.exit(1)
    
    try:
        # Initialize gateway
        print("\nInitializing Gemini LLM Gateway...")
        gateway = GeminiLLMGateway(
            api_key=api_key,
            model_name="gemini-1.5-pro",
            temperature=0.3,
            max_tokens=1024
        )
        
        # Display model info
        print("\nModel Information:")
        print("-"*80)
        info = gateway.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Test connection
        print("\n" + "="*80)
        print("Testing Connection...")
        print("="*80)
        
        if gateway.test_connection():
            print("✓ Connection successful!")
        else:
            print("✗ Connection failed!")
            sys.exit(1)
        
        # Test simple generation
        print("\n" + "="*80)
        print("Test 1: Simple Generation")
        print("="*80)
        
        prompt = "What is machine learning? Answer in 2 sentences."
        print(f"\nPrompt: {prompt}")
        
        result = gateway.generate(prompt)
        
        if result['success']:
            print(f"\nResponse:\n{result['response']}")
            if result.get('usage'):
                print(f"\nToken Usage: {result['usage']}")
        else:
            print(f"\nError: {result['error']}")
        
        # Test generation with context (RAG simulation)
        print("\n" + "="*80)
        print("Test 2: Generation with Context (RAG)")
        print("="*80)
        
        context = """
        Machine learning is a subset of artificial intelligence that focuses on building 
        systems that can learn from data. These systems improve their performance over time 
        without being explicitly programmed for every scenario.
        """
        
        query = "What is machine learning?"
        print(f"\nQuery: {query}")
        print(f"Context: {context[:100]}...")
        
        result = gateway.generate_with_context(
            query=query,
            context=context,
            system_instruction="You are a helpful technical assistant. Provide clear, concise answers."
        )
        
        if result['success']:
            print(f"\nResponse:\n{result['response']}")
        else:
            print(f"\nError: {result['error']}")
        
        # Test chat
        print("\n" + "="*80)
        print("Test 3: Multi-turn Chat")
        print("="*80)
        
        messages = [
            {'role': 'user', 'content': 'What is Python?'},
            {'role': 'model', 'content': 'Python is a high-level programming language.'},
            {'role': 'user', 'content': 'What is it used for?'}
        ]
        
        print("\nConversation:")
        for msg in messages:
            print(f"{msg['role']}: {msg['content']}")
        
        result = gateway.chat(messages)
        
        if result['success']:
            print(f"\nmodel: {result['response']}")
        else:
            print(f"\nError: {result['error']}")
        
        # Test token counting
        print("\n" + "="*80)
        print("Test 4: Token Counting")
        print("="*80)
        
        test_text = "This is a test sentence for token counting."
        token_count = gateway.count_tokens(test_text)
        print(f"\nText: {test_text}")
        print(f"Estimated tokens: {token_count}")
        
        print("\n" + "="*80)
        print("All tests completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        