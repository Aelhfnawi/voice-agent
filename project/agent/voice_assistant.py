"""
Voice Assistant implementation using Google Speech APIs.

This module provides:
- Speech-to-Text (Google Cloud Speech)
- Text generation with RAG (Gemini)
- Text-to-Speech (Google Cloud TTS)
"""

import logging
import asyncio
import io
from typing import Optional

try:
    from google.cloud import speech_v1 as speech
    from google.cloud import texttospeech_v1 as texttospeech
    import google.generativeai as genai
    GOOGLE_APIS_AVAILABLE = True
except ImportError:
    GOOGLE_APIS_AVAILABLE = False
    logging.warning("Google Cloud APIs not available")

from agent.rag_helper import RAGHelper
from config import Config

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """
    Voice assistant that processes speech and generates responses.
    
    Uses:
    - Google Speech-to-Text for transcription
    - RAG for context retrieval
    - Gemini for response generation
    - Google Text-to-Speech for audio output
    """
    
    def __init__(self, config: Config, rag_helper: RAGHelper):
        """
        Initialize voice assistant.
        
        Args:
            config: Configuration object
            rag_helper: RAG helper for context retrieval
        """
        self.config = config
        self.rag_helper = rag_helper
        
        # Initialize Gemini for text generation
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        
        logger.info("VoiceAssistant initialized")
    
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[str]:
        """
        Process audio chunk and return transcription.
        
        Args:
            audio_data: Raw audio bytes (PCM format)
            
        Returns:
            Transcribed text or None
        """
        # This would use Google Speech-to-Text API
        # For now, return None as we need proper audio streaming setup
        return None
    
    async def generate_response(self, user_query: str) -> str:
        """
        Generate response using RAG and Gemini.
        
        Args:
            user_query: User's question
            
        Returns:
            Generated response text
        """
        try:
            logger.info(f"Generating response for: {user_query[:100]}")
            
            # Step 1: Retrieve context from RAG
            context = await self.rag_helper.retrieve_context(user_query)
            
            # Step 2: Build prompt with context
            if context and context != "No relevant information found in the knowledge base.":
                prompt = f"""Context from knowledge base:
{context}

User question: {user_query}

Please provide a clear and concise answer based on the context above. If the context doesn't contain enough information, say so."""
            else:
                prompt = f"""User question: {user_query}

Please provide a helpful response. Note: No specific context was found in the knowledge base for this question."""
            
            # Step 3: Generate response with Gemini
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            answer = response.text if hasattr(response, 'text') else str(response)
            
            logger.info(f"Generated response: {len(answer)} characters")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return "I apologize, but I encountered an error processing your question. Please try again."
    
    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to convert
            
        Returns:
            Audio data as bytes
        """
        # This would use Google Text-to-Speech API
        # For now, return empty bytes
        # In production, implement:
        # client = texttospeech.TextToSpeechClient()
        # synthesis_input = texttospeech.SynthesisInput(text=text)
        # voice = texttospeech.VoiceSelectionParams(...)
        # audio_config = texttospeech.AudioConfig(...)
        # response = client.synthesize_speech(...)
        # return response.audio_content
        return b""


if __name__ == "__main__":
    """Test the voice assistant."""
    import sys
    from pathlib import Path
    from dotenv import load_dotenv
    
    sys.path.append(str(Path(__file__).parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    
    print("="*80)
    print("VOICE ASSISTANT TEST")
    print("="*80)
    
    try:
        config = Config()
        rag_helper = RAGHelper(config)
        assistant = VoiceAssistant(config, rag_helper)
        
        # Test text generation
        async def test():
            query = "What is two-factor authentication?"
            print(f"\nQuery: {query}")
            
            response = await assistant.generate_response(query)
            print(f"\nResponse:\n{response}")
        
        asyncio.run(test())
        
        print("\n" + "="*80)
        print("Test completed!")
        print("="*80)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()