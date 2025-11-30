"""
Optimized Voice Assistant with smart response handling.
"""

import logging
import asyncio
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent))

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GOOGLE_APIS_AVAILABLE = True
except ImportError:
    GOOGLE_APIS_AVAILABLE = False
    logging.warning("Google APIs not available")

from agent.rag_helper import RAGHelper
from config import Config

logger = logging.getLogger(__name__)

# Quick responses for common patterns
QUICK_RESPONSES = {
    'hi': "Hello! I'm your AI assistant. How can I help you?",
    'hello': "Hi there! What can I help you with?",
    'hey': "Hey! What would you like to know?",
    'thanks': "You're welcome!",
    'thank you': "Happy to help!",
    'bye': "Goodbye!",
}


class VoiceAssistant:
    """Optimized voice assistant."""
    
    def __init__(self, config: Config, rag_helper: RAGHelper):
        self.config = config
        self.rag_helper = rag_helper
        
        genai.configure(api_key=config.GEMINI_API_KEY)
        
        # Configure safety settings to allow more content
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        self.model = genai.GenerativeModel(
            config.GEMINI_MODEL,
            safety_settings=safety_settings,
            generation_config={
                'temperature': 0.3,
                'max_output_tokens': 512,
                'top_p': 0.9
            }
        )
        
        logger.info("VoiceAssistant initialized")
    
    def _get_quick_response(self, query: str) -> Optional[str]:
        """Check for quick response patterns."""
        normalized = query.lower().strip()
        
        if normalized in QUICK_RESPONSES:
            return QUICK_RESPONSES[normalized]
        
        if 'thank' in normalized:
            return QUICK_RESPONSES['thanks']
        if 'bye' in normalized:
            return QUICK_RESPONSES['bye']
        
        return None
    
    async def generate_response(self, user_query: str) -> str:
        """Generate response with optimization."""
        try:
            quick_response = self._get_quick_response(user_query)
            if quick_response:
                logger.info(f"Quick response for: {user_query[:50]}")
                return quick_response
            
            logger.info(f"Generating response for: {user_query[:50]}")
            
            context = await self.rag_helper.retrieve_context(
                user_query,
                max_context_length=800
            )
            
            if context and "No relevant information" not in context:
                prompt = f"""Context: {context}

Question: {user_query}

Answer briefly (2-3 sentences max)."""
            else:
                prompt = f"""Question: {user_query}

Answer briefly (2-3 sentences)."""
            
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.model.generate_content,
                    prompt
                ),
                timeout=15.0
            )
            
            try:
                answer = response.text
            except ValueError:
                # Handle cases where response is blocked or empty
                logger.warning(f"Response content generation failed. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
                logger.warning(f"Safety ratings: {response.candidates[0].safety_ratings if response.candidates else 'Unknown'}")
                logger.warning(f"Prompt feedback: {response.prompt_feedback}")
                return "I apologize, but I am unable to generate a response to that specific query."
            
            logger.info(f"Generated response: {len(answer)} chars")
            return answer
            
        except asyncio.TimeoutError:
            return "Response timed out. Please try again."
        except Exception as e:
            logger.error(f"Error: {e}")
            return "I encountered an error. Please rephrase your question."
    
    async def text_to_speech(self, text: str) -> bytes:
        return b""