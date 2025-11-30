"""
Gemini Live API Client for WebSocket-based audio streaming.

This client handles:
- WebSocket connection to Gemini Live API
- Bidirectional audio streaming (send/receive)
- Session management and reconnection
- RAG context injection
- Audio format conversion
"""

import asyncio
import logging
import json
import base64
from typing import Optional, Callable, Dict, Any
from google.genai import Client
from google.genai import types
import io

logger = logging.getLogger(__name__)


class GeminiLiveClient:
    """
    WebSocket client for Gemini Live API with audio streaming support.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-native-audio-preview-09-2025",
        system_instruction: Optional[str] = None,
        on_audio_received: Optional[Callable[[bytes], None]] = None,
        on_text_received: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None
    ):
        """
        Initialize the Gemini Live API client.
        
        Args:
            api_key: Gemini API key
            model: Model name for Live API
            system_instruction: System instruction for the model
            on_audio_received: Callback for received audio data
            on_text_received: Callback for received text
            on_error: Callback for errors
        """
        self.api_key = api_key
        self.model = model
        self.system_instruction = system_instruction or self._default_system_instruction()
        self.on_audio_received = on_audio_received
        self.on_text_received = on_text_received
        self.on_error = on_error
        
        self.client = Client(api_key=api_key)
        self.session = None
        self.is_connected = False
        self._receive_task = None
        
        logger.info(f"GeminiLiveClient initialized with model: {model}")
    
    def _default_system_instruction(self) -> str:
        """Default system instruction for the voice assistant."""
        return """You are a helpful voice assistant with access to a knowledge base.

When answering questions:
1. Use the provided context from the knowledge base when available
2. Be conversational and natural in your speech
3. Keep responses concise for voice interaction (2-3 sentences max)
4. If you don't have enough information, ask clarifying questions
5. Cite sources briefly when using information from the knowledge base

You have access to a RAG system that retrieves relevant information.
Use it whenever the user asks a question that might be in the knowledge base."""
    
    async def connect(self) -> bool:
        """
        Establish connection to Gemini Live API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Connecting to Gemini Live API...")
            
            config = {
                "response_modalities": ["AUDIO"],
                "system_instruction": self.system_instruction,
            }
            
            # Connect to Live API using context manager
            session_context = self.client.aio.live.connect(
                model=self.model,
                config=config
            )
            
            # Enter the context manager
            self.session = await session_context.__aenter__()
            self._session_context = session_context  # Store for cleanup
            
            self.is_connected = True
            logger.info("Successfully connected to Gemini Live API")
            
            # Start receiving responses
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Gemini Live API: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def disconnect(self):
        """Disconnect from Gemini Live API."""
        try:
            self.is_connected = False
            
            if self._receive_task:
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass
            
            if self.session and hasattr(self, '_session_context'):
                await self._session_context.__aexit__(None, None, None)
                self.session = None
            
            logger.info("Disconnected from Gemini Live API")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def send_audio(self, audio_data: bytes, sample_rate: int = 16000):
        """
        Send audio data to Gemini Live API.
        
        Args:
            audio_data: Raw PCM audio data (16-bit, mono)
            sample_rate: Sample rate of the audio (default: 16kHz)
        """
        if not self.is_connected or not self.session:
            logger.warning("Cannot send audio: not connected")
            return
        
        try:
            # Send audio as realtime input
            await self.session.send_realtime_input(
                audio=types.Blob(
                    data=audio_data,
                    mime_type=f"audio/pcm;rate={sample_rate}"
                )
            )
            
            logger.debug(f"Sent {len(audio_data)} bytes of audio")
            
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
            if self.on_error:
                self.on_error(e)
    
    async def send_text(self, text: str):
        """
        Send text input to Gemini Live API.
        
        Args:
            text: Text message to send
        """
        if not self.is_connected or not self.session:
            logger.warning("Cannot send text: not connected")
            return
        
        try:
            await self.session.send_realtime_input(
                text=text
            )
            
            logger.debug(f"Sent text: {text}")
            
        except Exception as e:
            logger.error(f"Error sending text: {e}")
            if self.on_error:
                self.on_error(e)
    
    async def inject_context(self, context: str):
        """
        Inject RAG context into the conversation.
        
        Args:
            context: Retrieved context from RAG system
        """
        if not context:
            return
        
        # Format context for injection
        context_message = f"\n[Knowledge Base Context]\n{context}\n[End Context]\n"
        
        try:
            # Send context as a system message
            await self.send_text(context_message)
            logger.info(f"Injected RAG context: {len(context)} chars")
            
        except Exception as e:
            logger.error(f"Error injecting context: {e}")
    
    async def _receive_loop(self):
        """Background task to receive responses from Gemini."""
        try:
            logger.info("Starting receive loop...")
            
            async for response in self.session.receive():
                if not self.is_connected:
                    break
                
                # Handle audio data
                if response.data is not None:
                    audio_data = response.data
                    logger.debug(f"Received {len(audio_data)} bytes of audio")
                    
                    if self.on_audio_received:
                        self.on_audio_received(audio_data)
                
                # Handle text data
                if hasattr(response, 'text') and response.text:
                    logger.debug(f"Received text: {response.text}")
                    
                    if self.on_text_received:
                        self.on_text_received(response.text)
                
                # Handle server content (for debugging)
                if hasattr(response, 'server_content') and response.server_content:
                    if hasattr(response.server_content, 'model_turn'):
                        model_turn = response.server_content.model_turn
                        if model_turn:
                            logger.debug(f"Model turn: {model_turn}")
            
            logger.info("Receive loop ended")
            
        except asyncio.CancelledError:
            logger.info("Receive loop cancelled")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
            if self.on_error:
                self.on_error(e)
    
    def update_system_instruction(self, instruction: str):
        """
        Update the system instruction (requires reconnection).
        
        Args:
            instruction: New system instruction
        """
        self.system_instruction = instruction
        logger.info("System instruction updated (reconnect to apply)")


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    
    async def test_client():
        """Test the Gemini Live Client."""
        
        def on_audio(audio_data):
            print(f"Received audio: {len(audio_data)} bytes")
        
        def on_text(text):
            print(f"Received text: {text}")
        
        def on_error(error):
            print(f"Error: {error}")
        
        # Create client
        client = GeminiLiveClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            on_audio_received=on_audio,
            on_text_received=on_text,
            on_error=on_error
        )
        
        # Connect
        if await client.connect():
            print("Connected successfully!")
            
            # Send a test message
            await client.send_text("Hello, can you hear me?")
            
            # Wait for response
            await asyncio.sleep(5)
            
            # Disconnect
            await client.disconnect()
        else:
            print("Failed to connect")
    
    asyncio.run(test_client())
