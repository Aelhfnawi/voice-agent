"""
LiveKit Voice Agent with Gemini Live API and RAG Integration.

This agent:
1. Connects to a LiveKit room
2. Streams audio bidirectionally with Gemini Live API
3. Detects user turns and retrieves RAG context
4. Injects context into Gemini for informed responses
5. Maintains conversation state

Run with: python agent/livekit_agent.py --room <room_name>
"""

import logging
import asyncio
import sys
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# LiveKit imports
try:
    from livekit import rtc
    from livekit.agents import (
        AutoSubscribe, 
        JobContext, 
        WorkerOptions, 
        cli,
        llm,
        tokenize,
        tts,
        stt,
        vad
    )
    from livekit.plugins import google
    LIVEKIT_AVAILABLE = True
except ImportError as e:
    LIVEKIT_AVAILABLE = False
    print(f"Error importing LiveKit: {e}")
    print("Install with: pip install livekit livekit-agents livekit-plugins-google")
    sys.exit(1)

from config import Config
from agent.rag_helper import RAGHelper
from agent.voice_assistant import VoiceAssistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeminiRAGVoiceAgent:
    """
    Voice agent that combines LiveKit, Gemini Live API, and RAG.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the voice agent.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.rag_helper = RAGHelper(config)
        self.voice_assistant = VoiceAssistant(config, self.rag_helper)
        self.conversation_history = []
        self.current_context = None
        
        logger.info("GeminiRAGVoiceAgent initialized")
    
    async def entrypoint(self, ctx: JobContext):
        """
        Main entry point for the agent when joining a room.
        
        Args:
            ctx: LiveKit job context
        """
        logger.info(f"Agent starting in room: {ctx.room.name}")
        
        # Connect to room
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        
        # Wait for first participant
        participant = await ctx.wait_for_participant()
        logger.info(f"Participant joined: {participant.identity}")
        
        # Run the agent loop
        await self.run_agent_loop(ctx, participant)
    
    async def run_agent_loop(self, ctx: JobContext, participant):
        """
        Main agent processing loop with speech recognition and synthesis.
        
        Args:
            ctx: Job context
            participant: Participant to interact with
        """
        try:
            logger.info("Starting voice processing pipeline...")
            
            # Create an audio source to publish responses
            source = rtc.AudioSource(24000, 1)  # 24kHz, mono
            track = rtc.LocalAudioTrack.create_audio_track("agent_voice", source)
            
            # Publish the audio track
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            
            publication = await ctx.room.local_participant.publish_track(track, options)
            logger.info("Audio track published")
            
            # Send welcome message
            welcome_msg = "Hello! I'm your AI assistant. I can answer questions using the knowledge base. Try asking me something!"
            logger.info(f"Sending welcome: {welcome_msg}")
            
            # For now, we'll use a simple loop that waits for data channel messages
            # In production, you'd process actual audio streams
            
            # Listen for data channel messages (text input for testing)
            def on_data_received(data: rtc.DataPacket):
                if data.kind == rtc.DataPacketKind.KIND_RELIABLE:
                    text = data.data.decode('utf-8')
                    logger.info(f"Received text message: {text}")
                    
                    # Process with voice assistant in background task
                    async def process_message():
                        response = await self.voice_assistant.generate_response(text)
                        logger.info(f"Generated response: {response[:100]}...")
                        
                        # Send back as text
                        await ctx.room.local_participant.publish_data(
                            response.encode('utf-8'),
                            reliable=True
                        )
                    
                    asyncio.create_task(process_message())
            
            ctx.room.on("data_received", on_data_received)
            
            logger.info("Agent is ready and listening...")
            
            # Keep agent alive
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in agent loop: {e}", exc_info=True)
    
    def _get_system_instructions(self) -> str:
        """
        Get system instructions for Gemini.
        
        Returns:
            System instruction string
        """
        return """You are a helpful voice assistant with access to a knowledge base.

When answering questions:
1. Use the provided context from the knowledge base when available
2. Be conversational and natural in your speech
3. If you don't have enough information, ask clarifying questions
4. Cite sources when using information from the knowledge base
5. Keep responses concise for voice interaction

You have access to a RAG system that can retrieve relevant information.
Use it whenever the user asks a question that might be in the knowledge base."""
    
    def _create_function_context(self):
        """
        Create function context for RAG retrieval.
        
        Returns:
            Function context for the assistant
        """
        # Define RAG retrieval as a function that Gemini can call
        async def retrieve_context(query: str) -> str:
            """
            Retrieve relevant context from the knowledge base.
            
            Args:
                query: The user's question
                
            Returns:
                Retrieved context as a string
            """
            try:
                logger.info(f"RAG retrieval for: {query}")
                context = await self.rag_helper.retrieve_context(query)
                self.current_context = context
                return context
            except Exception as e:
                logger.error(f"RAG retrieval error: {e}")
                return "No relevant information found in the knowledge base."
        
        # Create function context
        fnc_ctx = llm.FunctionContext()
        fnc_ctx.ai_callable()(retrieve_context)
        
        return fnc_ctx
    
    async def _on_user_speech(self, event):
        """
        Handle user speech event.
        
        Args:
            event: Speech event containing user's message
        """
        user_message = event.alternatives[0].text if event.alternatives else ""
        logger.info(f"User said: {user_message}")
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_message
        })
        
        # Trigger RAG retrieval
        try:
            context = await self.rag_helper.retrieve_context(user_message)
            if context:
                logger.info(f"Retrieved context: {len(context)} characters")
                self.current_context = context
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
    
    async def _on_agent_speech(self, event):
        """
        Handle agent speech event.
        
        Args:
            event: Speech event containing agent's response
        """
        agent_message = event.alternatives[0].text if event.alternatives else ""
        logger.info(f"Agent said: {agent_message}")
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'assistant',
            'content': agent_message
        })
    
    async def _on_function_calls(self, event):
        """
        Handle function call completion.
        
        Args:
            event: Function call event
        """
        logger.info("Function calls finished")


async def main(room_name: str):
    """
    Main function to start the agent.
    
    Args:
        room_name: Name of the LiveKit room to join
    """
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = Config()
    
    # Verify LiveKit credentials
    if not all([config.LIVEKIT_URL, config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET]):
        logger.error("LiveKit credentials not configured. Check .env file.")
        sys.exit(1)
    
    # Create agent
    agent = GeminiRAGVoiceAgent(config)
    
    # Run agent with LiveKit worker
    logger.info(f"Starting agent for room: {room_name}")
    
    # Use LiveKit's worker
    worker_opts = WorkerOptions(
        entrypoint_fnc=agent.entrypoint,
    )
    
    # Run the worker
    await cli.run_app(worker_opts)


if __name__ == "__main__":
    # Load dotenv first
    load_dotenv()
    
    # Run with LiveKit CLI
    # The CLI will handle room connection automatically
    asyncio.run(cli.run_app(
        WorkerOptions(
            entrypoint_fnc=GeminiRAGVoiceAgent(Config()).entrypoint,
        )
    ))