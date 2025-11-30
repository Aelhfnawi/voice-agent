"""
Test script for Gemini Live Client with audio streaming.
"""
import asyncio
import logging
import os
from dotenv import load_dotenv
from agent.gemini_live_client import GeminiLiveClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_gemini_live_client():
    """Test the Gemini Live Client with text input."""
    
    load_dotenv()
    
    print("\n" + "="*80)
    print("Testing Gemini Live API Client")
    print("="*80 + "\n")
    
    # Track responses
    audio_chunks = []
    text_responses = []
    
    def on_audio(audio_data):
        audio_chunks.append(audio_data)
        print(f"[AUDIO] Received {len(audio_data)} bytes")
    
    def on_text(text):
        text_responses.append(text)
        print(f"[TEXT] {text}")
    
    def on_error(error):
        print(f"[ERROR] {error}")
    
    # Create client
    client = GeminiLiveClient(
        api_key=os.getenv("GEMINI_API_KEY"),
        on_audio_received=on_audio,
        on_text_received=on_text,
        on_error=on_error
    )
    
    # Connect
    print("Connecting to Gemini Live API...")
    if await client.connect():
        print("[SUCCESS] Connected to Gemini Live API\n")
        
        # Test 1: Simple greeting
        print("\n--- Test 1: Simple Greeting ---")
        await client.send_text("Hello! Can you introduce yourself briefly?")
        await asyncio.sleep(5)
        
        # Test 2: Question that might use RAG
        print("\n--- Test 2: Question ---")
        await client.send_text("What can you help me with?")
        await asyncio.sleep(5)
        
        # Test 3: Context injection
        print("\n--- Test 3: Context Injection ---")
        test_context = """
        [Knowledge Base Context]
        This is a test document about voice agents.
        Voice agents use speech recognition and synthesis to interact with users.
        They can be powered by large language models like Gemini.
        [End Context]
        """
        await client.inject_context(test_context)
        await client.send_text("Based on the context, what are voice agents?")
        await asyncio.sleep(5)
        
        # Disconnect
        await client.disconnect()
        print("\n[SUCCESS] Disconnected from Gemini Live API")
        
        # Summary
        print("\n" + "="*80)
        print("Test Summary")
        print("="*80)
        print(f"Total audio chunks received: {len(audio_chunks)}")
        print(f"Total audio bytes: {sum(len(chunk) for chunk in audio_chunks)}")
        print(f"Text responses: {len(text_responses)}")
        
        if audio_chunks:
            print("\n[SUCCESS] Audio streaming is working!")
        else:
            print("\n[WARNING] No audio received. Check API configuration.")
        
    else:
        print("[FAILED] Could not connect to Gemini Live API")

if __name__ == "__main__":
    asyncio.run(test_gemini_live_client())
