"""
Test Gemini Live API access and list available models.
"""
from google.genai import Client
import os
from dotenv import load_dotenv

load_dotenv()

client = Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Testing Gemini Live API access...")
print("\nAvailable models:")
print("-" * 80)

models = client.models.list()
for m in models:
    print(f"  {m.name}")

print("-" * 80)

# Check for the native audio model
target_model = "gemini-2.5-flash-native-audio-preview-09-2025"
model_names = [m.name for m in models]

if target_model in model_names:
    print(f"\n[SUCCESS] {target_model} is available!")
else:
    print(f"\n[WARNING] {target_model} not found in available models.")
    print("   You may need to request access or use an alternative model.")
