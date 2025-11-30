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
    from livekit.plugins import google, silero
    print("Imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    exit(1)
