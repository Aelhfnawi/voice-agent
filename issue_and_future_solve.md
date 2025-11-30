# Issue and Future Solutions

## Current Issue: Audio Streaming Instability
**Symptom:** The user hears "Hi there" followed by gibberish or silence.
**Error Log:** `Exception: an RtcError occured: InvalidState - failed to capture frame`
**Status:** ⚠️ Partially Working (Text Chat & RAG work, Voice Streaming has bugs)

### Root Cause Analysis
The integration between the **Gemini Live API** (WebSocket) and **LiveKit** (WebRTC) is experiencing synchronization issues.
1.  **Race Condition:** The agent receives audio chunks from Gemini *before* the LiveKit audio track is fully initialized and ready to accept frames.
2.  **Frame Capture Error:** The `source.capture_frame()` method fails because the internal state of the track is not yet "published" or "ready," leading to `InvalidState`.
3.  **Buffer Underrun/Overrun:** The mismatch in timing causes audio data to be dropped or corrupted ("gibberish").

---

## Future Solutions

### 1. Robust Audio Synchronization (Immediate Fix)
**Goal:** Prevent `capture_frame` from being called until the track is 100% ready.
- **Implementation:**
    - Add a strict `is_ready` flag that is only set to `True` after `publish_track` returns successfully **AND** a small safety delay has passed.
    - Implement a **Ring Buffer**: Store incoming audio from Gemini in a buffer while waiting for the track. Once ready, drain the buffer into the track.
    - Add error handling to `capture_frame` to silently drop frames during initialization instead of crashing the loop.

### 2. Client-Side TTS (Alternative Approach)
**Goal:** Bypass server-side streaming complexity.
- **Implementation:**
    - Agent sends **Text** responses to the frontend via LiveKit Data Channel.
    - Frontend uses the browser's native `SpeechSynthesis` API or a stable cloud TTS (like ElevenLabs/Google Cloud TTS) to generate audio locally.
- **Pros:** Extremely stable, no synchronization issues, lower latency perception.
- **Cons:** Lose the specific "voice" and prosody of the Gemini Live model.

### 3. Official Plugin Integration (Long Term)
**Goal:** Use supported libraries instead of custom WebSocket implementation.
- **Implementation:**
    - Wait for `livekit-plugins-google` to officially support the `gemini-2.5-flash-native-audio` model.
    - Replace the custom `GeminiLiveClient` with the official LiveKit plugin once available.
- **Pros:** Guaranteed compatibility and maintenance by the LiveKit team.

---

## Current Workaround
**Use Text Chat:**
The **RAG system** and **Gemini Intelligence** are fully functional.
- **Input:** Type questions in the chat box.
- **Output:** Read text responses (with citations).
- **Benefit:** Verifies the core logic (Knowledge Base + AI) works, isolating the issue to just the audio transport layer.
