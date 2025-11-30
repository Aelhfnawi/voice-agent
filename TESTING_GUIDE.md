# Testing Guide - Voice Agent with Gemini Live API

## Quick Start

### Option 1: One-Click Start (Windows)
Double-click `start_app.bat` in the root directory.

This will automatically open 3 terminal windows:
1. **Backend API** - Port 8000
2. **Voice Agent** - LiveKit agent in dev mode
3. **Frontend** - Port 3000 (will open browser automatically)

### Option 2: Manual Start

#### Terminal 1: Backend API
```bash
cd project
uvicorn api.server:app --host 127.0.0.1 --port 8000
```
**Expected output:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
```

#### Terminal 2: Voice Agent
```bash
cd project
python agent/livekit_agent.py dev
```
**Expected output:**
```
INFO:agent.livekit_agent:GeminiRAGVoiceAgent initialized
INFO:livekit.agents:Worker started
```

#### Terminal 3: Frontend
```bash
cd frontend
npm start
```
**Expected output:**
```
Compiled successfully!
You can now view the app in the browser.
Local: http://localhost:3000
```

## Testing Steps

### 1. Open the Application
- Browser should auto-open to `http://localhost:3000`
- If not, manually navigate to `http://localhost:3000`

### 2. Join a Room
- **Room Name**: Enter any name (e.g., "test-room")
- **Your Name**: Enter your name (e.g., "John")
- Click **"Join Room"**
- Allow microphone access when prompted

### 3. Call the Agent
- Click the **"📞 Call Agent"** button
- Wait 2-3 seconds for agent to join
- You should see the agent appear in the Participants list

### 4. Test Text Interaction (Recommended First)

#### Test 1: Simple Greeting
- Type: "Hello"
- Press Enter or click Send
- **Expected**: Agent responds with a greeting

#### Test 2: RAG Query
- Type: "What is this document about?"
- Press Enter
- **Expected**: 
  - Agent retrieves context from knowledge base
  - Responds with information from the PDF
  - RAG Context panel shows "RAG System Active"

#### Test 3: Follow-up Question
- Type a follow-up based on the document content
- **Expected**: Agent provides relevant answer

### 5. Test Voice Interaction (Optional)

> **Note**: Voice interaction requires the agent to process audio streams. This is more complex and may need additional debugging.

- Speak into your microphone
- The agent should process your speech via Gemini Live API
- You should hear a voice response

## Troubleshooting

### Backend API Issues

**Problem**: Port 8000 already in use
```bash
# Find and kill the process
netstat -ano | findstr :8000
taskkill /PID <process_id> /F
```

**Problem**: Import errors
```bash
cd project
pip install -r requirements.txt
```

### Voice Agent Issues

**Problem**: "LiveKit credentials not configured"
- Check `project/.env` file
- Verify `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` are set

**Problem**: "Failed to connect to Gemini Live API"
- Check `GEMINI_API_KEY` in `.env`
- Run: `python test_gemini_live_access.py` to verify API access

**Problem**: "No vector store found"
```bash
cd project
python main.py --ingest --data-dir ./data/raw
```

### Frontend Issues

**Problem**: Dependencies not installed
```bash
cd frontend
npm install
```

**Problem**: Port 3000 already in use
```bash
# Kill the process or use a different port
set PORT=3001 && npm start
```

### Connection Issues

**Problem**: Agent doesn't join the room
1. Check that Backend API is running (http://localhost:8000)
2. Check that Voice Agent is running (no errors in terminal)
3. Try clicking "Call Agent" again
4. Check browser console for errors (F12)

**Problem**: No audio/text response
1. Check Voice Agent terminal for errors
2. Verify Gemini API key is valid
3. Check that RAG system is initialized (should show "22 docs" in logs)

## Expected Behavior

### ✅ Successful Test Indicators

1. **Backend API**: 
   - Shows "Uvicorn running on http://127.0.0.1:8000"
   - No errors in terminal

2. **Voice Agent**:
   - Shows "GeminiRAGVoiceAgent initialized"
   - Shows "RAG ready: 22 docs"
   - Shows "Agent starting in room: <room_name>" when you join

3. **Frontend**:
   - Shows "Connected to LiveKit" status
   - Shows participants (You + Agent)
   - Messages appear in transcript
   - RAG Context panel shows "RAG System Active"

### 🎯 Test Scenarios

#### Scenario 1: Knowledge Base Query
```
You: "What is this document about?"
Agent: [Retrieves context from PDF and responds with summary]
```

#### Scenario 2: General Question
```
You: "Hello, how are you?"
Agent: [Responds conversationally without RAG]
```

#### Scenario 3: Specific Document Question
```
You: "Tell me about [topic from the PDF]"
Agent: [Uses RAG to provide accurate answer from document]
```

## Verification Checklist

- [ ] Backend API starts without errors
- [ ] Voice Agent starts and shows "RAG ready: 22 docs"
- [ ] Frontend opens in browser
- [ ] Can join a room
- [ ] Can call the agent
- [ ] Agent appears in participants list
- [ ] Can send text messages
- [ ] Agent responds to messages
- [ ] RAG context panel shows active status
- [ ] Agent provides relevant answers from knowledge base

## Next Steps After Testing

1. **If everything works**: Ready for demo video!
2. **If issues found**: Note the specific errors and we'll debug
3. **For production**: Consider adding audio resampling and VAD

## Quick Commands Reference

```bash
# Test Gemini API access
cd project
python test_gemini_live_access.py

# Test Gemini Live Client
cd project
python test_gemini_live_client.py

# Ingest documents
cd project
python main.py --ingest --data-dir ./data/raw

# Query RAG system (CLI)
cd project
python main.py --query "What is this document about?"
```

## Support

If you encounter issues:
1. Check the terminal outputs for error messages
2. Verify all environment variables in `.env`
3. Ensure all dependencies are installed
4. Check that ports 8000 and 3000 are available
