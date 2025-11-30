# Quick Start Instructions

## Testing the Voice Agent System

### Current Status
✅ Backend API is already running on port 8000

### Next Steps

#### Option 1: Use the Batch File (Easiest)
1. Close the current backend terminal
2. Double-click `start_app.bat` in the voice agent folder
3. This will start all 3 services automatically

#### Option 2: Manual Start (Recommended for Testing)

**Step 1: Start Voice Agent**
Open a new terminal and run:
```bash
cd "c:\Users\medoo\OneDrive\Desktop\voice agent\project"
python agent/livekit_agent.py dev
```

**Step 2: Start Frontend**
Open another new terminal and run:
```bash
cd "c:\Users\medoo\OneDrive\Desktop\voice agent\frontend"
npm start
```

**Step 3: Test in Browser**
1. Browser should auto-open to http://localhost:3000
2. Enter room name: "test-room"
3. Enter your name
4. Click "Join Room"
5. Click "📞 Call Agent"
6. Type a message: "What is this document about?"

### What to Expect
- Agent should appear in participants list
- Agent should respond to your messages
- RAG Context panel should show "RAG System Active"
- Responses should reference the PDF content

### Troubleshooting
See TESTING_GUIDE.md for detailed troubleshooting steps.
