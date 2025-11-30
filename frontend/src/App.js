import React, { useState, useEffect } from 'react';
import {
  LiveKitRoom,
  useParticipants,
  useLocalParticipant,
  useTracks,
  useRoomContext,
  RoomAudioRenderer,
} from '@livekit/components-react';
import { Track } from 'livekit-client';
import axios from 'axios';
import '@livekit/components-styles';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [token, setToken] = useState('');
  const [roomName, setRoomName] = useState('');
  const [participantName, setParticipantName] = useState('');
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState('');

  // Join room handler
  const handleJoinRoom = async () => {
    if (!roomName || !participantName) {
      setError('Please enter room name and your name');
      return;
    }

    setIsConnecting(true);
    setError('');

    try {
      // Get token from backend
      const response = await axios.post(`${API_URL}/livekit/token`, {
        room_name: roomName,
        participant_name: participantName,
      });

      setToken(response.data.token);
    } catch (err) {
      setError(`Failed to join room: ${err.message}`);
      setIsConnecting(false);
    }
  };

  // Disconnect handler
  const handleDisconnect = () => {
    setToken('');
    setIsConnecting(false);
  };

  // If not connected, show join form
  if (!token) {
    return (
      <div className="App">
        <div className="join-container">
          <h1>🎤 Voice AI Agent</h1>
          <p className="subtitle">Talk to your RAG-powered assistant</p>

          {error && <div className="error">{error}</div>}

          <div className="form-group">
            <label>Room Name</label>
            <input
              type="text"
              value={roomName}
              onChange={(e) => setRoomName(e.target.value)}
              placeholder="Enter room name"
              disabled={isConnecting}
            />
          </div>

          <div className="form-group">
            <label>Your Name</label>
            <input
              type="text"
              value={participantName}
              onChange={(e) => setParticipantName(e.target.value)}
              placeholder="Enter your name"
              disabled={isConnecting}
            />
          </div>

          <button
            onClick={handleJoinRoom}
            disabled={isConnecting}
            className="join-button"
          >
            {isConnecting ? 'Connecting...' : 'Join Room'}
          </button>

          <div className="info-box">
            <h3>ℹ️ How it works:</h3>
            <ol>
              <li>Enter a room name and your name</li>
              <li>Click "Join Room" to connect</li>
              <li>Allow microphone access when prompted</li>
              <li>Start talking - the AI agent will respond!</li>
            </ol>
          </div>
        </div>
      </div>
    );
  }

  // If connected, show the room
  return (
    <div className="App">
      <LiveKitRoom
        token={token}
        serverUrl="wss://voice-agent-task-bmujyln2.livekit.cloud"
        connect={true}
        audio={true}
        video={false}
        onDisconnected={handleDisconnect}
        className="room-container"
      >
        <RoomContent
          roomName={roomName}
          participantName={participantName}
          onDisconnect={handleDisconnect}
        />
        <RoomAudioRenderer />
      </LiveKitRoom>
    </div>
  );
}

// Room content component
function RoomContent({ roomName, participantName, onDisconnect }) {
  const room = useRoomContext();
  const participants = useParticipants();
  const { localParticipant } = useLocalParticipant();
  const tracks = useTracks([Track.Source.Microphone]);
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');

  useEffect(() => {
    console.log('Participants:', participants.length);
    console.log('Local participant:', localParticipant?.identity);
  }, [participants, localParticipant]);

  // Send message
  const handleSendMessage = async () => {
    if (!inputText.trim() || !localParticipant) return;

    const userMessage = {
      role: 'user',
      content: inputText,
      timestamp: new Date().toLocaleTimeString()
    };
    setMessages(prev => [...prev, userMessage]);

    try {
      console.log('Sending message:', inputText);

      // Send to agent via data channel
      const encoder = new TextEncoder();
      await localParticipant.publishData(
        encoder.encode(inputText),
        { reliable: true }
      );

      setInputText('');
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  // Listen for agent responses - FIXED VERSION
  useEffect(() => {
    if (!room) return;

    const handleDataReceived = (payload, participant) => {
      console.log('Data received event triggered!');
      console.log('From participant:', participant?.identity);
      console.log('Payload:', payload);

      try {
        const decoder = new TextDecoder();
        const text = decoder.decode(payload);

        console.log('Decoded text:', text);

        // Always add agent messages
        const agentMessage = {
          role: 'agent',
          content: text,
          timestamp: new Date().toLocaleTimeString()
        };

        console.log('Adding agent message:', agentMessage);
        setMessages(prev => [...prev, agentMessage]);
      } catch (error) {
        console.error('Error processing received data:', error);
      }
    };

    console.log('Setting up dataReceived listener on ROOM');
    room.on('dataReceived', handleDataReceived);

    return () => {
      console.log('Cleaning up dataReceived listener');
      room.off('dataReceived', handleDataReceived);
    };
  }, [room]);

  return (
    <div className="room-content">
      {/* Header */}
      <div className="room-header">
        <div className="room-info">
          <h2>🎤 {roomName}</h2>
          <p>Connected as: {participantName}</p>
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={async () => {
              try {
                await axios.post(`${API_URL}/livekit/dispatch-agent`, {
                  room_name: roomName
                });
                console.log('Agent dispatched!');
              } catch (error) {
                console.error('Error dispatching agent:', error);
              }
            }}
            className="join-button"
            style={{ padding: '8px 16px', fontSize: '14px' }}
          >
            📞 Call Agent
          </button>
          <button onClick={onDisconnect} className="disconnect-button">
            Disconnect
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="room-main">
        {/* Participants */}
        <div className="participants-panel">
          <h3>Participants ({participants.length})</h3>
          <div className="participants-list">
            {participants.map((participant) => (
              <div key={participant.identity} className="participant">
                <span className="participant-icon">
                  {participant.identity === localParticipant?.identity
                    ? '🎤'
                    : '🤖'}
                </span>
                <span className="participant-name">
                  {participant.identity}
                  {participant.identity === localParticipant?.identity &&
                    ' (You)'}
                </span>
                {participant.isSpeaking && (
                  <span className="speaking-indicator">🔊</span>
                )}
              </div>
            ))}
          </div>

          {/* RAG Context Panel */}
          <div className="rag-context-panel">
            <h3>📚 Knowledge Base Context</h3>
            <div className="rag-context-content">
              {messages.length > 0 ? (
                <div className="context-info">
                  <p className="context-hint">
                    When you ask questions, the agent retrieves relevant
                    information from the knowledge base to provide accurate
                    answers.
                  </p>
                  <div className="context-status">
                    <span className="status-dot active"></span>
                    RAG System Active
                  </div>
                </div>
              ) : (
                <p className="context-empty">
                  Ask a question to see retrieved context
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Transcript */}
        <div className="transcript-panel">
          <h3>🎙️ Live Conversation</h3>
          <div className="transcript">
            {messages.length === 0 ? (
              <div className="transcript-empty">
                <p>🎙️ Ready to chat!</p>
                <p className="hint">
                  Click "Call Agent" to start, then speak or type your question
                </p>
                <p className="hint">
                  Try asking: "What is this document about?"
                </p>
              </div>
            ) : (
              messages.map((message, index) => (
                <div
                  key={index}
                  className={`message ${message.role}`}
                >
                  <strong>
                    {message.role === 'user' ? '👤 You' : '🤖 Agent'}:
                  </strong>
                  <p>{message.content}</p>
                  {message.timestamp && (
                    <span className="timestamp">{message.timestamp}</span>
                  )}
                </div>
              ))
            )}
          </div>

          {/* Text Input */}
          <div className="text-input-container">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Type your question... (e.g., 'What is this about?')"
              className="text-input"
            />
            <button onClick={handleSendMessage} className="send-button">
              Send
            </button>
          </div>
        </div>
      </div>

      {/* Status bar */}
      <div className="status-bar">
        <div className="status-item">
          <span className="status-dot connected"></span>
          Connected to LiveKit
        </div>
        <div className="status-item">
          {tracks.length > 0 ? (
            <>
              <span className="status-dot active"></span>
              Microphone Active
            </>
          ) : (
            <>
              <span className="status-dot inactive"></span>
              Microphone Inactive
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;