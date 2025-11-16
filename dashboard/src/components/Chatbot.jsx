import React, { useState, useRef, useEffect } from 'react';
import { Card, Form, Button, Spinner } from 'react-bootstrap';

export default function Chatbot() {
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  const handleChatSubmit = (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    
    setChatHistory(prev => [...prev, { role: 'user', content: chatInput, timestamp: new Date() }]);
    setChatInput('');
    setIsTyping(true);
    
    setTimeout(() => {
      setChatHistory(prev => [...prev, {
        role: 'assistant',
        content: 'This is a simulated AI response. In production, this would connect to a real AI service like OpenAI GPT or Claude.',
        timestamp: new Date()
      }]);
      setIsTyping(false);
    }, 1500);
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <Card className="fade-in">
      <Card.Header as="h5">💬 AI Chat Assistant</Card.Header>
      <Card.Body>
        <div 
          className="chat-container p-3 mb-3"
          style={{ 
            height: '450px', 
            overflowY: 'auto',
            border: '2px solid rgb(196, 196, 196)',
            borderRadius: '0.5rem',
            backgroundColor: '#f8f9fa'
          }}
          role="log"
          aria-live="polite"
          aria-label="Chat conversation"
        >
          {chatHistory.length === 0 ? (
            <div className="text-center text-muted mt-5">
              <div className="fs-1 mb-3">🤖</div>
              <p className="fs-5 fw-semibold">Start a conversation</p>
              <p className="small">Ask me anything about your predictions or the ML model</p>
            </div>
          ) : (
            <>
              {chatHistory.map((msg, idx) => (
                <div 
                  key={idx} 
                  className={`mb-3 ${msg.role === 'user' ? 'text-end' : ''} fade-in`}
                >
                  <small className="text-muted fw-semibold d-block mb-1">
                    {msg.role === 'user' ? '👤 You' : '🤖 AI Assistant'} · {formatTime(msg.timestamp)}
                  </small>
                  <div 
                    className={`chat-message ${msg.role} d-inline-block p-3 rounded-3 shadow-sm`}
                    style={{ 
                      maxWidth: '80%',
                      wordWrap: 'break-word'
                    }}
                  >
                    {msg.content}
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="mb-3 fade-in">
                  <small className="text-muted fw-semibold d-block mb-1">
                    🤖 AI Assistant
                  </small>
                  <div className="chat-message assistant d-inline-block p-3 rounded-3 shadow-sm">
                    <Spinner animation="grow" size="sm" className="me-1" />
                    <Spinner animation="grow" size="sm" className="me-1" />
                    <Spinner animation="grow" size="sm" />
                  </div>
                </div>
              )}
            </>
          )}
          <div ref={chatEndRef} />
        </div>

        <Form onSubmit={handleChatSubmit}>
          <div className="d-flex gap-2">
            <Form.Control
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="Type your message..."
              aria-label="Chat message input"
              className="shadow-sm"
              disabled={isTyping}
            />
            <Button 
              variant="primary" 
              type="submit"
              className="shadow-sm px-4"
              aria-label="Send chat message"
              disabled={isTyping || !chatInput.trim()}
            >
              Send
            </Button>
          </div>
        </Form>
      </Card.Body>
    </Card>
  );
}
