import React, { useState, useRef, useEffect } from 'react';
import styles from './styles.module.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  mode?: 'book' | 'selected';
}

interface ChatbotProps {
  apiUrl?: string;
}

export default function Chatbot({ apiUrl = 'http://localhost:8000' }: ChatbotProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Hello! I can answer questions about this Physical AI & Humanoid Robotics book. Select text on the page to ask about it, or ask any question about the book!',
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [mode, setMode] = useState<'book' | 'selected'>('book');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Listen for text selection
  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();
      const text = selection?.toString().trim();
      if (text && text.length > 10) {
        setSelectedText(text);
        setMode('selected');
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => document.removeEventListener('mouseup', handleSelection);
  }, []);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    const currentMode = mode;
    const currentSelectedText = selectedText;

    setInput('');
    setMessages((prev) => [
      ...prev,
      {
        role: 'user',
        content: currentMode === 'selected'
          ? `[About selected text] ${userMessage}`
          : userMessage,
        mode: currentMode,
      },
    ]);
    setIsLoading(true);

    try {
      const requestBody: Record<string, string> = {
        question: userMessage,
      };

      // If in selected mode, send the selected text
      if (currentMode === 'selected' && currentSelectedText) {
        requestBody.selected_text = currentSelectedText;
      }

      const response = await fetch(`${apiUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer || 'Sorry, I could not find an answer.',
          mode: currentMode,
        },
      ]);
    } catch (error) {
      console.error('Chat error:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, there was an error connecting to the chatbot service. Please make sure the backend is running.',
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearSelection = () => {
    setSelectedText('');
    setMode('book');
  };

  return (
    <>
      {/* Chat Button */}
      <button
        className={styles.chatButton}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Open chat"
      >
        {isOpen ? '\u00D7' : '\uD83D\uDCAC'}
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className={styles.chatWindow}>
          <div className={styles.chatHeader}>
            <span>Physical AI Assistant</span>
            <button onClick={() => setIsOpen(false)} aria-label="Close chat">{'\u00D7'}</button>
          </div>

          {/* Mode Toggle */}
          <div className={styles.modeToggle}>
            <button
              className={`${styles.modeButton} ${mode === 'book' ? styles.activeMode : ''}`}
              onClick={() => { setMode('book'); setSelectedText(''); }}
            >
              📚 Entire Book
            </button>
            <button
              className={`${styles.modeButton} ${mode === 'selected' ? styles.activeMode : ''}`}
              onClick={() => setMode('selected')}
              disabled={!selectedText}
            >
              ✂️ Selected Text
            </button>
          </div>

          {/* Mode indicator */}
          {selectedText && mode === 'selected' && (
            <div className={styles.modeIndicator}>
              <span className={styles.modeTag}>Selected Text Mode</span>
              <span className={styles.selectedPreview}>
                "{selectedText.slice(0, 50)}..."
              </span>
              <button onClick={clearSelection} className={styles.clearButton}>
                Clear
              </button>
            </div>
          )}

          <div className={styles.chatMessages}>
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`${styles.message} ${
                  msg.role === 'user' ? styles.userMessage : styles.assistantMessage
                }`}
              >
                {msg.content}
              </div>
            ))}
            {isLoading && (
              <div className={`${styles.message} ${styles.assistantMessage}`}>
                <span className={styles.typingIndicator}>
                  <span></span>
                  <span></span>
                  <span></span>
                </span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className={styles.chatInput}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                mode === 'selected'
                  ? 'Ask about selected text...'
                  : 'Ask about the book...'
              }
              disabled={isLoading}
            />
            <button onClick={sendMessage} disabled={isLoading || !input.trim()}>
              Send
            </button>
          </div>
        </div>
      )}
    </>
  );
}
