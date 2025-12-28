# Data Model: RAG Chatbot

**Feature**: 002-rag-chatbot
**Date**: 2025-12-27

## Overview

This feature uses client-side state management with localStorage persistence. No server-side database is required for the chatbot widget itself (RAG processing happens in backend-api).

## TypeScript Interfaces

### Chat Types (`types/chat.ts`)

```typescript
// Chat mode
export type ChatMode = "book-wide" | "selected-text";

// Single message in conversation
export interface ChatMessage {
  id: string;                    // UUID
  role: "user" | "assistant";
  content: string;
  timestamp: number;             // Unix timestamp
  sources?: Source[];            // Only for assistant messages
  mode: ChatMode;                // Mode when message was sent
  selectedText?: string;         // Only if mode is "selected-text"
}

// Source reference in answer
export interface Source {
  chapterId: string;             // e.g., "chapter-3-algorithms"
  sectionTitle: string;          // e.g., "Linear Regression"
  snippet: string;               // Brief excerpt
  relevanceScore: number;        // 0-1 confidence
  anchorId?: string;             // For deep linking
}

// Full conversation
export interface Conversation {
  id: string;                    // UUID
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
}

// Widget state
export interface ChatWidgetState {
  isOpen: boolean;
  isLoading: boolean;
  error: string | null;
  mode: ChatMode;
  selectedText: string | null;
  conversation: Conversation | null;
}
```

### API Types (matching backend contract)

```typescript
// Request to backend
export interface ChatRequest {
  question: string;
  selectedText?: string;
  conversationId?: string;
}

// Response from backend
export interface ChatResponse {
  answer: string;
  sources: Source[];
  conversationId: string;
  processingTimeMs: number;
}

// Error response
export interface ChatError {
  error: string;
  code: "RATE_LIMIT" | "SERVER_ERROR" | "TIMEOUT" | "NETWORK_ERROR";
  retryAfter?: number;           // Seconds until retry allowed
}
```

## localStorage Schema

### Key: `chatbot_conversation`

```typescript
interface StoredConversation {
  version: 1;                    // Schema version for migrations
  conversation: Conversation;
}
```

**Example**:
```json
{
  "version": 1,
  "conversation": {
    "id": "conv_123abc",
    "messages": [
      {
        "id": "msg_001",
        "role": "user",
        "content": "What is supervised learning?",
        "timestamp": 1703702400000,
        "mode": "book-wide"
      },
      {
        "id": "msg_002",
        "role": "assistant",
        "content": "Supervised learning is a type of machine learning where...",
        "timestamp": 1703702403000,
        "mode": "book-wide",
        "sources": [
          {
            "chapterId": "chapter-1-intro-to-ai",
            "sectionTitle": "Types of Machine Learning",
            "snippet": "In supervised learning, the algorithm learns from labeled data...",
            "relevanceScore": 0.95,
            "anchorId": "types-of-machine-learning"
          }
        ]
      }
    ],
    "createdAt": 1703702400000,
    "updatedAt": 1703702403000
  }
}
```

## State Transitions

### Widget State Machine

```
CLOSED → OPEN (user clicks widget button)
OPEN → CLOSED (user clicks close or outside)
OPEN + IDLE → OPEN + LOADING (user sends message)
OPEN + LOADING → OPEN + IDLE (response received)
OPEN + LOADING → OPEN + ERROR (request failed)
OPEN + ERROR → OPEN + LOADING (user clicks retry)
```

### Mode Transitions

```
BOOK_WIDE → SELECTED_TEXT (user selects text on page)
SELECTED_TEXT → BOOK_WIDE (user clicks "Ask about whole book" or clears selection)
SELECTED_TEXT → SELECTED_TEXT (user selects different text)
```

## Validation Rules

### Message Content
- Question: Required, 1-2000 characters
- Answer: Required, non-empty from backend

### Selected Text
- Minimum: 10 characters (spec requirement FR-012)
- Maximum: 5000 characters (edge case limit)

### Conversation
- Maximum messages: 100 (auto-trim oldest)
- Maximum age: 24 hours (auto-clear on load)

## Component Props

### ChatbotWidget

```typescript
interface ChatbotWidgetProps {
  apiUrl: string;                // Required: Backend API URL
  position?: "bottom-right" | "bottom-left";
  initiallyOpen?: boolean;
  maxHeight?: number;            // Widget max height in px
}
```

### ChatMessage Component

```typescript
interface ChatMessageProps {
  message: ChatMessage;
  onSourceClick: (source: Source) => void;
}
```

### ChatInput Component

```typescript
interface ChatInputProps {
  onSend: (question: string) => void;
  disabled: boolean;
  placeholder?: string;
}
```

## Notes

- All timestamps are Unix milliseconds (Date.now())
- UUIDs generated with `crypto.randomUUID()` or fallback
- localStorage has ~5MB limit; conversation pruning prevents overflow
- Selected text is NOT persisted to localStorage (ephemeral)
