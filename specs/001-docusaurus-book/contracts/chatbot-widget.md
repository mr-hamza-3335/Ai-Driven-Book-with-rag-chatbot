# Contract: Chatbot Widget Integration

**Feature**: 001-docusaurus-book
**Date**: 2025-12-27

## Overview

The Docusaurus book provides a `ChatbotWidget` React component that integrates with the backend API (feature 003-backend-api). This document defines the integration contract.

## Component Interface

```typescript
interface ChatbotWidgetProps {
  apiUrl: string;           // Backend API URL (from env)
  position?: "bottom-right" | "bottom-left";  // Widget position
  initiallyOpen?: boolean;  // Start expanded
}
```

## Expected Backend API

The widget expects the backend to provide:

### POST /api/chat

**Request**:
```typescript
interface ChatRequest {
  question: string;
  selectedText?: string;    // For selected-text mode
  conversationId?: string;  // For follow-up questions
}
```

**Response**:
```typescript
interface ChatResponse {
  answer: string;
  sources: Source[];
  conversationId: string;
  processingTimeMs: number;
}

interface Source {
  chapterId: string;
  sectionTitle: string;
  snippet: string;
  relevanceScore: number;
}
```

## Widget Behavior

### States

1. **Collapsed**: Shows floating button icon
2. **Expanded**: Shows chat interface
3. **Loading**: Shows spinner while waiting for response
4. **Error**: Shows error message with retry option

### Text Selection Integration

When user selects text on the page:
1. Widget detects selection via `document.getSelection()`
2. Shows "Ask about selection" option
3. Sends `selectedText` in request
4. Displays response with "Context: Selected text" indicator

### Source Navigation

When user clicks a source reference:
1. Widget collapses (optional)
2. Page navigates to `#chapterId` anchor
3. Section is highlighted briefly

## Environment Configuration

```typescript
// Expected environment variable
const CHATBOT_API_URL = process.env.REACT_APP_CHATBOT_API_URL
  || 'https://api.example.com';
```

## Error Handling

| Error | User Message | Action |
|-------|--------------|--------|
| Network error | "Unable to connect. Please try again." | Show retry button |
| 429 Rate limit | "Too many requests. Please wait." | Show countdown |
| 500 Server error | "Something went wrong. Please try again." | Show retry button |
| Empty response | "I don't have information about that." | Show suggestion |

## Accessibility

- Widget is keyboard navigable (Tab, Enter, Escape)
- ARIA labels for all interactive elements
- Focus management when opening/closing
- Screen reader announcements for new messages
