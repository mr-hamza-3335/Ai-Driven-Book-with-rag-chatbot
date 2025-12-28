# Research: RAG Chatbot

**Feature**: 002-rag-chatbot
**Date**: 2025-12-27
**Status**: Complete

## Research Tasks

### 1. Chat UI Component Architecture

**Decision**: Single floating widget with expand/collapse behavior

**Rationale**:
- Non-intrusive to reading experience
- Familiar UX pattern (similar to Intercom, Drift)
- Can be positioned bottom-right without blocking content
- Collapse state shows minimal floating button

**Alternatives Considered**:
- Sidebar chat: Takes too much horizontal space
- Modal dialog: Disruptive to reading flow
- Inline chat: Complex integration with markdown

### 2. State Management Approach

**Decision**: Custom React hooks (useChat, useTextSelection)

**Rationale**:
- Docusaurus already uses React, no need for external state library
- Custom hooks provide clean separation of concerns
- Easy to test in isolation
- Minimal bundle size impact

**Alternatives Considered**:
- Redux/Zustand: Overkill for single widget state
- React Context: Good for global state, but hooks suffice here
- MobX: Learning curve for small scope

### 3. Text Selection Detection

**Decision**: `document.getSelection()` with debounced event handler

**Rationale**:
- Native browser API, no dependencies
- Works across all major browsers
- Can detect selection across multiple elements
- Debounce prevents excessive state updates

**Alternatives Considered**:
- Custom selection library: Unnecessary complexity
- Range API only: Less convenient than Selection API

### 4. API Communication

**Decision**: Fetch API with async/await

**Rationale**:
- Built into modern browsers, no dependency needed
- Simpler than axios for straightforward requests
- Sufficient for JSON API communication
- Easy error handling with try/catch

**Alternatives Considered**:
- Axios: Adds dependency for minimal benefit
- SWR/React Query: Overkill for simple chat requests
- WebSocket: Server doesn't require real-time push

### 5. Conversation Persistence

**Decision**: localStorage with JSON serialization

**Rationale**:
- Works offline, no backend required
- Persists across browser sessions
- Simple API for read/write operations
- Sufficient capacity for chat history (~5MB)

**Alternatives Considered**:
- IndexedDB: More complex API for minimal benefit
- SessionStorage: Doesn't persist after closing browser
- Server-side: Adds backend complexity

### 6. Source Navigation

**Decision**: Smooth scroll to anchor with highlight animation

**Rationale**:
- Docusaurus generates heading anchors automatically
- Smooth scroll provides visual feedback
- Temporary highlight helps locate the source
- Native browser behavior, performant

**Alternatives Considered**:
- Open in new tab: Disrupts reading flow
- Modal preview: Extra implementation work
- Copy to clipboard: Doesn't help locate source

### 7. Zero Hallucination Enforcement

**Decision**: Display backend response exactly as received; show default message for empty/error responses

**Rationale**:
- Frontend cannot generate answers (constitution requirement)
- Backend is source of truth for RAG responses
- Default message: "I don't have information about that in the book"
- No client-side text generation or modification

**Alternatives Considered**:
- Client-side LLM: Violates zero hallucination principle
- Cached responses: Could become stale

## Dependencies Summary

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@testing-library/react": "^14.0.0",
    "@testing-library/jest-dom": "^6.0.0",
    "typescript": "^5.3.0"
  }
}
```

## Resolved Clarifications

All technical decisions have been made. No outstanding NEEDS CLARIFICATION items remain.
