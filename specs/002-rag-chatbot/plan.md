# Implementation Plan: RAG Chatbot

**Branch**: `002-rag-chatbot` | **Date**: 2025-12-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-rag-chatbot/spec.md`

## Summary

Build an embedded RAG chatbot widget that provides book-wide and selected-text Q&A functionality. The chatbot enforces zero hallucination by only answering from retrieved book content. It integrates with the backend API (003-backend-api) for RAG processing and displays answers with source references.

## Technical Context

**Language/Version**: TypeScript 5.x, React 18
**Primary Dependencies**: React 18, Axios/Fetch API, @docusaurus/theme-common (for theming)
**Storage**: localStorage for conversation history (client-side)
**Testing**: Jest for unit tests, React Testing Library, Playwright for E2E
**Target Platform**: Browser (embedded in Docusaurus book)
**Project Type**: React component library (integrated into book project)
**Performance Goals**: < 100ms UI response, < 5 second API response display
**Constraints**: Must work within Docusaurus React context, SSR-compatible
**Scale/Scope**: Single widget component, ~500 lines of code

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Spec-Driven Development | PASS | Spec complete at spec.md, plan follows spec |
| II. Zero Hallucination | PASS | Widget displays only backend responses; shows "I don't have information" for unanswerable queries (FR-006) |
| III. Context-Only Answers | PASS | Selected-text mode sends only selection to backend (FR-011); UI indicates active mode (FR-003) |
| IV. Clean Architecture | PASS | Component follows single responsibility; API client separated from UI |
| V. Secure by Default | PASS | API URL from environment variable; no secrets in frontend |
| VI. Small Testable Changes | PASS | Component broken into ChatInput, ChatMessage, SourceLink subcomponents |

**Gate Result**: PASSED - No violations

## Project Structure

### Documentation (this feature)

```text
specs/002-rag-chatbot/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # API contract with backend
└── tasks.md             # Phase 2 output (created by /sp.tasks)
```

### Source Code (integrated into book/)

```text
book/src/
├── components/
│   └── ChatbotWidget/
│       ├── index.tsx              # Main widget component
│       ├── ChatbotWidget.module.css
│       ├── ChatInput.tsx          # Message input component
│       ├── ChatMessage.tsx        # Single message display
│       ├── SourceLink.tsx         # Clickable source reference
│       ├── ModeIndicator.tsx      # Book-wide vs Selected mode
│       └── hooks/
│           ├── useChat.ts         # Chat state management
│           ├── useTextSelection.ts # Text selection detection
│           └── useLocalStorage.ts  # Conversation persistence
├── services/
│   └── chatApi.ts                 # API client for backend
└── types/
    └── chat.ts                    # TypeScript interfaces

book/src/__tests__/
├── ChatbotWidget.test.tsx
├── ChatInput.test.tsx
├── ChatMessage.test.tsx
└── useChat.test.ts
```

**Structure Decision**: Chatbot is a React component integrated into the Docusaurus book project (001-docusaurus-book). Uses custom hooks for state management and a separate API service layer.

## Complexity Tracking

No violations requiring justification.

## Implementation Approach

### Phase 1: Core UI Components
1. Create ChatbotWidget container with expand/collapse
2. Implement ChatInput with send functionality
3. Implement ChatMessage for displaying Q&A pairs
4. Add loading and error states

### Phase 2: API Integration
1. Create chatApi service with axios/fetch
2. Implement useChat hook for state management
3. Handle streaming responses (if supported)
4. Add retry logic for failed requests

### Phase 3: Selected-Text Mode
1. Implement useTextSelection hook
2. Add ModeIndicator component
3. Wire selection to chat context
4. Test isolation (selection-only answers)

### Phase 4: Source References
1. Implement SourceLink component
2. Add navigation to source sections
3. Highlight source text on navigation
4. Test source reference accuracy

### Phase 5: Persistence & Polish
1. Implement conversation history (localStorage)
2. Add "Clear Chat" functionality
3. Accessibility improvements (ARIA, keyboard nav)
4. Mobile responsive adjustments
