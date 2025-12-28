# Quickstart: RAG Chatbot

**Feature**: 002-rag-chatbot
**Date**: 2025-12-27

## Prerequisites

- Completed 001-docusaurus-book setup
- Node.js 18.0 or higher
- Backend API running (003-backend-api)

## Setup

### 1. Ensure Book Project Exists

```bash
cd hackathonnnnn/book
npm install  # If not already done
```

### 2. Configure Backend URL

Create or update `.env.local`:

```bash
# For local development
REACT_APP_CHATBOT_API_URL=http://localhost:8000

# For production (set in GitHub Secrets)
# REACT_APP_CHATBOT_API_URL=https://your-api.example.com
```

### 3. Start Development

```bash
# Terminal 1: Start backend (003-backend-api)
cd backend
uvicorn main:app --reload

# Terminal 2: Start book with chatbot
cd book
npm start
```

## Component Usage

### Basic Integration

The ChatbotWidget is automatically included in the Docusaurus layout. No additional configuration needed.

### Manual Integration (if customizing)

```tsx
import ChatbotWidget from '@site/src/components/ChatbotWidget';

function CustomLayout({ children }) {
  return (
    <>
      {children}
      <ChatbotWidget
        apiUrl={process.env.REACT_APP_CHATBOT_API_URL}
        position="bottom-right"
      />
    </>
  );
}
```

## Testing the Chatbot

### Book-Wide Mode

1. Click the chat bubble icon (bottom-right)
2. Type a question: "What is machine learning?"
3. Verify response includes source references
4. Click a source link to navigate to that section

### Selected-Text Mode

1. Select text on any book page
2. Notice the chat mode changes to "Selected text"
3. Ask: "Explain this in simpler terms"
4. Verify answer only references the selected text
5. Click "Ask about whole book" to return to book-wide mode

## Development Commands

| Command | Description |
|---------|-------------|
| `npm start` | Start dev server with hot reload |
| `npm test` | Run chatbot component tests |
| `npm run test:e2e` | Run Playwright E2E tests |

## Component Testing

```bash
# Run unit tests
npm test -- --testPathPattern=ChatbotWidget

# Run specific test
npm test -- ChatInput.test.tsx

# Watch mode
npm test -- --watch
```

## Project Structure

```
book/src/components/ChatbotWidget/
├── index.tsx              # Main widget (entry point)
├── ChatInput.tsx          # Message input
├── ChatMessage.tsx        # Message display
├── SourceLink.tsx         # Clickable sources
├── ModeIndicator.tsx      # Mode display
└── hooks/
    ├── useChat.ts         # Chat logic
    ├── useTextSelection.ts # Selection detection
    └── useLocalStorage.ts  # Persistence
```

## Verification Checklist

- [ ] Widget appears on all book pages
- [ ] Chat opens/closes correctly
- [ ] Messages send and receive
- [ ] Sources are clickable and navigate correctly
- [ ] Selected-text mode activates on selection
- [ ] "Clear Chat" removes conversation
- [ ] Conversation persists after page reload
- [ ] Error messages display for failed requests
- [ ] Widget is accessible via keyboard

## Troubleshooting

### Widget Not Appearing
- Check if ChatbotWidget is in Docusaurus layout
- Verify no JavaScript errors in console
- Check CSS z-index conflicts

### API Errors
- Verify backend is running
- Check REACT_APP_CHATBOT_API_URL is set
- Check CORS settings on backend

### Text Selection Not Detected
- Ensure selection is at least 10 characters
- Check if selection handler is attached
- Verify no overlapping event handlers
