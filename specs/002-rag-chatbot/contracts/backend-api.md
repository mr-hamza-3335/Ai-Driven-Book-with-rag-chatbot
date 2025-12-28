# Contract: Backend API Integration

**Feature**: 002-rag-chatbot
**Date**: 2025-12-27
**Depends On**: 003-backend-api

## Overview

The RAG Chatbot widget communicates with the Backend API to process questions and retrieve answers. This document defines the expected API contract.

## Base URL

```
Development: http://localhost:8000
Production: Configured via REACT_APP_CHATBOT_API_URL
```

## Endpoints

### POST /api/chat

Process a chat question and return RAG-based answer.

**Request Headers**:
```
Content-Type: application/json
```

**Request Body**:
```json
{
  "question": "What is supervised learning?",
  "selectedText": null,
  "conversationId": "conv_abc123"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| question | string | Yes | User's question (1-2000 chars) |
| selectedText | string | No | Selected text for context-only mode (10-5000 chars) |
| conversationId | string | No | Previous conversation ID for follow-ups |

**Response 200 OK**:
```json
{
  "answer": "Supervised learning is a type of machine learning where the algorithm learns from labeled training data...",
  "sources": [
    {
      "chapterId": "chapter-1-intro-to-ai",
      "sectionTitle": "Types of Machine Learning",
      "snippet": "In supervised learning, the algorithm is trained on a labeled dataset...",
      "relevanceScore": 0.95
    }
  ],
  "conversationId": "conv_abc123",
  "processingTimeMs": 1250
}
```

| Field | Type | Description |
|-------|------|-------------|
| answer | string | RAG-generated answer grounded in book content |
| sources | Source[] | References to book content used in answer |
| conversationId | string | ID for follow-up questions |
| processingTimeMs | number | Backend processing time in milliseconds |

**Response 400 Bad Request**:
```json
{
  "error": "Question is required",
  "code": "VALIDATION_ERROR"
}
```

**Response 429 Too Many Requests**:
```json
{
  "error": "Rate limit exceeded. Please wait before sending more requests.",
  "code": "RATE_LIMIT",
  "retryAfter": 60
}
```

**Response 500 Internal Server Error**:
```json
{
  "error": "An error occurred processing your request",
  "code": "SERVER_ERROR"
}
```

**Response 503 Service Unavailable**:
```json
{
  "error": "Search service temporarily unavailable",
  "code": "SERVICE_UNAVAILABLE"
}
```

**Response 504 Gateway Timeout**:
```json
{
  "error": "Answer generation timed out, please retry",
  "code": "TIMEOUT"
}
```

## Error Handling

The chatbot widget handles errors as follows:

| HTTP Status | User Message | Retry? |
|-------------|--------------|--------|
| 400 | Display error message | No |
| 429 | "Too many requests. Please wait X seconds." | After retryAfter |
| 500 | "Something went wrong. Please try again." | Manual retry button |
| 503 | "Service temporarily unavailable." | Auto-retry after 5s (max 3) |
| 504 | "Request timed out. Please try again." | Manual retry button |
| Network Error | "Unable to connect. Check your connection." | Manual retry button |

## Source Object Schema

```typescript
interface Source {
  chapterId: string;        // Directory name, e.g., "chapter-1-intro-to-ai"
  sectionTitle: string;     // Human-readable section title
  snippet: string;          // 50-200 char excerpt from source
  relevanceScore: number;   // 0.0-1.0 confidence score
}
```

## CORS Requirements

The backend MUST allow requests from:
- `http://localhost:3000` (development)
- `https://<username>.github.io` (production)

Required headers:
```
Access-Control-Allow-Origin: <book-origin>
Access-Control-Allow-Methods: POST, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

## Rate Limiting

- **Limit**: 60 requests per minute per IP
- **Headers**: `X-RateLimit-Remaining`, `X-RateLimit-Reset`
- **Response**: 429 with `retryAfter` field

## Timeouts

- **Client Timeout**: 30 seconds
- **Expected Response**: < 5 seconds for 90% of requests
