# Feature Specification: Backend API

**Feature Branch**: `003-backend-api`
**Created**: 2025-12-27
**Status**: Draft
**Input**: User description: "FastAPI backend with Neon Postgres and Qdrant Cloud for RAG chatbot"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Process Chat Queries (Priority: P1)

The frontend chatbot widget sends user questions to the backend, which retrieves relevant content and generates answers.

**Why this priority**: Core backend functionality enabling the chatbot to work.

**Independent Test**: Send a POST request with a question, verify response contains answer and sources, check response time under 5 seconds.

**Acceptance Scenarios**:

1. **Given** a valid chat query, **When** the API receives it, **Then** it returns a JSON response with answer and sources
2. **Given** a query with no relevant content, **When** processed, **Then** the API returns "I don't have information about that in the book"
3. **Given** a malformed request, **When** received, **Then** the API returns a 400 error with helpful message

---

### User Story 2 - Index Book Content (Priority: P1)

An administrator needs to index/re-index book content into the vector store when content changes.

**Why this priority**: Without indexed content, RAG cannot function.

**Independent Test**: Run indexing endpoint with book content, verify vectors are stored in Qdrant, verify search returns results.

**Acceptance Scenarios**:

1. **Given** book markdown files, **When** indexing is triggered, **Then** content is chunked and stored as vectors
2. **Given** updated book content, **When** re-indexing is triggered, **Then** old vectors are replaced with new ones
3. **Given** indexing in progress, **When** checking status, **Then** progress percentage is returned

---

### User Story 3 - Selected-Text Queries (Priority: P1)

The frontend sends selected text along with a question, and the backend answers based only on that text.

**Why this priority**: Core selected-text feature enabling context-specific answers.

**Independent Test**: Send selected text with a question, verify answer references only the selection, not other book content.

**Acceptance Scenarios**:

1. **Given** selected text and a question, **When** the API processes it, **Then** it generates an answer using only that text
2. **Given** selected text too short (< 10 chars), **When** submitted, **Then** the API returns a validation error

---

### User Story 4 - Health Check (Priority: P2)

Operations need to monitor backend health and dependencies (Qdrant, Neon, LLM service).

**Why this priority**: Operational necessity but not user-facing functionality.

**Independent Test**: Call health endpoint, verify all dependency statuses are returned.

**Acceptance Scenarios**:

1. **Given** all services healthy, **When** calling /health, **Then** return 200 with all services "ok"
2. **Given** Qdrant unavailable, **When** calling /health, **Then** return 503 with Qdrant status "error"

---

### User Story 5 - Rate Limiting (Priority: P2)

The system needs to protect against abuse by limiting requests per user/IP.

**Why this priority**: Security and cost control, but core functionality works without it initially.

**Independent Test**: Send more requests than limit, verify 429 response after limit reached.

**Acceptance Scenarios**:

1. **Given** a user within rate limit, **When** they send a request, **Then** it is processed normally
2. **Given** a user exceeding rate limit, **When** they send a request, **Then** they receive 429 with retry-after header

---

### Edge Cases

- What if Qdrant is temporarily unavailable? Return 503 with message "Search service temporarily unavailable"
- What if LLM API times out? Return 504 with message "Answer generation timed out, please retry"
- What if book has no indexed content? Return empty results with message "No content has been indexed yet"
- What if request payload is too large? Return 413 with maximum size guidance

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a POST endpoint for chat queries accepting question and optional selected text
- **FR-002**: System MUST retrieve relevant book chunks using vector similarity search
- **FR-003**: System MUST generate answers using retrieved context only (no hallucination)
- **FR-004**: System MUST include source references in every response
- **FR-005**: System MUST provide an endpoint to index/re-index book content
- **FR-006**: System MUST chunk book content appropriately for vector storage
- **FR-007**: System MUST provide a health check endpoint reporting all dependency statuses
- **FR-008**: System MUST implement rate limiting per client
- **FR-009**: System MUST validate all input and return appropriate error responses
- **FR-010**: System MUST log all requests with timing information
- **FR-011**: Selected-text queries MUST bypass vector search and use only provided text

### Non-Functional Requirements

- **NFR-001**: API response time MUST be under 5 seconds for 90% of queries
- **NFR-002**: System MUST handle at least 100 concurrent requests
- **NFR-003**: System MUST be stateless (no server-side session storage required)
- **NFR-004**: All secrets MUST be loaded from environment variables

### API Endpoints (Conceptual)

- **Chat Query**: Accept question, optional selected text, return answer with sources
- **Index Content**: Accept book content, store vectors, return status
- **Health Check**: Return status of all dependencies
- **Indexing Status**: Return progress of ongoing indexing operation

### Key Entities

- **ChatRequest**: Question text, selected text (optional), conversation ID (optional)
- **ChatResponse**: Answer text, sources list, confidence score, processing time
- **Source**: Chapter ID, section title, relevance score, text snippet
- **IndexingJob**: Job ID, status, progress percentage, started/completed timestamps
- **HealthStatus**: Overall status, individual service statuses (qdrant, database, llm)

## Assumptions

- Qdrant Cloud is accessible via HTTPS with API key authentication
- Neon Postgres connection is available via connection string
- OpenAI API (or compatible) is available for embeddings and completions
- Book content is provided as markdown files
- Deployment target supports Python 3.11+
- Rate limiting uses IP-based identification (no user authentication required)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: API returns valid JSON responses for all endpoints
- **SC-002**: Chat queries return answers with sources in under 5 seconds (p90)
- **SC-003**: Indexing 100 pages of content completes in under 5 minutes
- **SC-004**: Health endpoint accurately reports dependency statuses
- **SC-005**: Rate limiting blocks requests exceeding 60 per minute per IP
- **SC-006**: All endpoints return appropriate HTTP status codes (200, 400, 429, 500, 503)
- **SC-007**: No hardcoded secrets in codebase (verified by code review)
