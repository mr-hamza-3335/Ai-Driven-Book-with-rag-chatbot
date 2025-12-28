# Tasks: Backend API

**Input**: Design documents from `/specs/003-backend-api/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are NOT explicitly requested in the feature specification. Test tasks are omitted. Add tests via `/sp.tasks --tdd` if needed.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Per plan.md, this is a single Python project with layered architecture:
- Source: `backend/app/`
- Tests: `backend/tests/`
- Scripts: `backend/scripts/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project directory structure per plan.md in backend/
- [x] T002 Create requirements.txt with pinned dependencies in backend/requirements.txt
- [x] T003 [P] Create requirements-dev.txt with dev dependencies in backend/requirements-dev.txt
- [x] T004 [P] Create .env.example with environment template in backend/.env.example
- [x] T005 [P] Create Dockerfile for container build in backend/Dockerfile
- [x] T006 [P] Create docker-compose.yml for local dev in backend/docker-compose.yml
- [x] T007 Create app/__init__.py with version info in backend/app/__init__.py
- [x] T008 Create config.py with environment configuration (Pydantic Settings) in backend/app/config.py
- [x] T009 [P] Create pytest configuration conftest.py in backend/tests/conftest.py

**Checkpoint**: Project structure ready, dependencies installed, Docker working

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**WARNING**: No user story work can begin until this phase is complete

- [x] T010 Create FastAPI app entry point with lifespan in backend/app/main.py
- [x] T011 Create database.py with SQLAlchemy engine and Base in backend/app/database.py
- [x] T012 [P] Create CORS middleware configuration in backend/app/middleware/cors.py
- [x] T013 [P] Create middleware __init__.py in backend/app/middleware/__init__.py
- [x] T014 [P] Create api/__init__.py in backend/app/api/__init__.py
- [x] T015 [P] Create routes/__init__.py in backend/app/api/routes/__init__.py
- [x] T016 [P] Create models/__init__.py in backend/app/models/__init__.py
- [x] T017 [P] Create services/__init__.py in backend/app/services/__init__.py
- [x] T018 [P] Create repositories/__init__.py in backend/app/repositories/__init__.py
- [x] T019 Create dependencies.py with dependency injection in backend/app/api/dependencies.py
- [x] T020 Create embedding_service.py with OpenAI embeddings in backend/app/services/embedding_service.py
- [x] T021 Create vector_service.py with Qdrant client operations in backend/app/services/vector_service.py
- [x] T022 Create llm_service.py with OpenAI completions in backend/app/services/llm_service.py
- [x] T023 Create chunking_service.py with markdown text splitter in backend/app/services/chunking_service.py

**Checkpoint**: Foundation ready - all core services available, user story implementation can now begin

---

## Phase 3: User Story 1 - Process Chat Queries (Priority: P1) MVP

**Goal**: Frontend chatbot sends user questions to backend, retrieves relevant content, generates answers

**Independent Test**: Send POST request with question, verify response contains answer and sources, check response time under 5 seconds

### Implementation for User Story 1

- [x] T024 [P] [US1] Create ChatRequest Pydantic model in backend/app/models/chat.py
- [x] T025 [P] [US1] Create ChatResponse Pydantic model in backend/app/models/chat.py
- [x] T026 [P] [US1] Create Source Pydantic model in backend/app/models/chat.py
- [x] T027 [P] [US1] Create ChatError Pydantic model in backend/app/models/chat.py
- [x] T028 [US1] Create RetrievedChunk and RAGContext dataclasses in backend/app/services/rag_service.py
- [x] T029 [US1] Implement RAGService with retrieve() method for vector search in backend/app/services/rag_service.py
- [x] T030 [US1] Implement RAGService generate() method with zero-hallucination prompts in backend/app/services/rag_service.py
- [x] T031 [US1] Implement RAGService answer() orchestration method in backend/app/services/rag_service.py
- [x] T032 [US1] Create POST /api/chat endpoint in backend/app/api/routes/chat.py
- [x] T033 [US1] Add chat router to main app in backend/app/main.py
- [x] T034 [US1] Add request timing and logging for chat endpoint in backend/app/api/routes/chat.py

**Checkpoint**: User Story 1 complete - chat queries work with book-wide RAG retrieval

---

## Phase 4: User Story 2 - Index Book Content (Priority: P1)

**Goal**: Administrator indexes/re-indexes book content into vector store when content changes

**Independent Test**: Run indexing endpoint with book content, verify vectors stored in Qdrant, verify search returns results

### Implementation for User Story 2

- [x] T035 [P] [US2] Create IndexStatus enum in backend/app/models/index.py
- [x] T036 [P] [US2] Create ChapterContent Pydantic model in backend/app/models/index.py
- [x] T037 [P] [US2] Create IndexRequest Pydantic model in backend/app/models/index.py
- [x] T038 [P] [US2] Create IndexResponse Pydantic model in backend/app/models/index.py
- [x] T039 [P] [US2] Create IndexStatusResponse Pydantic model in backend/app/models/index.py
- [x] T040 [US2] Create IndexingService with chunk_and_embed() method in backend/app/services/indexing_service.py
- [x] T041 [US2] Implement IndexingService index_chapters() method in backend/app/services/indexing_service.py
- [x] T042 [US2] Implement IndexingService get_status() method for progress tracking in backend/app/services/indexing_service.py
- [x] T043 [US2] Create POST /api/index endpoint in backend/app/api/routes/index.py
- [x] T044 [US2] Create GET /api/index/{job_id} endpoint for status in backend/app/api/routes/index.py
- [x] T045 [US2] Add index router to main app in backend/app/main.py
- [x] T046 [P] [US2] Create CLI script for book indexing in backend/scripts/index_book.py

**Checkpoint**: User Story 2 complete - book content can be indexed and re-indexed via API

---

## Phase 5: User Story 3 - Selected-Text Queries (Priority: P1)

**Goal**: Frontend sends selected text with question, backend answers based only on that text (bypasses RAG)

**Independent Test**: Send selected text with question, verify answer references only selection, not other book content

### Implementation for User Story 3

- [x] T047 [US3] Extend RAGService with selected-text mode (bypass vector search) in backend/app/services/rag_service.py
- [x] T048 [US3] Add selected_text validation (min 10 chars) to ChatRequest in backend/app/models/chat.py
- [x] T049 [US3] Update POST /api/chat to handle selected_text mode in backend/app/api/routes/chat.py
- [x] T050 [US3] Add context-only system prompt for selected-text mode in backend/app/services/rag_service.py

**Checkpoint**: User Story 3 complete - selected-text queries work independently from book-wide queries

---

## Phase 6: User Story 4 - Health Check (Priority: P2)

**Goal**: Operations monitor backend health and dependencies (Qdrant, Neon, LLM service)

**Independent Test**: Call /health endpoint, verify all dependency statuses returned

### Implementation for User Story 4

- [x] T051 [P] [US4] Create ServiceStatus Pydantic model in backend/app/models/health.py
- [x] T052 [P] [US4] Create HealthResponse Pydantic model in backend/app/models/health.py
- [x] T053 [US4] Create HealthService with check_qdrant() method in backend/app/services/health_service.py
- [x] T054 [US4] Add check_database() method to HealthService in backend/app/services/health_service.py
- [x] T055 [US4] Add check_openai() method to HealthService in backend/app/services/health_service.py
- [x] T056 [US4] Create GET /health endpoint in backend/app/api/routes/health.py
- [x] T057 [US4] Add health router to main app in backend/app/main.py

**Checkpoint**: User Story 4 complete - health endpoint reports all dependency statuses

---

## Phase 7: User Story 5 - Rate Limiting (Priority: P2)

**Goal**: System protects against abuse by limiting requests per user/IP (60 req/min)

**Independent Test**: Send more requests than limit, verify 429 response after limit reached

### Implementation for User Story 5

- [x] T058 [P] [US5] Create RateLimit SQLAlchemy model in backend/app/repositories/rate_limit_repo.py
- [x] T059 [US5] Create RateLimitRepository with check_and_increment() method in backend/app/repositories/rate_limit_repo.py
- [x] T060 [US5] Create rate_limit middleware in backend/app/middleware/rate_limit.py
- [x] T061 [US5] Add rate_limit middleware to main app in backend/app/main.py
- [x] T062 [US5] Add retry-after header to 429 responses in backend/app/middleware/rate_limit.py
- [x] T063 [US5] Create database migration for rate_limits table in backend/app/database.py

**Checkpoint**: User Story 5 complete - rate limiting enforced on all endpoints

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T064 [P] Create README.md with setup instructions in backend/README.md
- [x] T065 Add structured logging configuration to all endpoints in backend/app/main.py
- [x] T066 Add error handling for external service failures (503 for Qdrant, 504 for LLM timeout) in backend/app/api/routes/chat.py
- [x] T067 [P] Run quickstart.md verification checklist manually
- [x] T068 Validate all endpoints return appropriate HTTP status codes (200, 400, 429, 500, 503)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phases 3-7)**: All depend on Foundational phase completion
  - User stories can proceed in parallel (if team capacity allows)
  - Or sequentially in priority order (P1: US1, US2, US3 → P2: US4, US5)
- **Polish (Phase 8)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 3 (P1)**: Can start after Foundational (Phase 2) - Extends US1's RAGService but independently testable
- **User Story 4 (P2)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 5 (P2)**: Can start after Foundational (Phase 2) - No dependencies on other stories

### Within Each User Story

- Models before services
- Services before endpoints/routes
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- Setup: T003, T004, T005, T006 can run in parallel
- Foundational: T012-T018 can run in parallel
- US1: T024-T027 (models) can run in parallel
- US2: T035-T039 (models) can run in parallel, T046 independent
- US4: T051-T052 can run in parallel
- US5: T058 can run in parallel with other models
- Different user stories can be worked on by different team members after Foundational

---

## Parallel Example: User Story 1 Models

```bash
# Launch all models for User Story 1 together:
Task: "Create ChatRequest Pydantic model in backend/app/models/chat.py"
Task: "Create ChatResponse Pydantic model in backend/app/models/chat.py"
Task: "Create Source Pydantic model in backend/app/models/chat.py"
Task: "Create ChatError Pydantic model in backend/app/models/chat.py"
```

---

## Parallel Example: User Story 2 Models

```bash
# Launch all models for User Story 2 together:
Task: "Create IndexStatus enum in backend/app/models/index.py"
Task: "Create ChapterContent Pydantic model in backend/app/models/index.py"
Task: "Create IndexRequest Pydantic model in backend/app/models/index.py"
Task: "Create IndexResponse Pydantic model in backend/app/models/index.py"
Task: "Create IndexStatusResponse Pydantic model in backend/app/models/index.py"
```

---

## Implementation Strategy

### MVP First (User Stories 1, 2, 3)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Chat Queries)
4. **STOP and VALIDATE**: Test chat endpoint works with book retrieval
5. Complete Phase 4: User Story 2 (Indexing)
6. **STOP and VALIDATE**: Index content, then test chat returns indexed results
7. Complete Phase 5: User Story 3 (Selected-Text)
8. **STOP and VALIDATE**: Test selected-text queries work independently
9. Deploy MVP

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Core RAG working (MVP 1)
3. Add User Story 2 → Test independently → Content indexing works (MVP 2)
4. Add User Story 3 → Test independently → Selected-text works (Full MVP)
5. Add User Story 4 → Test independently → Health monitoring (Production ready)
6. Add User Story 5 → Test independently → Rate limiting (Production hardened)
7. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers after Foundational phase:
- Developer A: User Story 1 (Chat)
- Developer B: User Story 2 (Indexing)
- Developer C: User Story 3 (Selected-Text) + User Story 4 (Health)
- Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Selected-text mode (US3) extends RAGService from US1 but tests independently

---

## Summary

| Metric | Count |
|--------|-------|
| Total Tasks | 68 |
| Phase 1 (Setup) | 9 |
| Phase 2 (Foundational) | 14 |
| User Story 1 (Chat) | 11 |
| User Story 2 (Index) | 12 |
| User Story 3 (Selected-Text) | 4 |
| User Story 4 (Health) | 7 |
| User Story 5 (Rate Limiting) | 6 |
| Phase 8 (Polish) | 5 |
| Parallel Opportunities | 28 tasks marked [P] |
| MVP Scope | Setup + Foundational + US1 + US2 + US3 (50 tasks) |
