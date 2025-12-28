<!--
  Sync Impact Report
  ===================
  Version change: null → 1.0.0 (Initial adoption)

  Added principles:
  - I. Spec-Driven Development
  - II. Zero Hallucination
  - III. Context-Only Answers
  - IV. Clean Architecture
  - V. Secure by Default
  - VI. Small Testable Changes

  Added sections:
  - Tech Stack
  - Success Criteria
  - Governance

  Removed sections: None (initial version)

  Templates status:
  - .specify/templates/spec-template.md: ✅ Compatible (generic, uses FR- requirements)
  - .specify/templates/plan-template.md: ✅ Compatible (Constitution Check section present)
  - .specify/templates/tasks-template.md: ✅ Compatible (user story-driven structure)
  - .specify/templates/phr-template.prompt.md: ✅ Compatible (generic PHR format)

  Deferred items: None
-->

# AI-Driven Book with Embedded RAG Chatbot Constitution

## Core Principles

### I. Spec-Driven Development

All features MUST flow from specifications to implementation following the SDD workflow:
specification → plan → tasks → implementation.

- Every feature begins with a formal specification document
- Implementation MUST NOT proceed without an approved spec
- Changes to scope require spec amendments before code changes
- Rationale: Ensures alignment between requirements and implementation, reduces rework

### II. Zero Hallucination

The RAG system MUST rely strictly on retrieved content. No fabricated or inferred answers are permitted.

- Responses MUST be grounded in retrieved document chunks
- If no relevant content is found, the system MUST respond with "I don't have information about that in the book"
- Confidence scores MUST accompany answers when available
- The system MUST NOT extrapolate beyond retrieved context
- Rationale: Maintains user trust and prevents misinformation

### III. Context-Only Answers

Selected-text queries MUST ignore all other text in the book.

- When a user selects specific text, the RAG context MUST be limited to that selection only
- Book-wide search MUST be explicitly separate from selected-text mode
- UI MUST clearly indicate which mode is active
- Cross-reference suggestions are permitted but MUST be labeled as "related content"
- Rationale: Enables precise, focused answers when users have specific passages in mind

### IV. Clean Architecture

All code MUST be reproducible, well-documented, and maintainable.

- Repository MUST include complete setup instructions that work from a fresh clone
- All dependencies MUST be pinned to specific versions
- Code MUST follow consistent formatting (enforced by linters/formatters)
- Functions and modules MUST have clear, single responsibilities
- Rationale: Enables collaboration, reduces onboarding time, prevents "works on my machine" issues

### V. Secure by Default

No hardcoded secrets or tokens. All sensitive configuration MUST use environment variables.

- API keys, database credentials, and tokens MUST be stored in `.env` files
- `.env` files MUST be in `.gitignore`
- A `.env.example` file MUST document required environment variables
- Production secrets MUST use secure secret management (GitHub Secrets, etc.)
- Rationale: Prevents accidental credential exposure and security breaches

### VI. Small Testable Changes

Implementation MUST proceed via minimal diffs with incremental progress.

- Each commit SHOULD represent a single logical change
- Large features MUST be broken into smaller, independently testable increments
- Refactoring MUST be separate from feature changes
- Each change MUST be verifiable before proceeding to the next
- Rationale: Reduces risk, simplifies debugging, enables easier code review

## Tech Stack

The following technologies are mandated for this project:

| Layer | Technology | Purpose |
|-------|------------|---------|
| Development Workflow | Claude Code + Spec-Kit Plus | AI-assisted spec-driven development |
| Documentation/Book | Docusaurus v2+ | Static site generation for the book |
| Chatbot Intelligence | OpenAI Agents or ChatKit SDK | LLM-powered question answering |
| Backend API | FastAPI | REST API for chatbot and RAG |
| Database | Neon Serverless Postgres | Metadata and user session storage |
| Vector Store | Qdrant Cloud (Free Tier) | Embedding storage for RAG retrieval |
| Deployment | GitHub Pages | Static hosting for the book |

Technology substitutions require an ADR documenting the rationale and impact.

## Success Criteria

The project is complete when all of the following are achieved:

- **SC-001**: Book is live and accessible on GitHub Pages
- **SC-002**: Chatbot answers book-wide questions accurately using RAG
- **SC-003**: Chatbot answers selected-text questions using only the selected context
- **SC-004**: Zero hallucination verified through manual testing (10+ queries)
- **SC-005**: Repository can be cloned and deployed by following README instructions
- **SC-006**: All secrets are externalized via environment variables
- **SC-007**: Documentation covers setup, usage, and architecture

## Governance

This constitution is the authoritative source for project principles and standards.

**Amendment Process**:
1. Propose amendment via PR with rationale
2. Document impact on existing artifacts
3. Update version according to semantic versioning
4. Migrate affected templates and code

**Versioning Policy**:
- MAJOR: Backward-incompatible principle changes or removals
- MINOR: New principles or sections added
- PATCH: Clarifications and wording improvements

**Compliance**:
- All PRs MUST pass Constitution Check in plan.md
- Violations require documented justification in Complexity Tracking
- Runtime guidance is in CLAUDE.md

**Version**: 1.0.0 | **Ratified**: 2025-12-27 | **Last Amended**: 2025-12-27
