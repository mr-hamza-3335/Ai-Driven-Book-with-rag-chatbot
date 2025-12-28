---
id: 001
title: Project Constitution Creation
stage: constitution
date: 2025-12-27
surface: agent
model: claude-opus-4-5-20251101
feature: none
branch: master
user: user
command: /sp.constitution
labels: ["constitution", "project-setup", "principles"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - N/A (constitution document, no tests)
---

## Prompt

Project: AI-Driven Book with Embedded RAG Chatbot

Core Principles:
1. Spec-Driven Development - All features flow from specifications to implementation
2. Zero Hallucination - RAG must rely strictly on retrieved content, no fabricated answers
3. Context-Only Answers - Selected-text queries must ignore all other text
4. Clean Architecture - Reproducible, well-documented, maintainable code
5. Secure by Default - No hardcoded secrets, proper .env handling
6. Small Testable Changes - Minimal diffs, incremental progress

Tech Stack:
- Claude Code + Spec-Kit Plus (development workflow)
- Docusaurus v2+ (documentation/book framework)
- OpenAI Agents or ChatKit SDK (chatbot intelligence)
- FastAPI (backend API)
- Neon Serverless Postgres (database)
- Qdrant Cloud Free Tier (vector store for RAG)

Success Criteria:
- Book live on GitHub Pages
- Chatbot fully functional and accurate
- Repository is reproducible and documented

## Response snapshot

Created project constitution v1.0.0 with 6 core principles:
- I. Spec-Driven Development
- II. Zero Hallucination
- III. Context-Only Answers
- IV. Clean Architecture
- V. Secure by Default
- VI. Small Testable Changes

Added Tech Stack section mandating: Docusaurus, FastAPI, Neon Postgres, Qdrant Cloud, OpenAI/ChatKit.
Added Success Criteria section with 7 measurable outcomes (SC-001 through SC-007).
Added Governance section with amendment process, versioning policy, and compliance requirements.

## Outcome

- Impact: Foundation document created for entire project
- Tests: N/A (constitution document)
- Files: 1 created (.specify/memory/constitution.md)
- Next prompts: /sp.specify for docusaurus-book, rag-chatbot, backend-api features
- Reflection: Constitution establishes clear boundaries for RAG behavior (zero hallucination, context-only)

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
