---
id: 002
title: Analyze Blocked Missing Tasks
stage: misc
date: 2025-12-28
surface: agent
model: claude-opus-4-5-20251101
feature: 003-backend-api
branch: 003-backend-api
user: Mak Tech
command: /sp.analyze
labels: [analysis, blocked, prerequisite]
links:
  spec: specs/003-backend-api/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - null
tests:
  - null
---

## Prompt

/sp.analyze

(User invoked the analyze command to perform cross-artifact consistency analysis)

## Response snapshot

Analysis blocked because `tasks.md` does not exist yet in `specs/003-backend-api/`. The `/sp.analyze` command requires all three core artifacts (`spec.md`, `plan.md`, `tasks.md`) to perform cross-artifact consistency analysis.

Recommended next action: Run `/sp.tasks` first to generate the task list, then re-run `/sp.analyze`.

## Outcome

- Impact: None - command blocked due to missing prerequisite
- Tests: N/A
- Files: None modified (read-only analysis blocked)
- Next prompts: /sp.tasks, then /sp.analyze
- Reflection: Prerequisite check working correctly; user needs to complete task generation before analysis

## Evaluation notes (flywheel)

- Failure modes observed: Missing tasks.md file
- Graders run and results (PASS/FAIL): N/A - blocked before analysis
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
