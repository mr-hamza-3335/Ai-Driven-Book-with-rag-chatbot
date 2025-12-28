# Data Model: AI/ML Fundamentals Book

**Feature**: 001-docusaurus-book
**Date**: 2025-12-27

## Overview

This feature uses a file-based content model (Markdown/MDX files) with client-side state management for reading progress. No database is required.

## Entities

### Chapter

Represented as a directory in `docs/` containing markdown files.

```typescript
interface Chapter {
  id: string;           // e.g., "chapter-1-intro-to-ai"
  title: string;        // e.g., "Introduction to AI & Machine Learning"
  order: number;        // 1-7 for ordering
  sections: Section[];  // Subsections within the chapter
  estimatedReadingTime: number; // Minutes, calculated from word count
}
```

**File Structure**:
```
docs/chapter-1-intro-to-ai/
├── index.md           # Main chapter content
├── _category_.json    # Docusaurus category config
└── section-*.md       # Optional additional sections
```

**_category_.json Example**:
```json
{
  "label": "Chapter 1: Introduction to AI & ML",
  "position": 1,
  "collapsible": true,
  "collapsed": false
}
```

### Section

Represented as headings within chapter markdown or separate markdown files.

```typescript
interface Section {
  id: string;           // Anchor ID, e.g., "what-is-machine-learning"
  title: string;        // e.g., "What is Machine Learning?"
  parentChapterId: string;
  anchorLink: string;   // e.g., "#what-is-machine-learning"
}
```

### Code Example

Embedded within markdown files using fenced code blocks.

```typescript
interface CodeExample {
  language: "python" | "bash" | "json";
  code: string;
  description?: string;  // Optional caption
  chapterReference: string;
}
```

**Markdown Representation**:
````markdown
```python title="Linear Regression Example"
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
````

### Reading Progress (Client-Side)

Stored in browser localStorage.

```typescript
interface ReadingProgress {
  chapterId: string;
  completionStatus: "not_started" | "in_progress" | "completed";
  lastReadTimestamp: number;  // Unix timestamp
  scrollPosition?: number;    // Optional scroll position
}

// localStorage key: "book_reading_progress"
// localStorage value: Record<string, ReadingProgress>
```

**localStorage Example**:
```json
{
  "chapter-1-intro-to-ai": {
    "chapterId": "chapter-1-intro-to-ai",
    "completionStatus": "completed",
    "lastReadTimestamp": 1703702400000
  },
  "chapter-2-data": {
    "chapterId": "chapter-2-data",
    "completionStatus": "in_progress",
    "lastReadTimestamp": 1703705000000,
    "scrollPosition": 450
  }
}
```

## State Transitions

### Reading Progress State Machine

```
not_started → in_progress (user opens chapter)
in_progress → in_progress (user scrolls, updates timestamp)
in_progress → completed (user reaches end of chapter OR clicks "Mark Complete")
completed → in_progress (user reopens and scrolls)
```

## Validation Rules

### Chapter Content
- Title: Required, 5-100 characters
- Content: Required, minimum 500 words per chapter
- Order: Required, unique integer 1-7

### Code Examples
- Language: Must be one of: python, bash, json
- Code: Required, non-empty
- Maximum 100 lines per example for readability

### Reading Progress
- chapterId: Must match existing chapter
- completionStatus: Must be valid enum value
- lastReadTimestamp: Must be valid Unix timestamp

## Relationships

```
Book (1) ────────< Chapter (1..7)
                       │
                       ├──────< Section (0..n)
                       │
                       └──────< CodeExample (0..n)

User Session (1) ────────< ReadingProgress (0..7)
```

## Notes

- No server-side persistence required for this feature
- Reading progress is per-browser, not synced across devices
- Content is statically generated at build time
- Search index is generated from markdown content during build
