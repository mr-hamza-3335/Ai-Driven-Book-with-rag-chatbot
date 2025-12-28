# Research: AI/ML Fundamentals Book

**Feature**: 001-docusaurus-book
**Date**: 2025-12-27
**Status**: Complete

## Research Tasks

### 1. Docusaurus Version Selection

**Decision**: Docusaurus 3.x (latest stable)

**Rationale**:
- Docusaurus 3.x is the current stable release with long-term support
- Built on React 18 with improved performance
- Better TypeScript support out of the box
- MDX 3 support for enhanced markdown capabilities
- Algolia DocSearch integration for search functionality

**Alternatives Considered**:
- Docusaurus 2.x: Older, missing latest features
- VitePress: Faster build but less mature ecosystem
- Nextra: Good but smaller community

### 2. GitHub Pages Deployment Strategy

**Decision**: GitHub Actions with `peaceiris/actions-gh-pages`

**Rationale**:
- Native GitHub integration, no external services
- Free for public repositories
- Automatic deployment on push to main branch
- Well-documented and widely used

**Alternatives Considered**:
- Netlify: More features but adds external dependency
- Vercel: Excellent but optimized for Next.js
- Manual deployment: Error-prone and not automated

### 3. Search Implementation

**Decision**: Local search with @easyops-cn/docusaurus-search-local

**Rationale**:
- No external service required (Algolia requires application/approval)
- Works immediately without configuration
- Free and open source
- Meets < 1 second search requirement

**Alternatives Considered**:
- Algolia DocSearch: Better results but requires application process
- Lunr.js custom: More work to implement
- No search: Would fail FR-004 requirement

### 4. Code Highlighting

**Decision**: Prism.js (Docusaurus default) with copy button plugin

**Rationale**:
- Built into Docusaurus, zero configuration needed
- Supports Python syntax highlighting
- Copy button available via theme configuration
- Meets FR-005 and FR-006 requirements

**Alternatives Considered**:
- Shiki: Better accuracy but heavier bundle
- Highlight.js: Alternative but Prism is already included

### 5. Reading Progress Tracking

**Decision**: Custom React component with localStorage

**Rationale**:
- Client-side only, no backend required
- Persists across browser sessions
- Simple implementation using React hooks
- Meets FR-009 requirement

**Alternatives Considered**:
- IndexedDB: Overkill for simple progress data
- Cookies: Size limitations and privacy concerns
- Server-side: Adds unnecessary complexity

### 6. Chatbot Widget Integration

**Decision**: Placeholder component with API contract

**Rationale**:
- Book feature provides the integration point
- Actual chatbot implementation is separate feature (002-rag-chatbot)
- Component accepts backend URL via environment variable
- Meets FR-008 by providing the widget container

**Alternatives Considered**:
- Inline iframe: Less flexible for customization
- Web component: More complex setup
- Third-party widget: Doesn't meet zero-hallucination requirement

### 7. Mobile Responsiveness

**Decision**: Use Docusaurus default responsive theme with minor customizations

**Rationale**:
- Docusaurus themes are mobile-first by default
- Infima CSS framework handles responsive breakpoints
- Custom CSS only for specific adjustments
- Meets FR-007 requirement (320px+)

**Alternatives Considered**:
- Custom theme: Unnecessary work
- Tailwind CSS: Would replace Infima, more work

## Dependencies Summary

```json
{
  "@docusaurus/core": "^3.6.0",
  "@docusaurus/preset-classic": "^3.6.0",
  "@easyops-cn/docusaurus-search-local": "^0.44.0",
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "typescript": "^5.3.0"
}
```

## Resolved Clarifications

All technical decisions have been made. No outstanding NEEDS CLARIFICATION items remain.
