# Quickstart: AI/ML Fundamentals Book

**Feature**: 001-docusaurus-book
**Date**: 2025-12-27

## Prerequisites

- Node.js 18.0 or higher
- npm 9.0 or higher (comes with Node.js)
- Git

## Setup

### 1. Clone and Navigate

```bash
git clone <repository-url>
cd hackathonnnnn
```

### 2. Install Dependencies

```bash
cd book
npm install
```

### 3. Start Development Server

```bash
npm start
```

The book will be available at `http://localhost:3000`.

## Project Commands

| Command | Description |
|---------|-------------|
| `npm start` | Start development server with hot reload |
| `npm run build` | Build production static files |
| `npm run serve` | Serve production build locally |
| `npm run clear` | Clear Docusaurus cache |

## Adding Content

### Add a New Chapter

1. Create directory: `docs/chapter-N-topic-name/`
2. Add `index.md` with chapter content
3. Add `_category_.json` for sidebar config:

```json
{
  "label": "Chapter N: Topic Name",
  "position": N,
  "collapsible": true
}
```

### Add Code Examples

Use fenced code blocks with language specifier:

````markdown
```python title="Example Title"
# Your Python code here
print("Hello, ML!")
```
````

### Add Images

1. Place images in `static/img/`
2. Reference in markdown: `![Alt text](/img/your-image.png)`

## Deployment

### GitHub Pages (Automatic)

Push to `main` branch triggers automatic deployment via GitHub Actions.

### Manual Build

```bash
npm run build
# Output in build/ directory
```

## Configuration Files

| File | Purpose |
|------|---------|
| `docusaurus.config.ts` | Main site configuration |
| `sidebars.ts` | Sidebar navigation structure |
| `src/css/custom.css` | Theme customization |

## Environment Variables

For chatbot integration (handled by backend):

```bash
# .env.local (optional, for local development)
REACT_APP_CHATBOT_API_URL=http://localhost:8000
```

## Verification Checklist

- [ ] `npm start` runs without errors
- [ ] All chapters display in sidebar
- [ ] Search returns results
- [ ] Code blocks have syntax highlighting
- [ ] Copy button works on code blocks
- [ ] Site is responsive on mobile (320px width)
- [ ] `npm run build` completes successfully

## Troubleshooting

### Cache Issues
```bash
npm run clear
npm start
```

### Dependency Issues
```bash
rm -rf node_modules package-lock.json
npm install
```

### Build Errors
Check for:
- Invalid markdown syntax
- Missing frontmatter in MDX files
- Broken internal links
