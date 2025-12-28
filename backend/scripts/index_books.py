"""Script to index all book chapters into Qdrant."""

import os
import re
import json
import requests
from pathlib import Path

# Configuration
API_URL = "http://localhost:8001/api/index"
BOOKS_DIR = Path(__file__).parent.parent.parent / "books"

def extract_title(content: str) -> str:
    """Extract title from markdown content."""
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    return match.group(1) if match else "Untitled"

def extract_sections(content: str) -> list:
    """Extract section titles from markdown content."""
    sections = re.findall(r'^##\s+(.+)$', content, re.MULTILINE)
    return sections

def load_chapters() -> list:
    """Load all chapter markdown files."""
    chapters = []

    # Find all markdown files
    md_files = sorted(BOOKS_DIR.glob("*.md"))

    for md_file in md_files:
        if md_file.name.startswith("README"):
            continue

        content = md_file.read_text(encoding="utf-8")

        # Extract chapter ID from filename (e.g., "01-introduction" from "01-introduction.md")
        chapter_id = md_file.stem.lower().replace(" ", "-")

        # Clean chapter_id to match pattern ^[a-z0-9-]+$
        chapter_id = re.sub(r'[^a-z0-9-]', '', chapter_id)

        title = extract_title(content)
        sections = extract_sections(content)

        chapters.append({
            "chapter_id": chapter_id,
            "title": title,
            "content": content,
            "sections": sections
        })

        print(f"Loaded: {chapter_id} - {title}")

    return chapters

def index_chapters(chapters: list, force_reindex: bool = True):
    """Send chapters to indexing API."""
    payload = {
        "chapters": chapters,
        "force_reindex": force_reindex
    }

    print(f"\nIndexing {len(chapters)} chapters...")

    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300
        )
        response.raise_for_status()

        result = response.json()
        print(f"\nIndexing result:")
        print(f"  Job ID: {result['job_id']}")
        print(f"  Status: {result['status']}")
        print(f"  Chapters indexed: {result['chapters_indexed']}")
        print(f"  Chunks created: {result['chunks_created']}")

        return result

    except requests.exceptions.RequestException as e:
        print(f"Error indexing chapters: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None

def main():
    print("=" * 60)
    print("Book Indexing Script")
    print("=" * 60)

    # Check if books directory exists
    if not BOOKS_DIR.exists():
        print(f"Error: Books directory not found: {BOOKS_DIR}")
        return

    # Load chapters
    chapters = load_chapters()

    if not chapters:
        print("No chapters found to index.")
        return

    print(f"\nFound {len(chapters)} chapters")

    # Index chapters
    result = index_chapters(chapters, force_reindex=True)

    if result and result.get("status") == "completed":
        print("\n[OK] Indexing completed successfully!")
    else:
        print("\n[WARN] Indexing may have issues. Check the API logs.")

if __name__ == "__main__":
    main()
