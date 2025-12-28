#!/usr/bin/env python3
"""CLI script for indexing book content into the vector store."""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Index book content into the RAG vector store"
    )
    parser.add_argument(
        "--book-dir",
        type=str,
        required=True,
        help="Path to Docusaurus docs directory",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Backend API URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindex (delete existing chunks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without indexing",
    )
    return parser.parse_args()


def find_markdown_files(book_dir: Path) -> list[Path]:
    """Find all markdown files in the book directory."""
    files = list(book_dir.glob("**/*.md"))
    files.extend(book_dir.glob("**/*.mdx"))
    return sorted(files)


def parse_markdown_file(file_path: Path, book_dir: Path) -> dict:
    """Parse a markdown file into chapter format."""
    content = file_path.read_text(encoding="utf-8")

    # Generate chapter ID from relative path
    rel_path = file_path.relative_to(book_dir)
    chapter_id = str(rel_path.with_suffix("")).replace("/", "-").replace("\\", "-")

    # Extract title from first heading or filename
    title = file_path.stem.replace("-", " ").replace("_", " ").title()
    for line in content.split("\n"):
        if line.startswith("# "):
            title = line[2:].strip()
            break

    return {
        "chapter_id": chapter_id.lower(),
        "title": title,
        "content": content,
    }


def index_content(api_url: str, chapters: list[dict], force: bool) -> dict:
    """Send chapters to the API for indexing."""
    response = httpx.post(
        f"{api_url}/api/index",
        json={
            "chapters": chapters,
            "force_reindex": force,
        },
        timeout=300.0,  # 5 minute timeout for large books
    )
    response.raise_for_status()
    return response.json()


def main():
    """Main entry point."""
    args = parse_args()
    book_dir = Path(args.book_dir)

    if not book_dir.exists():
        print(f"Error: Book directory not found: {book_dir}")
        sys.exit(1)

    # Find markdown files
    files = find_markdown_files(book_dir)
    print(f"Found {len(files)} markdown files in {book_dir}")

    if not files:
        print("No markdown files found. Exiting.")
        sys.exit(0)

    # Parse files into chapters
    chapters = []
    for file_path in files:
        try:
            chapter = parse_markdown_file(file_path, book_dir)
            chapters.append(chapter)
            print(f"  - {chapter['chapter_id']}: {chapter['title']}")
        except Exception as e:
            print(f"  ! Error parsing {file_path}: {e}")

    if args.dry_run:
        print(f"\nDry run complete. Would index {len(chapters)} chapters.")
        print("\nChapter IDs:")
        for ch in chapters:
            print(f"  - {ch['chapter_id']}")
        sys.exit(0)

    # Index content
    print(f"\nIndexing {len(chapters)} chapters to {args.api_url}...")
    try:
        result = index_content(args.api_url, chapters, args.force)
        print(f"\nIndexing complete!")
        print(f"  Job ID: {result['job_id']}")
        print(f"  Status: {result['status']}")
        print(f"  Chapters indexed: {result['chapters_indexed']}")
        print(f"  Chunks created: {result['chunks_created']}")
    except httpx.HTTPError as e:
        print(f"\nError indexing content: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
