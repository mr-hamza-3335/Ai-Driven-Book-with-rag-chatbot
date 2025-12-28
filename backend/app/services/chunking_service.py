"""Text chunking service for markdown content."""

import logging
import re
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class ChunkingService:
    """Service for chunking markdown text for vector storage."""

    # Chunk size in characters (approximately 500 tokens)
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 200

    def __init__(self):
        """Initialize the chunking service."""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=[
                "\n## ",  # H2 headers
                "\n### ",  # H3 headers
                "\n#### ",  # H4 headers
                "\n\n",  # Paragraphs
                "\n",  # Lines
                ". ",  # Sentences
                " ",  # Words
            ],
        )

    def chunk_chapter(
        self,
        chapter_id: str,
        title: str,
        content: str,
    ) -> List[Dict[str, Any]]:
        """Split a chapter into chunks.

        Args:
            chapter_id: Unique chapter identifier
            title: Chapter title
            content: Markdown content

        Returns:
            List of chunk dictionaries
        """
        # Extract section titles from markdown headers
        sections = self._extract_sections(content)

        # Split content into chunks
        texts = self.splitter.split_text(content)

        chunks = []
        for i, text in enumerate(texts):
            # Find which section this chunk belongs to
            section_title = self._find_section_for_chunk(text, sections) or title

            chunk = {
                "chunk_id": f"{chapter_id}_chunk_{i:03d}",
                "chapter_id": chapter_id,
                "section_title": section_title,
                "content": text.strip(),
                "chunk_index": i,
            }
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks for chapter: {chapter_id}")
        return chunks

    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract section headers from markdown.

        Args:
            content: Markdown content

        Returns:
            List of section info with title and position
        """
        sections = []
        header_pattern = r"^(#{2,4})\s+(.+)$"

        for match in re.finditer(header_pattern, content, re.MULTILINE):
            sections.append({
                "level": len(match.group(1)),
                "title": match.group(2).strip(),
                "position": match.start(),
            })

        return sections

    def _find_section_for_chunk(
        self,
        chunk_text: str,
        sections: List[Dict[str, Any]],
    ) -> str:
        """Find the section title for a chunk.

        Args:
            chunk_text: Text content of the chunk
            sections: List of section info

        Returns:
            Section title or empty string
        """
        # Check if chunk contains a header
        for section in sections:
            if section["title"] in chunk_text:
                return section["title"]

        return ""

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Input text

        Returns:
            Estimated token count (rough approximation)
        """
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4
