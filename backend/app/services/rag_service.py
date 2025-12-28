"""RAG (Retrieval-Augmented Generation) service for chat functionality."""

import logging
from dataclasses import dataclass
from typing import List, Optional

from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Single chunk retrieved from vector search."""

    chunk_id: str
    chapter_id: str
    section_title: str
    content: str
    relevance_score: float


@dataclass
class RAGContext:
    """Context for answer generation."""

    question: str
    chunks: List[RetrievedChunk]
    mode: str  # "book_wide" or "selected_text"
    selected_text: Optional[str] = None


class RAGService:
    """Service orchestrating RAG pipeline for question answering."""

    # Minimum relevance score for including chunks
    MIN_RELEVANCE_SCORE = 0.5
    # Maximum chunks to retrieve
    MAX_CHUNKS = 5

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_service: VectorService,
        llm_service: LLMService,
    ):
        """Initialize RAG service.

        Args:
            embedding_service: Service for generating embeddings
            vector_service: Service for vector similarity search
            llm_service: Service for answer generation
        """
        self.embedding_service = embedding_service
        self.vector_service = vector_service
        self.llm_service = llm_service

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        """Retrieve relevant chunks for a question.

        Args:
            question: User's question

        Returns:
            List of relevant chunks sorted by relevance
        """
        # Generate embedding for the question (optimized for retrieval)
        query_embedding = self.embedding_service.embed_query(question)

        # Search for similar chunks
        results = self.vector_service.search(
            query_vector=query_embedding,
            limit=self.MAX_CHUNKS,
        )

        # Filter by minimum relevance and convert to dataclass
        chunks = []
        for result in results:
            if result["relevance_score"] >= self.MIN_RELEVANCE_SCORE:
                chunk = RetrievedChunk(
                    chunk_id=result["chunk_id"],
                    chapter_id=result["chapter_id"],
                    section_title=result["section_title"],
                    content=result["content"],
                    relevance_score=result["relevance_score"],
                )
                chunks.append(chunk)

        logger.info(f"Retrieved {len(chunks)} relevant chunks for question")
        return chunks

    def generate(self, context: RAGContext) -> str:
        """Generate answer from context.

        Args:
            context: RAG context with question and chunks/selected text

        Returns:
            Generated answer string
        """
        if context.mode == "selected_text" and context.selected_text:
            # Selected-text mode: use provided text directly
            return self.llm_service.generate_answer(
                question=context.question,
                context_chunks=[],
                selected_text=context.selected_text,
            )
        else:
            # Book-wide mode: use retrieved chunks
            context_texts = [
                f"[{chunk.chapter_id} - {chunk.section_title}]\n{chunk.content}"
                for chunk in context.chunks
            ]
            return self.llm_service.generate_answer(
                question=context.question,
                context_chunks=context_texts,
                selected_text=None,
            )

    def answer(
        self,
        question: str,
        selected_text: Optional[str] = None,
    ) -> tuple[str, List[RetrievedChunk]]:
        """Full RAG pipeline: retrieve and generate answer.

        Args:
            question: User's question
            selected_text: Optional selected text (bypasses retrieval)

        Returns:
            Tuple of (answer, sources)
        """
        if selected_text:
            # Selected-text mode: bypass retrieval
            context = RAGContext(
                question=question,
                chunks=[],
                mode="selected_text",
                selected_text=selected_text,
            )
            answer = self.generate(context)
            return answer, []
        else:
            # Book-wide mode: retrieve and generate
            chunks = self.retrieve(question)
            context = RAGContext(
                question=question,
                chunks=chunks,
                mode="book_wide",
            )
            answer = self.generate(context)
            return answer, chunks
