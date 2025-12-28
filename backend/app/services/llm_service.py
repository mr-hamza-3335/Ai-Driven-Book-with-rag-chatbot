"""Gemini LLM service for answer generation."""

import logging
from typing import List, Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)


class LLMService:
    """Service for generating answers using Gemini LLM."""

    MODEL = "models/gemini-2.5-flash"

    # Zero-hallucination system prompt
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions about an AI/ML book.
You MUST only use information from the provided context.
If the context does not contain relevant information, respond with exactly:
"I don't have information about that in the book."
Do NOT make up or infer information not explicitly stated in the context.
Always cite the chapter and section for your answer."""

    # Selected-text mode system prompt
    SELECTED_TEXT_PROMPT = """You are a helpful assistant answering questions about specific text selected by the user.
You MUST only use information from the selected text provided.
If the selected text does not contain relevant information to answer the question, respond with:
"The selected text doesn't contain information about that."
Do NOT make up or infer information not in the selected text.
Be concise and directly reference the selected text in your answer."""

    def __init__(self, api_key: str):
        """Initialize the LLM service.

        Args:
            api_key: Gemini API key
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=self.MODEL,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1000,
            ),
        )

    def generate_answer(
        self,
        question: str,
        context_chunks: List[str],
        selected_text: Optional[str] = None,
    ) -> str:
        """Generate an answer based on context.

        Args:
            question: User's question
            context_chunks: List of relevant text chunks from the book
            selected_text: Optional user-selected text (bypasses RAG)

        Returns:
            Generated answer string
        """
        # Choose prompt based on mode
        if selected_text:
            system_prompt = self.SELECTED_TEXT_PROMPT
            context = f"Selected text:\n{selected_text}"
        else:
            system_prompt = self.SYSTEM_PROMPT
            if not context_chunks:
                return "I don't have information about that in the book."
            context = "\n\n---\n\n".join(context_chunks)

        user_message = f"""{system_prompt}

Context:
{context}

Question: {question}

Answer based only on the context provided:"""

        response = self.model.generate_content(user_message)
        return response.text

    def check_health(self) -> bool:
        """Check if Gemini API is accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Make a minimal API call
            models = genai.list_models()
            return any(m.name for m in models)
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            return False
