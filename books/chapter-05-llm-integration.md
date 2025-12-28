# Chapter 5: LLM Integration with Gemini

## Learning Objectives

By the end of this chapter, you will be able to:
- Integrate Google's Gemini API for text generation
- Design prompts that ground responses in retrieved context
- Implement zero-hallucination guardrails
- Handle API rate limits and errors gracefully
- Optimize generation parameters for RAG use cases

---

## 5.1 Introduction to Gemini

Google's Gemini represents the latest advancement in large language models, offering powerful generation capabilities through a straightforward API. For RAG systems, Gemini serves as the generation component, synthesizing retrieved context into coherent, accurate responses.

Gemini comes in several variants optimized for different use cases. Gemini Pro provides a balance of capability and speed suitable for most applications. Gemini Flash prioritizes low latency for real-time interactions. Gemini Ultra offers maximum capability for complex reasoning tasks.

For RAG applications, Gemini Flash (including the 2.5 version) typically offers the best tradeoff. The lower latency improves user experience, while the capability is sufficient for synthesizing retrieved information. The reduced cost per token also matters when processing many requests.

The Gemini API supports both synchronous and streaming responses. Synchronous calls wait for complete generation before returning. Streaming returns tokens incrementally, improving perceived latency for longer responses. RAG systems often use synchronous calls for simplicity, but streaming enhances user experience for interactive applications.

## 5.2 Setting Up the Gemini Client

Integration begins with installing the SDK and configuring authentication.

```bash
pip install google-generativeai
```

Configure the API key:

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
```

Create a model instance with generation parameters:

```python
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=genai.GenerationConfig(
        temperature=0.1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=1024,
    )
)
```

The generation configuration controls output characteristics:
- **temperature**: Lower values (0.1-0.3) produce more focused, deterministic outputs—ideal for RAG where accuracy matters.
- **top_p** and **top_k**: Control token sampling diversity. Conservative values reduce hallucination risk.
- **max_output_tokens**: Limits response length to control costs and focus answers.

## 5.3 Prompt Engineering for RAG

The prompt design is critical for RAG effectiveness. The prompt must instruct the model to use only the provided context, cite sources, and decline questions it cannot answer from the given information.

A well-structured RAG prompt has several components:

```python
SYSTEM_PROMPT = """You are a helpful assistant answering questions about a technical book on RAG systems.

CRITICAL INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. If the context doesn't contain relevant information, say "I don't have information about that in the book."
3. NEVER make up information not in the context
4. Cite the chapter and section for your answers
5. Be concise and direct

Context from the book:
{context}

Question: {question}

Answer based only on the context above:"""
```

Key elements of this prompt:

1. **Role definition**: Establishes the assistant's purpose and domain
2. **Explicit constraints**: Clear instructions about using only context
3. **Fallback behavior**: Specifies what to do when information is missing
4. **Citation requirement**: Encourages source attribution
5. **Structure**: Clear separation of context and question

## 5.4 Implementing the LLM Service

The LLM service encapsulates Gemini interaction logic.

```python
# app/services/llm_service.py
import logging
from typing import Optional
import google.generativeai as genai
from app.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    MODEL = "models/gemini-2.5-flash"

    BOOK_PROMPT = """You are a helpful assistant answering questions about a technical book on RAG systems.

INSTRUCTIONS:
- Answer ONLY using information from the provided context
- If the context doesn't contain relevant information, respond: "I don't have information about that in the book."
- NEVER invent or assume information not in the context
- Cite the chapter and section when answering
- Be concise and accurate

Context:
{context}

Question: {question}

Answer:"""

    SELECTED_TEXT_PROMPT = """You are a helpful assistant explaining selected text from a technical book.

INSTRUCTIONS:
- Answer ONLY about the selected text provided
- If the question isn't answerable from the selected text, say so
- Be concise and directly reference the text

Selected Text:
{selected_text}

Question: {question}

Answer:"""

    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=self.MODEL,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1000,
            )
        )

    def generate(self, question: str, context: str) -> str:
        prompt = self.BOOK_PROMPT.format(
            context=context,
            question=question
        )

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    def generate_from_selection(
        self, question: str, selected_text: str
    ) -> str:
        prompt = self.SELECTED_TEXT_PROMPT.format(
            selected_text=selected_text,
            question=question
        )

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Selection generation error: {e}")
            raise

    def check_health(self) -> bool:
        try:
            models = genai.list_models()
            return any(m for m in models)
        except Exception:
            return False
```

## 5.5 Zero-Hallucination Strategies

Preventing hallucination is paramount for RAG systems. Several strategies help ensure responses are grounded in retrieved content.

**Explicit instructions** in the system prompt tell the model not to invent information. While not foolproof, clear instructions significantly reduce hallucination rates.

**Low temperature** settings make generation more deterministic, reducing creative deviations from the source material.

**Fallback responses** provide a safe default when information is missing. The model is instructed to admit uncertainty rather than guess.

**Source verification** can be implemented by asking the model to quote relevant passages. If it cannot quote supporting text, the answer may be fabricated.

**Confidence thresholds** on retrieval can reject queries where no sufficiently similar chunks are found. If the best match has low similarity, the system declines to answer.

```python
def answer(self, question: str) -> Tuple[str, List[Source]]:
    query_embedding = self.embedding_service.embed_query(question)
    chunks = self.vector_service.search(query_vector=query_embedding, limit=5)

    # Confidence threshold - reject if best match is poor
    if not chunks or chunks[0].score < 0.5:
        return "I don't have information about that in the book.", []

    # Continue with generation...
```

## 5.6 Handling Rate Limits

The Gemini API has rate limits that must be handled gracefully.

```python
import time
from google.api_core.exceptions import ResourceExhausted

class LLMService:
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    def generate_with_retry(self, prompt: str) -> str:
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except ResourceExhausted as e:
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = self.RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Rate limited, retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries exceeded")
                    raise
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise
```

Exponential backoff increases wait times between retries, respecting rate limits while maximizing throughput.

## 5.7 Streaming Responses

For better user experience, streaming returns tokens as they're generated.

```python
def generate_stream(self, question: str, context: str):
    prompt = self.BOOK_PROMPT.format(
        context=context,
        question=question
    )

    response = self.model.generate_content(
        prompt,
        stream=True
    )

    for chunk in response:
        if chunk.text:
            yield chunk.text
```

The API endpoint can use Server-Sent Events (SSE) to stream to the frontend:

```python
from fastapi.responses import StreamingResponse

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        for token in llm_service.generate_stream(
            request.question, context
        ):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

## 5.8 Context Window Management

Gemini models have context window limits. Gemini 2.5 Flash supports up to 1 million tokens, but practical limits exist for cost and latency.

Calculate token usage and truncate if necessary:

```python
def prepare_context(self, chunks: List[Chunk], max_tokens: int = 8000) -> str:
    context_parts = []
    current_tokens = 0

    for chunk in chunks:
        chunk_text = self._format_chunk(chunk)
        chunk_tokens = len(chunk_text.split()) * 1.3  # Rough estimate

        if current_tokens + chunk_tokens > max_tokens:
            break

        context_parts.append(chunk_text)
        current_tokens += chunk_tokens

    return "\n\n---\n\n".join(context_parts)
```

Prioritize higher-relevance chunks by processing in score order.

## 5.9 Response Post-Processing

Post-processing can clean and validate generated responses.

```python
def process_response(self, response: str) -> str:
    # Remove potential prompt leakage
    response = response.strip()

    # Check for refusal patterns
    refusal_patterns = [
        "I don't have information",
        "I cannot find",
        "The context doesn't contain"
    ]

    is_refusal = any(p in response for p in refusal_patterns)

    if is_refusal:
        return "I don't have information about that in the book."

    return response
```

---

## Chapter Summary

This chapter covered LLM integration using Google's Gemini API for RAG generation. We configured the Gemini client with parameters optimized for accuracy over creativity. Prompt engineering ensures responses are grounded in retrieved context. Zero-hallucination strategies including explicit instructions, low temperature, and confidence thresholds reduce fabrication. Rate limit handling with exponential backoff ensures reliability. Streaming responses improve user experience. Context window management ensures prompts fit within limits. These techniques combine to create a robust generation component for production RAG systems.

---

## Review Questions

1. Why is low temperature recommended for RAG generation?
2. What elements should a RAG system prompt include to prevent hallucination?
3. How does exponential backoff help with rate limiting?
4. What are the tradeoffs between streaming and synchronous generation?
5. How can confidence thresholds prevent answering with insufficient context?

---

## Hands-On Exercises

**Exercise 5.1**: Implement the LLMService class and test with sample prompts. Compare outputs at temperature 0.1 vs 0.9.

**Exercise 5.2**: Create a test set of questions where answers should be refused. Verify the model correctly declines rather than hallucinating.

**Exercise 5.3**: Implement streaming generation and create a simple frontend that displays tokens as they arrive.
