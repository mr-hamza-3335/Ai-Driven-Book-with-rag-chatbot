# Chapter 3: Embeddings and Chunking Strategies

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand how text embeddings represent semantic meaning
- Compare popular embedding models and their characteristics
- Implement effective chunking strategies for different document types
- Handle metadata preservation during chunking
- Evaluate and optimize embedding quality for RAG applications

---

## 3.1 The Nature of Text Embeddings

Text embeddings are dense vector representations that capture the semantic meaning of text in a high-dimensional space. Unlike sparse representations such as bag-of-words or TF-IDF, embeddings encode meaning in every dimension, allowing for nuanced similarity comparisons.

The fundamental insight behind embeddings is the distributional hypothesis: words that appear in similar contexts tend to have similar meanings. Modern embedding models extend this principle to sentences and paragraphs, learning representations where semantically related texts cluster together regardless of surface-level lexical differences.

Consider two sentences: "The cat sat on the mat" and "A feline rested on the rug." Despite sharing few words, these sentences express similar meanings. A good embedding model will produce vectors that are close together for these sentences. Conversely, "The cat sat on the mat" and "Financial markets crashed today" should produce distant vectors.

Embeddings enable semantic search, which is fundamentally different from keyword matching. With keyword search, a query for "automobile" would miss documents containing only "car." With embedding-based search, the semantic relationship between "automobile" and "car" is captured in vector space, enabling relevant retrieval despite vocabulary differences.

The dimensionality of embeddings represents a tradeoff. Higher dimensions can capture more nuanced relationships but require more storage and computation. Modern embedding models typically produce vectors between 384 and 1536 dimensions. Gemini's text-embedding-004 uses 768 dimensions, striking a balance between expressiveness and efficiency.

## 3.2 Embedding Model Architectures

Modern embedding models are built on transformer architectures, specifically encoder models that process entire sequences bidirectionally to capture contextual relationships.

BERT-based embeddings pioneered the approach. The original BERT model used 768-dimensional representations from the [CLS] token or mean-pooling of token embeddings. Sentence-BERT (SBERT) improved on this by training specifically for sentence similarity tasks using siamese and triplet network structures.

Contrastive learning has become the dominant training paradigm. Models learn by distinguishing between positive pairs (semantically similar texts) and negative pairs (dissimilar texts). This approach, used in models like OpenAI's text-embedding and Cohere's embed, produces embeddings well-suited for retrieval tasks.

Gemini's embedding models represent Google's state-of-the-art approach. The text-embedding-004 model produces 768-dimensional embeddings optimized for semantic similarity and retrieval. It handles texts up to 2048 tokens and supports task-specific prefixes to guide embedding generation.

```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")

result = genai.embed_content(
    model="models/text-embedding-004",
    content="RAG combines retrieval with generation",
    task_type="retrieval_document"
)

embedding = result['embedding']  # 768-dimensional vector
```

The `task_type` parameter is important. Use "retrieval_document" when embedding documents for storage and "retrieval_query" when embedding user queries. This asymmetric approach improves retrieval accuracy.

## 3.3 Comparing Embedding Models

Choosing the right embedding model impacts RAG system performance significantly. Key factors include dimensionality, maximum sequence length, multilingual support, and benchmark performance.

OpenAI's text-embedding-3-large produces 3072-dimensional embeddings (or configurable smaller sizes) with strong performance across benchmarks. It supports up to 8191 tokens and offers good multilingual capabilities. The higher dimensionality captures more nuance but increases storage and search costs.

Cohere's embed-v3 offers 1024-dimensional embeddings with excellent multilingual support for over 100 languages. It includes specific optimizations for search and classification tasks and supports up to 512 tokens.

Open-source alternatives like Sentence Transformers provide flexibility without API costs. Models like all-MiniLM-L6-v2 (384 dimensions) offer good performance for many use cases while being free and self-hostable. Larger models like instructor-xl provide state-of-the-art accuracy.

Gemini's text-embedding-004 provides 768-dimensional embeddings with Google's latest AI research. It supports 2048 tokens, handles multilingual content well, and integrates seamlessly with other Gemini capabilities. For projects already using Gemini for generation, using Gemini embeddings simplifies the architecture.

Evaluation should consider your specific use case. General benchmarks like MTEB (Massive Text Embedding Benchmark) provide guidance, but testing on your actual data is essential. Create a small evaluation set with query-document pairs and measure retrieval accuracy with different models.

## 3.4 The Importance of Chunking

Chunking—splitting documents into smaller segments—is one of the most impactful decisions in RAG system design. The goal is to create chunks that are semantically coherent, appropriately sized for the embedding model and LLM context window, and optimized for retrieval precision.

Why not embed entire documents? Several reasons. Embedding models have maximum sequence lengths (typically 512-8192 tokens). Longer texts may be truncated. Additionally, a query about a specific topic might not match well against a long document covering many topics. Finally, retrieved chunks are passed to the LLM, and large chunks consume precious context window space.

The challenge is finding the optimal chunk size. Chunks too small may lack necessary context—a sentence fragment might be meaningless without surrounding sentences. Chunks too large may cover multiple topics, diluting relevance signals and including irrelevant information in the generation context.

There is no universal optimal chunk size. It depends on your content type, embedding model, retrieval requirements, and downstream LLM. Experimentation and evaluation are essential.

## 3.5 Chunking Strategies

Several chunking strategies exist, each with advantages for different content types.

Fixed-size chunking splits text into segments of a specified token or character count, typically with overlap between chunks. This simple approach works reasonably well for uniform content like transcripts or articles.

```python
def fixed_size_chunk(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
```

Overlap ensures that content spanning chunk boundaries is captured in at least one chunk. Common overlap values are 10-20% of chunk size.

Semantic chunking uses natural boundaries like paragraphs, sections, or topic shifts. This preserves coherent units of meaning but produces variable-sized chunks.

```python
def paragraph_chunk(text):
    paragraphs = text.split('\n\n')
    return [p.strip() for p in paragraphs if p.strip()]
```

Recursive chunking applies a hierarchy of separators. First try to split on section headers, then paragraphs, then sentences, then by character count. This approach adapts to document structure.

Markdown-aware chunking respects document structure in formatted content. Headers become chunk boundaries, preserving section integrity.

```python
import re

def markdown_chunk(text, max_size=1000):
    sections = re.split(r'\n(?=##?\s)', text)
    chunks = []
    for section in sections:
        if len(section) <= max_size:
            chunks.append(section)
        else:
            # Further split large sections
            paragraphs = section.split('\n\n')
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) <= max_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
            if current_chunk:
                chunks.append(current_chunk.strip())
    return chunks
```

## 3.6 Metadata Preservation

Effective RAG requires not just chunk text but also metadata that enables filtering, attribution, and context reconstruction.

Essential metadata includes source document identification (filename, URL, document ID), position information (page number, section, offset), structural context (chapter title, section header), and timestamps (creation date, last modified).

During chunking, preserve hierarchical context. A chunk from section 3.2 should carry information about chapter 3's title and the section header. This enables filtered retrieval and proper citations.

```python
def chunk_with_metadata(document, doc_id):
    chunks = []
    sections = parse_sections(document)

    for section in sections:
        for i, chunk_text in enumerate(chunk_section(section.content)):
            chunks.append({
                "id": f"{doc_id}-{section.number}-{i}",
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "chapter": section.chapter,
                    "section": section.title,
                    "section_number": section.number,
                    "chunk_index": i
                }
            })
    return chunks
```

Consider including surrounding context in the embedding while keeping it separate from the retrieved text. Some systems prepend section titles to chunk text before embedding, improving retrieval accuracy while keeping the raw text clean for generation.

## 3.7 Handling Different Document Types

Different document types require tailored chunking approaches.

PDFs often contain complex layouts with tables, images, and multi-column text. Tools like PyPDF2, pdfplumber, or commercial solutions like Textract extract text, but layout information may be lost. Consider using document AI services for complex PDFs or OCR for scanned documents.

Code repositories benefit from file-based and function-based chunking. Split on function definitions, classes, or logical blocks. Preserve file paths and language information in metadata.

Structured data like tables should be handled specially. Convert rows to natural language statements or maintain tabular format with clear headers. Tables chunked poorly become incomprehensible.

Conversational content (chat logs, transcripts) should be chunked to maintain dialogue coherence. Split at topic boundaries rather than mid-conversation.

## 3.8 Evaluating Chunk Quality

Measuring chunking effectiveness requires systematic evaluation.

Retrieval evaluation uses a test set of queries with known relevant chunks. Calculate recall (percentage of relevant chunks retrieved) and precision (percentage of retrieved chunks that are relevant) at different k values.

Downstream evaluation measures end-to-end RAG performance. Do different chunking strategies improve final answer quality? This is the ultimate test but requires labeled question-answer pairs.

Semantic coherence can be measured by embedding chunks and analyzing their clustering. Coherent chunks should have embeddings with consistent topics. Incoherent chunks may produce embeddings in unexpected regions.

Manual inspection remains valuable. Read through sample chunks and assess whether they make sense in isolation, contain complete thoughts, and would provide useful context for answering questions.

---

## Chapter Summary

This chapter explored the critical roles of embeddings and chunking in RAG systems. Embeddings convert text into semantic vector representations enabling similarity search. We compared embedding models including Gemini's text-embedding-004 and discussed selection criteria. Chunking strategies range from simple fixed-size splitting to sophisticated structure-aware approaches. Metadata preservation ensures proper attribution and enables filtered retrieval. Handling different document types requires tailored approaches, and systematic evaluation helps optimize these choices for your specific application.

---

## Review Questions

1. How do text embeddings enable semantic search beyond keyword matching?
2. What is the purpose of the task_type parameter in Gemini embeddings?
3. Why is chunk size a critical decision in RAG system design?
4. Compare fixed-size chunking with semantic chunking. When would you use each?
5. What metadata should be preserved during chunking and why?

---

## Hands-On Exercises

**Exercise 3.1**: Generate embeddings for 10 sample sentences using Gemini's text-embedding-004. Calculate cosine similarity between all pairs and visualize which sentences are most similar.

**Exercise 3.2**: Implement three chunking strategies (fixed-size, paragraph-based, markdown-aware) and apply them to a sample document. Compare the resulting chunks.

**Exercise 3.3**: Create a chunking pipeline that preserves metadata including source file, section headers, and chunk position. Test on a multi-section markdown document.
