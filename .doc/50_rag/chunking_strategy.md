---
title: RAG Chunking Strategy
updated_at: 2025-11-17
owner: core-dev
lang: en
tags: [rag, chunking, embedding, retrieval, ai]
purpose: "Defines how documentation is chunked for vector embedding and retrieval"
---

# RAG Chunking Strategy

## Purpose

This document defines the chunking strategy for breaking documentation into semantic units for vector embedding and retrieval-augmented generation (RAG). Proper chunking balances context preservation with retrieval precision.

## Chunking Parameters

### Chunk Size

**Target:** 500-1000 tokens per chunk
**Rationale:**
- **Too small (<300 tokens):** Loses context, requires more chunks to answer questions
- **Too large (>1500 tokens):** Reduces retrieval precision, wastes embedding space
- **Sweet spot (500-1000):** Preserves semantic context while maintaining granularity

### Overlap

**Size:** 50-100 tokens
**Rationale:**
- Prevents information loss at chunk boundaries
- Ensures code examples aren't split mid-block
- Helps with cross-reference continuity

**Example:**
```
Chunk 1: Tokens 0-1000
Chunk 2: Tokens 950-1950  (50 token overlap)
Chunk 3: Tokens 1900-2900 (50 token overlap)
```

## Chunk Boundaries

### Primary: Semantic Sections

Split on markdown headings (`##`, `###`) to preserve semantic cohesion:

```markdown
## API Methods              ← Chunk boundary

### insert_order()          ← Potential sub-chunk if section >1000 tokens
Content about insert_order...

### cancel_order()          ← Sub-chunk boundary
Content about cancel_order...

## Configuration            ← Chunk boundary
```

**Advantages:**
- Each chunk is self-contained topic
- Natural question-answer alignment
- Preserves document structure

### Secondary: Code Block Preservation

**Rule:** NEVER split code blocks mid-block.

**Strategy:**
- If code block fits in current chunk → include it
- If code block would overflow → start new chunk before code block
- If code block alone exceeds chunk size → keep it intact anyway

**Example:**
```python
# This entire code block stays together
def on_depth(context, depth):
    # ... 200 lines of code ...
    pass
```

Even if this exceeds 1000 tokens, it remains a single chunk for readability.

### Tertiary: Paragraph Boundaries

Within a section, prefer splitting at paragraph boundaries (double newline).

## Metadata Embedding

Each chunk includes metadata for retrieval:

```json
{
  "chunk_id": "order_contract_001",
  "document_path": "30_contracts/order_object_contract.md",
  "document_title": "Order Object Contract",
  "section_heading": "## Structure Definition",
  "section_level": 2,
  "tags": ["contract", "order", "trading"],
  "code_refs": [
    "core/cpp/wingchun/include/kungfu/wingchun/msg.h:666-730"
  ],
  "chunk_index": 1,
  "total_chunks": 5,
  "token_count": 847,
  "layer": "30_contracts"
}
```

**Usage in Retrieval:**
- `tags`: For filtering (e.g., only "contract" tags for API queries)
- `layer`: For context prioritization (30_contracts > 10_modules for API questions)
- `section_heading`: For result presentation
- `code_refs`: For linking to source code

## Chunking Algorithm

### Pseudocode

```python
def chunk_document(md_content: str, front_matter: dict) -> list[Chunk]:
    chunks = []
    current_chunk = ""
    current_tokens = 0
    current_heading = ""

    for block in parse_markdown(md_content):
        if block.type == "heading":
            # Heading triggers new chunk (if current chunk non-empty)
            if current_tokens > 0:
                chunks.append(finalize_chunk(current_chunk, front_matter, current_heading))
                current_chunk = ""
                current_tokens = 0

            current_heading = block.text
            current_chunk += block.raw_text
            current_tokens += count_tokens(block.raw_text)

        elif block.type == "code_block":
            code_tokens = count_tokens(block.raw_text)

            # If adding code would overflow, finalize current chunk first
            if current_tokens + code_tokens > TARGET_SIZE and current_tokens > 0:
                chunks.append(finalize_chunk(current_chunk, front_matter, current_heading))
                current_chunk = ""
                current_tokens = 0

            # Add code block (even if it exceeds target)
            current_chunk += block.raw_text
            current_tokens += code_tokens

        elif block.type == "paragraph":
            para_tokens = count_tokens(block.raw_text)

            # If adding para would significantly overflow, start new chunk
            if current_tokens + para_tokens > MAX_SIZE:
                chunks.append(finalize_chunk(current_chunk, front_matter, current_heading))
                # Add overlap from previous chunk
                current_chunk = get_overlap(chunks[-1], OVERLAP_TOKENS)
                current_tokens = count_tokens(current_chunk)

            current_chunk += block.raw_text
            current_tokens += para_tokens

    # Finalize last chunk
    if current_tokens > 0:
        chunks.append(finalize_chunk(current_chunk, front_matter, current_heading))

    return chunks
```

## Special Cases

### Front-Matter

YAML front-matter is included ONLY in the first chunk:

```yaml
---
title: Order Object Contract
updated_at: 2025-11-17
tags: [contract, order, trading]
---
```

This preserves metadata context for the document's introduction.

### Tables

**Strategy:** Keep tables intact if possible.

- Small tables (<200 tokens): Include in current chunk
- Large tables (>200 tokens): Start new chunk, keep table together
- Huge tables (>1000 tokens): Allow splitting at row boundaries

### Lists

**Strategy:** Keep list items together when possible.

- If list fits in chunk → include it
- If list would overflow → split at list item boundaries
- Preserve list hierarchy (nested lists stay with parent)

### Cross-References

**Strategy:** Preserve link context.

When a chunk contains links to other documents:
```markdown
See [Order Contract](../30_contracts/order_object_contract.md) for details.
```

The chunk metadata includes:
```json
"outbound_links": [
  "30_contracts/order_object_contract.md"
]
```

This enables graph-based retrieval (follow links to related docs).

## Retrieval Considerations

### Query-to-Chunk Matching

**Embedding Model:** `text-embedding-3-small` (OpenAI) or equivalent
**Similarity Metric:** Cosine similarity
**Top-K:** Retrieve top 5-10 chunks per query

### Re-Ranking

After initial retrieval, re-rank by:
1. **Cosine similarity** (primary)
2. **Layer priority** (30_contracts > 10_modules for API questions)
3. **Tag relevance** (bonus for matching query tags)
4. **Recency** (`updated_at` timestamp)

### Context Assembly

For a single query, assemble context from:
- **Top 3 chunks** (highest similarity)
- **Adjacent chunks** (chunk_index ± 1 for context)
- **Linked documents** (follow `outbound_links` for 1 hop)

**Total context budget:** 20,000 tokens (leaves room for system prompt + output)

## Example Chunking Output

### Input Document

`30_contracts/order_object_contract.md` (3,200 tokens)

### Chunked Output

**Chunk 1** (650 tokens):
- Front-matter
- Title
- Purpose section
- Structure Definition heading + table

**Chunk 2** (720 tokens):
- Overlap from Chunk 1 (last 50 tokens)
- Enum Values section
- Side enum listing
- OrderStatus enum listing

**Chunk 3** (850 tokens):
- Overlap from Chunk 2
- Invariants section (all 6 invariants)
- State Machine heading

**Chunk 4** (680 tokens):
- Overlap from Chunk 3
- State machine diagram (Mermaid code)
- Terminal States explanation

**Chunk 5** (580 tokens):
- Overlap from Chunk 4
- Usage Examples section
- C++ example code block

## Implementation

### Tooling

**Recommended:** LangChain `RecursiveCharacterTextSplitter` with markdown mode

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,           # Target size in tokens
    chunk_overlap=50,          # Overlap in tokens
    length_function=tiktoken_len,  # Use tiktoken for accurate token counting
    separators=[
        "\n## ",               # H2 headings (highest priority)
        "\n### ",              # H3 headings
        "\n#### ",             # H4 headings
        "\n\n",                # Paragraphs
        "\n",                  # Lines
        " ",                   # Words
        ""                     # Characters (last resort)
    ],
    is_separator_regex=False
)

chunks = splitter.split_text(markdown_content)
```

### Embedding Storage

**Vector Database Options:**
- **ChromaDB** (local, simple)
- **Pinecone** (cloud, scalable)
- **Weaviate** (self-hosted, feature-rich)

**Schema:**
```sql
CREATE TABLE embeddings (
    chunk_id TEXT PRIMARY KEY,
    document_path TEXT,
    section_heading TEXT,
    layer TEXT,
    tags TEXT[],
    chunk_index INTEGER,
    token_count INTEGER,
    embedding VECTOR(1536),  -- text-embedding-3-small dimension
    content TEXT,
    metadata JSONB
);

CREATE INDEX idx_layer ON embeddings(layer);
CREATE INDEX idx_tags ON embeddings USING GIN(tags);
```

## Evaluation

### Quality Metrics

**Retrieval Precision:** % of retrieved chunks that are relevant
**Retrieval Recall:** % of relevant chunks that are retrieved
**Context Sufficiency:** % of queries answerable from retrieved chunks

**Target:** >80% precision, >70% recall

### Test Queries

See [retrieval_eval.yaml](retrieval_eval.yaml) for test query set.

**Sample Query:**
```yaml
query: "How do I place an order in my strategy?"
expected_chunks:
  - 30_contracts/strategy_context_api.md#insert_order
  - 10_modules/strategy_framework.md#order-management
  - 30_contracts/order_object_contract.md#usage-examples
```

## Version History

- **2025-11-17:** Initial chunking strategy definition
