---
title: Token Budget Policy
updated_at: 2025-11-17
owner: core-dev
lang: en
tags: [rag, tokens, budget, context-window, llm]
purpose: "Defines token allocation policies for AI-assisted development context management"
---

# Token Budget Policy

## Purpose

This document defines token allocation policies for assembling documentation context for LLM-assisted development. It establishes which documents auto-load, which require manual selection, and how to stay within context window limits while maximizing useful information.

## Context Window Limits

### LLM Model Capacities

| Model | Context Window | Reserved (System) | Reserved (Output) | Available for Docs |
|-------|----------------|-------------------|-------------------|-------------------|
| Claude Sonnet 4.5 | 200,000 tokens | ~2,000 tokens | ~4,000 tokens | **~194,000 tokens** |
| Claude Opus 3 | 200,000 tokens | ~2,000 tokens | ~4,000 tokens | ~194,000 tokens |
| GPT-4 Turbo | 128,000 tokens | ~1,500 tokens | ~4,000 tokens | ~122,500 tokens |
| GPT-4 | 32,000 tokens | ~1,500 tokens | ~4,000 tokens | ~26,500 tokens |

**Primary Target**: Claude Sonnet 4.5 (200k context window)

**Design Constraint**: Context assemblies should fit within 150k tokens to leave buffer for:
- User queries and conversation history (up to 40k tokens)
- Code snippets read during session (up to 10k tokens)
- Tool outputs and error messages (up to 4k tokens)

## Token Allocation Strategy

### Per-Layer Budget Allocation

Based on [chunking_strategy.md](chunking_strategy.md), we allocate tokens across documentation layers:

| Layer | Purpose | Auto-Load | Token Budget | Rationale |
|-------|---------|-----------|--------------|-----------|
| **00_index** | Navigation & design | Yes | 10,000 | Essential orientation |
| **10_modules** | Module cards | Yes | 40,000 | Core architecture knowledge |
| **20_interactions** | Sequence flows | Selective | 20,000 | Load by task context |
| **30_contracts** | API specifications | Yes | 50,000 | Critical for development |
| **40_config** | Configuration docs | Yes | 8,000 | Small, always useful |
| **50_rag** | RAG metadata | No | 5,000 | Meta-documentation |
| **85_memory** | Session memory | Yes | 5,000 | Tiny, stateful |
| **90_operations** | Operations guides | Selective | 15,000 | Load by task context |
| **95_adr** | Architecture decisions | Selective | 10,000 | Historical context |
| **Buffer** | Unallocated | - | 7,000 | Overflow/expansion |
| **Total** | | | **150,000** | |

### Auto-Load vs. Manual-Load

#### Auto-Load (Always Included)

These documents are included in every context assembly by default:

**00_index/** (10k tokens):
- ✅ `DESIGN.md` (3.2k tokens) - System overview
- ✅ `index.yaml` (2.5k tokens) - Document registry
- ✅ `modules.yaml` (4.0k tokens) - Module directory

**10_modules/** (40k tokens):
- ✅ All module cards (8 files × ~5k tokens each)
- Rationale: Core architecture knowledge, essential for any development task

**30_contracts/** (50k tokens):
- ✅ `order_object_contract.md` (3.2k)
- ✅ `depth_object_contract.md` (2.8k)
- ✅ `ticker_object_contract.md` (2.5k, future)
- ✅ `trade_object_contract.md` (2.5k, future)
- ✅ `strategy_context_api.md` (5.8k)
- ✅ `binance_config_contract.md` (3.4k)
- ✅ Other contracts as created (~30k remaining budget)
- Rationale: API contracts are referenced constantly during development

**40_config/** (8k tokens):
- ✅ All configuration documentation
- Rationale: Small, frequently needed for setup/debugging

**85_memory/** (5k tokens):
- ✅ `session_state.json` (dynamic, <5k)
- Rationale: Session continuity, tiny footprint

**Total Auto-Load**: ~113,000 tokens (well within budget)

#### Selective Load (Task-Specific)

These documents are loaded only when relevant to the current task:

**20_interactions/** (20k budget):
- 🔵 Load for **order management tasks**: `order_lifecycle_flow.md`, `trading_flow.md`
- 🔵 Load for **market data tasks**: `market_data_flow.md`, `event_flow.md`
- 🔵 Load for **strategy development**: `strategy_lifecycle_flow.md`

**90_operations/** (15k budget):
- 🔵 Load for **deployment tasks**: `pm2_startup_guide.md`, `docker_deployment.md`
- 🔵 Load for **debugging tasks**: `DEBUGGING.md`, `LOG_LOCATIONS.md`
- 🔵 Load for **CLI usage**: `cli_operations_guide.md`

**95_adr/** (10k budget):
- 🔵 Load for **architecture decisions**: Relevant ADRs only
- Example: Loading `004-binance-market-toggle.md` when working on Binance configuration

#### Opt-In Only (Manual Request)

These documents are NEVER auto-loaded due to size or low utility:

**❌ Large Documents:**
- ⛔ `doc/quantitative-trading-learning-path.plan.md` (4.8k tokens)
  - **Reason**: Educational content, not actionable development context
  - **Load when**: User explicitly asks "show me the learning path"

- ⛔ Future large files >10k tokens
  - **Reason**: Exceeds single-document budget
  - **Load when**: User explicitly references by name

**❌ External References:**
- ⛔ `ref/` directory (external code samples, examples)
  - **Reason**: Not part of core system, potentially outdated
  - **Load when**: User asks about specific reference material

**❌ Deprecated Documentation:**
- ⛔ `.doc_deprecated/` (old documentation archive)
  - **Reason**: Historical only, superseded by current docs
  - **Load when**: Investigating historical context or migration

## Token Counting Standards

### Estimation Rules

**Words-to-Tokens Ratio**: Use **1.5:1** for markdown documentation
- 1,000 words ≈ 1,500 tokens
- Accounts for markdown syntax, code blocks, YAML front-matter

**Accurate Counting**: Use `tiktoken` with `gpt-4` encoding
```python
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4")
tokens = len(encoding.encode(content))
```

### Token Estimates in index.yaml

Every document in `index.yaml` MUST have accurate `tokens_estimate`:

```yaml
documents:
  - path: 30_contracts/order_object_contract.md
    title: Order Object Contract
    tokens_estimate: 3200  # REQUIRED - use estimate_tokens.py
    layer: 30_contracts
    status: complete
```

**Validation**: Run `estimate_tokens.py` to verify estimates:
```bash
cd .doc/90_operations/scripts
python estimate_tokens.py  # Report discrepancies
python estimate_tokens.py --update-index  # Auto-update index.yaml
```

**Tolerance**: Estimates within ±10% are acceptable. Flag discrepancies >20% for review.

## Context Assembly Strategies

### Strategy 1: Full Context (Development)

**Use Case**: General development, strategy writing, debugging

**Includes**:
- All auto-load documents (~113k tokens)
- Relevant interaction flows (~10k tokens)
- Total: ~123k tokens

**Remaining Capacity**: 71k tokens for conversation, code reads, tool outputs

### Strategy 2: Contract-Focused (API Development)

**Use Case**: Implementing new exchange connectors, API changes

**Includes**:
- 00_index/ (10k)
- 30_contracts/ (50k) - ALL contracts
- 40_config/ (8k)
- Relevant 10_modules/ (20k) - Only gateway/framework modules
- Total: ~88k tokens

**Remaining Capacity**: 106k tokens (maximum space for dense work)

### Strategy 3: Operations-Focused (Deployment/Debugging)

**Use Case**: System administration, troubleshooting, deployment

**Includes**:
- 00_index/ (10k)
- 40_config/ (8k)
- 90_operations/ (15k) - ALL ops docs
- Relevant 10_modules/ (15k) - Runtime, journal, system modules
- 95_adr/ (5k) - Infrastructure ADRs
- Total: ~53k tokens

**Remaining Capacity**: 141k tokens (maximum space for log analysis, config files)

### Strategy 4: Minimal Context (Quick Questions)

**Use Case**: Simple queries, syntax questions, quick checks

**Includes**:
- 00_index/ (10k)
- Relevant single contract (~3k)
- Total: ~13k tokens

**Remaining Capacity**: 181k tokens (allows deep code exploration)

## Document Size Guidelines

### Maximum Document Size

**Soft Limit**: 8,000 tokens per document
- **Rationale**: Allows 18+ documents in auto-load budget
- Ensures granular retrieval (don't force-load huge irrelevant sections)

**Hard Limit**: 15,000 tokens per document
- **Rationale**: Any document >15k should be split into sub-documents
- Exception: Auto-generated API references (mark as opt-in)

### When to Split Documents

**Split if**:
- Single document exceeds 10k tokens
- Document covers multiple independent topics
- Only subset is relevant for most queries

**Example - Before Split**:
```
strategy_complete_guide.md (18k tokens)
  - Architecture (4k)
  - API Reference (8k)
  - Examples (6k)
```

**Example - After Split**:
```
10_modules/strategy_framework.md (5k tokens) - Architecture
30_contracts/strategy_context_api.md (6k tokens) - API Reference
30_contracts/strategy_examples.md (4k tokens) - Examples (opt-in)
```

## Large File Registry

### Known Large Files

| File | Tokens | Status | Policy |
|------|--------|--------|--------|
| `quantitative-trading-learning-path.plan.md` | 4,800 | Opt-in only | Educational, low dev utility |
| `10_modules/strategy_framework.md` | 5,100 | Auto-load | Core module, acceptable size |
| `30_contracts/strategy_context_api.md` | 5,800 | Auto-load | Critical API, acceptable size |

### Future Monitoring

Run token analysis regularly to catch size growth:

```bash
cd .doc/90_operations/scripts
python estimate_tokens.py | grep "⚠️"  # Files with >10% error
```

**Review Trigger**: Any document growing beyond 8k tokens should be evaluated for splitting.

## RAG Retrieval Budget

### Top-K Retrieval Limits

When using RAG retrieval (vector similarity search):

**Initial Retrieval**: Top 10 chunks
- 10 chunks × 750 tokens avg = **7,500 tokens**

**Re-ranked Selection**: Top 5 chunks
- 5 chunks × 750 tokens avg = **3,750 tokens**

**Adjacent Context**: ±1 chunk for each selected
- 5 selections × 2 adjacent × 750 tokens = **7,500 tokens**

**Linked Documents**: 1-hop links (1-2 additional chunks)
- 2 chunks × 750 tokens = **1,500 tokens**

**Total RAG Context**: ~20,000 tokens (conservative estimate)

This leaves **130k tokens** for base auto-load documents and conversation.

### Retrieval vs. Full Context

**Use Retrieval When**:
- Query is specific and narrow (e.g., "How to parse Binance depth message?")
- Only 1-2 documents needed
- Want to minimize token usage

**Use Full Context When**:
- Query is broad (e.g., "How does the trading system work?")
- Multiple documents likely relevant
- Developing complex features requiring cross-references

## Context Window Overflow Handling

### If Context Exceeds 150k

**Priority-Based Dropping** (in order):

1. **Drop opt-in documents first** (if accidentally loaded)
2. **Drop 95_adr/** (unless explicitly needed)
3. **Drop 90_operations/** (except for ops-focused tasks)
4. **Drop 20_interactions/** (except for flow-specific tasks)
5. **Reduce 10_modules/** (keep only directly relevant modules)
6. **Never drop**: 00_index/, 30_contracts/, 40_config/, 85_memory/

### Warning Thresholds

**Yellow Alert** (140k tokens): Review loaded documents, consider dropping ADRs
**Red Alert** (150k tokens): Immediate action required, drop non-essential layers

## Updates and Maintenance

### Regular Audits

**Monthly**:
- Run `estimate_tokens.py` to update all token counts
- Review documents exceeding size guidelines
- Check total auto-load budget (should be <120k)

**Per-Document**:
- Update `tokens_estimate` in `index.yaml` when document changes
- Mark documents as `opt-in` if they exceed 10k tokens
- Split documents approaching 15k tokens

### CI Integration

See [.doc/90_operations/ci_validation.md](../90_operations/ci_validation.md) (future) for automated checks:

```yaml
# .github/workflows/docs-validation.yml
- name: Check Token Budget
  run: |
    cd .doc/90_operations/scripts
    python estimate_tokens.py

    # Fail if auto-load exceeds 120k
    total=$(python -c "import yaml; print(sum(d['tokens_estimate'] for d in yaml.safe_load(open('../../00_index/index.yaml'))['documents'] if d.get('auto_load', False)))")
    if [ $total -gt 120000 ]; then
      echo "ERROR: Auto-load budget exceeded: $total tokens"
      exit 1
    fi
```

## Version History

- **2025-11-17**: Initial token budget policy
