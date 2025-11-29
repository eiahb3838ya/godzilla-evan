---
title: .doc System Design Philosophy
updated_at: 2025-11-17
owner: core-dev
lang: en
---

# .doc System Design Philosophy

## Purpose

This document explains the design philosophy and structure of the `.doc/` documentation system for the godzilla-evan trading framework (kungfu fork).

## Design Principles

### 1. Single Source of Truth
Each concept, API, or configuration is documented in exactly one place:
- Module functionality: `10_modules/`
- Cross-module flows: `20_interactions/`
- Interface contracts: `30_contracts/`
- Configuration keys: `40_config/`

All other documents reference the authoritative source.

### 2. Layered Abstraction
Documentation is organized by abstraction level:
```
00_index/         <- Start here (high-level overview)
10_modules/       <- Individual component details
20_interactions/  <- How components work together
30_contracts/     <- Interface specifications
40_config/        <- Configuration mappings
50_rag/           <- AI retrieval optimization
85_memory/        <- Troubleshooting & lessons learned
90_operations/    <- Maintenance procedures
adr/              <- Architecture decisions (cross-cutting)
```

### 3. Traceability
Every technical claim must be verifiable:
- Code references: `core/cpp/yijinjing/src/journal/journal.cpp:150`
- Documentation references: `.doc/10_modules/yijinjing.md#performance`
- No speculation without marking it as such

### 4. RAG-Friendly Structure
All documents use structured YAML front-matter:
```yaml
---
title: Clear title
updated_at: 2025-11-17
owner: Responsible person/team
tags: [module, trading, event-sourcing]
code_refs:
  - path/to/file.cpp:line
---
```

This enables:
- Vector database indexing
- Semantic search
- Automated context assembly
- Version tracking

### 5. Minimal Code Invasion
We do NOT modify source code for documentation:
- ❌ No `# ctx:anchor:name` comments in code
- ✅ Use `path:line` references only
- ✅ Documentation stays in `.doc/`

## Documentation Layers Explained

### 00_index/ - Metadata Layer
- `DESIGN.md` - This file
- `index.yaml` - Machine-readable document catalog
- `modules.yaml` - Context module definitions for different scenarios

### 00_index/ - Entry Point
High-level overviews for different audiences:
- New developers: Project overview, setup guide
- Feature developers: Architecture, module map
- Operators: Deployment, monitoring

### 10_modules/ - Component Cards
One card per independent module:
- `yijinjing.md` - Event sourcing system
- `wingchun.md` - Trading gateway
- `binance_extension.md` - Binance exchange connector
- etc.

Each card follows a standard template (see DOC.md for details).

### 20_interactions/ - Flow Documentation
How modules interact:
- `trading_flow.md` - Complete order execution flow
- `event_flow.md` - Event propagation through system
- `config_to_runtime.md` - Configuration consumption

Uses sequence diagrams (Mermaid) and detailed parameter mappings.

### 30_contracts/ - Interface Specifications
Formal contracts between modules:
- Input/output types
- Invariants (conditions that must always hold)
- Error handling boundaries
- Performance guarantees

### 40_config/ - Configuration Management
- `config_usage_map.md` - Which code uses which config key
- `dangerous_keys.md` - High-risk configurations (testnet vs live)
- Prevents "config lost in translation" issues

### 50_rag/ - RAG Optimization
- Document chunking strategy
- Retrieval evaluation benchmarks
- QA sample pairs for testing

### 85_memory/ - Troubleshooting Database
- Real debugging case studies
- Common pitfalls and solutions
- Performance tuning guides
- Lessons learned

### 90_operations/ - Maintenance Guides
- Documentation update procedures
- Verification scripts
- Deployment checklists

### adr/ - Architecture Decision Records
Why we made certain choices:
- ADR-001: Docker design
- ADR-002: WSL2 strategy
- ADR-003: DNS configuration
- ADR-004: Binance market toggle

## Usage Scenarios

### Scenario 1: New Developer Onboarding
1. Read `00_index/INDEX.md` - Project overview
2. Read `00_index/INSTALL.md` - Setup environment
3. Read `10_modules/yijinjing.md` and `10_modules/wingchun.md` - Core concepts
4. Read `20_interactions/trading_flow.md` - Understand end-to-end flow
5. Estimated time: 2-3 hours

### Scenario 2: Implementing a New Feature
1. Check `10_modules/` for affected components
2. Check `30_contracts/` for interface constraints
3. Check `20_interactions/` to understand impact on flows
4. Update all three after implementation
5. Update `40_config/` if new configuration added

### Scenario 3: Debugging an Issue
1. Check `85_memory/DEBUGGING.md` for similar cases
2. Check `40_config/config_usage_map.md` for config issues
3. Check `20_interactions/` to understand expected flow
4. Check `10_modules/` for component-specific details

### Scenario 4: AI-Assisted Development
The AI can load different context modules based on task:
- `onboarding` - For explaining project structure
- `development` - For implementing features
- `troubleshooting` - For debugging
- `deep_dive` - For architectural understanding

See `00_index/modules.yaml` for definitions.

## Evolution Strategy

This documentation system follows **gradual evolution**:

### Phase 1 (Current) - Core Foundations
- ✅ Directory structure created
- ⏳ Populate `00_index/` with essential guides
- ⏳ Create core module cards (yijinjing, wingchun, binance)
- ⏳ Document critical interaction flows
- ⏳ Build config usage map

### Phase 2 (Next Month) - Deepening
- Add contract specifications
- Enhance RAG optimization
- Consolidate troubleshooting knowledge
- Expand module card coverage

### Phase 3 (Ongoing) - Maintenance
- Keep docs synchronized with code
- Refine based on actual usage
- Expand as needed, not preemptively

## Quality Standards

Every document should:
- ✅ Have complete front-matter
- ✅ Include executable examples
- ✅ Reference actual code locations
- ✅ Be up-to-date with current code
- ✅ Use consistent terminology
- ✅ Be concise and scannable
- ❌ Contain no emojis (professional style)
- ❌ Have no vague statements ("might", "could", "possibly")

## Maintenance

### When Adding a Feature
1. Update relevant module card(s) in `10_modules/`
2. Update interaction diagrams in `20_interactions/`
3. Update contracts if interfaces changed in `30_contracts/`
4. Update config map if new config keys in `40_config/`
5. Add to ADR if architecturally significant

### When Refactoring
1. Verify all `path:line` references still valid
2. Update affected contracts in `30_contracts/`
3. Update module cards if APIs changed
4. Document rationale in relevant ADR or create new one

### Monthly Review
1. Check all `updated_at` timestamps
2. Verify code references with actual code
3. Test AI retrieval quality
4. Clean up stale information

## Tools and Automation

### Verification Scripts
Located in `.doc/scripts/`:
- `verify-code-refs.sh` - Check all `path:line` references
- `update-timestamps.sh` - Update `updated_at` fields
- `generate-index.py` - Regenerate `index.yaml`

### RAG Evaluation
See `.doc/50_rag/` for:
- Chunking configuration
- Retrieval benchmarks
- QA test pairs

## References

- Original design philosophy: DOC.md
- Context modules: 00_index/modules.yaml
- Document index: 00_index/index.yaml
- Deprecated old docs: ../.deprecated_doc/
