---
title: .doc System Organization
updated_at: 2025-11-17
owner: core-dev
lang: en
purpose: "Explains how the .doc documentation system is structured for AI-assisted development"
---

# .doc Documentation System Organization

## Purpose

This document explains the organization principles of the `.doc/` directory structure. This system is designed for AI consumption to assist development, not for human browsing.

## Core Principles

1. **AI-First**: All documentation is structured for AI retrieval and code generation assistance
2. **Single Source of Truth**: Each concept defined once, referenced elsewhere via `path:line`
3. **Layered Architecture**: Progressive depth from overview → modules → interactions → contracts
4. **Traceability**: All technical claims must have code references
5. **Zero Code Invasion**: Documentation stays in `.doc/`, no code modifications for doc anchors

## Directory Structure

```
.doc/
├── 00_index/          # Entry point + AI metadata
│   ├── DESIGN.md             # System design philosophy
│   ├── index.yaml            # Machine-readable document catalog
│   ├── modules.yaml          # AI context module definitions
│   ├── INSTALL.md            # Environment setup
│   ├── ARCHITECTURE.md       # System architecture overview
│   ├── TESTNET.md            # Binance testnet configuration
│   └── ... (other entry docs)
│
├── 10_modules/        # Component documentation
│   ├── yijinjing.md          # Event sourcing system
│   ├── wingchun.md           # Trading gateway framework
│   ├── binance_extension.md  # Binance connector
│   └── ...
│
├── 20_interactions/   # Cross-component flows
│   ├── trading_flow.md       # Order execution flow
│   ├── event_flow.md         # Event propagation patterns
│   └── ...
│
├── 30_contracts/      # Interface specifications
│   └── ... (to be created)
│
├── 40_config/         # Configuration management
│   ├── config_usage_map.md   # Config key → code location mapping
│   ├── dangerous_keys.md     # High-risk configuration documentation
│   └── examples/             # Configuration examples
│
├── 50_rag/            # RAG optimization
│   └── ... (to be created)
│
├── 85_memory/         # Troubleshooting knowledge
│   └── DEBUGGING.md          # Case studies and solutions
│
├── 90_operations/     # Maintenance procedures
│   └── scripts/              # Utility scripts
│
└── 95_adr/            # Architecture Decision Records
    ├── 001-docker.md
    ├── 002-wsl2.md
    ├── 003-dns.md
    └── 004-binance-market-toggle.md
```

## Layer Responsibilities

| Layer | Purpose | AI Use Case |
|-------|---------|-------------|
| **00_index/** | Entry point, AI context definitions | Initial system understanding, context assembly |
| **10_modules/** | Independent component details | Understanding specific modules, API reference |
| **20_interactions/** | Component collaboration patterns | Understanding end-to-end flows, debugging |
| **30_contracts/** | Interface contracts and invariants | Integration work, API compatibility |
| **40_config/** | Configuration key mappings | Configuration debugging, risk assessment |
| **50_rag/** | RAG system optimization | Improving AI retrieval quality |
| **85_memory/** | Troubleshooting case studies | Debugging, learning from past issues |
| **90_operations/** | Documentation maintenance | Maintaining the documentation system |
| **95_adr/** | Architecture decisions | Understanding design rationale |

## Document Standards

### Front-matter (Required)

Every `.md` file must include YAML front-matter:

```yaml
---
title: Clear descriptive title
updated_at: YYYY-MM-DD
owner: Responsible maintainer
lang: en
tags: [relevant, keywords, for, retrieval]
purpose: "One-line description of document purpose"
code_refs:  # Optional, list of relevant code locations
  - path/to/file.cpp:line
---
```

### Code References

Use `path:line` format for code references:
- `core/cpp/yijinjing/src/journal/journal.cpp:150`
- `core/extensions/binance/src/trader_binance.cpp:123-200`

**Never** modify source code for documentation purposes.

### Writing Style

- **Language**: English (consistent with codebase)
- **Precision**: Avoid ambiguity, use exact technical terms
- **Examples**: Provide executable code examples
- **Conciseness**: Focus on essential information for AI understanding
- **No Emojis**: Keep professional and clean

## Implementation Status

### Completed (Phase 1-2)

- ✅ Directory structure with all layers
- ✅ AI metadata system (`00_index/`: DESIGN.md, index.yaml, modules.yaml)
- ✅ Entry documentation (`00_index/`: INSTALL, ARCHITECTURE, TESTNET, etc.)
- ✅ Core module cards (`10_modules/`: yijinjing, wingchun, binance_extension)
- ✅ Interaction flows (`20_interactions/`: trading_flow, event_flow)
- ✅ Configuration management (`40_config/`: config_usage_map, dangerous_keys)
- ✅ Troubleshooting knowledge (`85_memory/`: DEBUGGING)
- ✅ Architecture decisions (`95_adr/`: 4 ADRs)

### Planned (Phase 3)

- ⏳ Contract specifications (`30_contracts/`)
- ⏳ RAG optimization (`50_rag/`)
- ⏳ Additional module cards as needed
- ⏳ Additional interaction flows as needed

## Maintenance

### When Adding Features

1. Update relevant module card (`10_modules/`)
2. Update interaction diagrams (`20_interactions/`) if flow changes
3. Update contracts (`30_contracts/`) if interfaces change
4. Update config mapping (`40_config/`) if new config keys added
5. Update AI context modules (`00_index/modules.yaml`) if new documentation type
6. Update document catalog (`00_index/index.yaml`)

### When Refactoring

1. Verify all `path:line` references still valid
2. Update affected module cards
3. Update affected interaction flows
4. Document rationale in ADR (`95_adr/`) if architecturally significant

## AI Context Assembly

AI assistants can load context using predefined modules (`00_index/modules.yaml`):

- **onboarding**: For new developer questions (entry docs + core modules)
- **development**: For feature implementation (modules + interactions + config)
- **troubleshooting**: For debugging (debugging + config + interactions)
- **deep_dive**: For architectural work (modules + interactions + contracts + ADRs)
- **operations**: For deployment (operations + config + testnet)

See `00_index/DESIGN.md` for detailed context assembly strategy.

## References

- Design philosophy: `00_index/DESIGN.md`
- Document catalog: `00_index/index.yaml`
- Context modules: `00_index/modules.yaml`
- Deprecated old docs: `.deprecated_doc/`

## Version History

- **2025-11-17**: Simplified structure, merged `.context/` into `00_index/`, removed INDEX.md (human navigation not needed for AI-first system)
- **2025-11-17**: Created core module cards (yijinjing, wingchun, binance_extension) and interaction flows (trading_flow, event_flow)
- **2025-11-17**: Initial layered architecture established
