# Context Engineering Design

This document describes the context management system design for this project.

## Design Principles

Based on Linux kernel documentation principles:

1. **Code is Truth**: Documentation must match code reality
2. **Flat Structure**: Avoid unnecessary nesting, keep it simple
3. **DRY**: Single source of truth, zero duplication
4. **Concise**: No emoji, no fluff, high information density
5. **Structured**: Machine-readable metadata (YAML)
6. **Testable**: Commands in docs are automatically verifiable

## Project Identity

**Code Reality**:
- Project name in code: `kungfu`
- CMakeLists.txt: `PROJECT(kungfu)`
- Python package: `kungfu`
- CLI command: `kfc`

**Repository Identity**:
- Repository name: `godzilla-evan`
- Fork of: kungfu trading framework
- Modified by: godzilla.dev team

**Truth**: This is a fork. We acknowledge it transparently in ORIGIN.md.

## Documentation Structure

```
.doc/
├── INDEX.md              # Documentation index
├── ORIGIN.md             # Explain fork relationship
├── INSTALL.md            # Setup guide (merged from 5 files)
├── HACKING.md            # Development workflow
├── ARCHITECTURE.md       # System architecture
├── CHANGELOG.md          # Structured change history
├── .context/             # AI context management
│   ├── DESIGN.md         # This file
│   ├── index.yaml        # Document metadata
│   └── modules.yaml      # Context module definitions
├── adr/                  # Architecture Decision Records
│   ├── 001-docker.md
│   ├── 002-wsl2.md
│   └── 003-dns.md
├── modules/              # Module docs (optional)
│   ├── yijinjing.md
│   └── wingchun.md
└── scripts/              # Doc-related scripts
    └── verify-commands.sh
```

## Why This Structure?

### Centralization in .doc/
All documentation lives in `.doc/` directory:
- Clear separation from code
- Easy to find
- Clean root directory

### Flat Within .doc/
No `guide/`, `setup/` subdirectories:
- Avoids unnecessary nesting
- Easier navigation
- Clear file names eliminate need for categorization

### .context/ for Metadata
Machine-readable context management:
- `index.yaml`: Document metadata, dependencies, token counts
- `modules.yaml`: Context loading strategies
- `DESIGN.md`: Design principles (this file)

### adr/ for Decisions
Architecture Decision Records:
- Why we use Docker
- Why we use WSL2
- How we solve DNS issues
- Permanent record of rationale

## Document Responsibilities

### ORIGIN.md
- Explain this is a kungfu fork
- List modifications
- License information
- Upstream tracking status

### INDEX.md
- Entry point for all documentation
- Quick navigation
- Replaces old verbose README.md

### INSTALL.md
Single source for setup:
- Merges: quickstart + development-guide + dns-fix
- System requirements
- Docker + WSL2 setup
- DNS troubleshooting
- References ADR-003

### HACKING.md
Development workflow:
- Code structure (yijinjing, wingchun)
- Build process
- Testing
- Commit conventions
- References ADR-001, ADR-002

### ARCHITECTURE.md
System design:
- Overall architecture
- yijinjing event system
- wingchun trading gateway
- Data flow
- Performance considerations

## Context Modules

Different scenarios load different document sets:

### onboarding
For new developers:
- ORIGIN.md
- INDEX.md (via README.md)
- INSTALL.md
- ~3300 tokens

### development
For daily coding:
- HACKING.md
- ARCHITECTURE.md
- Depends on: onboarding
- ~4000 tokens

### deep_dive
For architectural work:
- ARCHITECTURE.md
- modules/*.md
- ~6000 tokens

### troubleshooting
For fixing issues:
- INSTALL.md (troubleshooting section)
- ADR-003 (DNS)
- Standalone, no dependencies

## AI Model Usage

### Loading Strategy

1. **Always load**: `.context/index.yaml` (metadata)
2. **On demand**: Load modules based on user query
3. **Optimize**: Use token estimates to fit context

### Example Queries

"How do I set up the environment?"
→ Load: `onboarding` module

"How does the event system work?"
→ Load: `development` + `deep_dive` modules

"DNS resolution failing"
→ Load: `troubleshooting` module + ADR-003

### Token Budget

Each document has estimated tokens:
- Helps AI decide what to load
- Prevents context overflow
- Enables smart caching

## Deduplication Strategy

### Before (Old Docs)
- Docker setup: appeared in 5 files
- Common commands: appeared in 4 files
- DNS fix: appeared in 3 files
- Total: ~2000 lines, 60% duplication

### After (New Docs)
- Docker setup: only in INSTALL.md
- Common commands: only in HACKING.md
- DNS fix: only in INSTALL.md + ADR-003
- Total: ~5000 lines, 0% duplication, 3x information density

## Testing Strategy

### Automated Verification
`scripts/verify-commands.sh`:
- Extract all code blocks from docs
- Execute commands in test environment
- Fail if any command doesn't work

### Link Validation
- Check all internal links
- Verify file references
- Ensure ADR cross-references are valid

### Freshness Check
- Last verified date in metadata
- CI runs verification weekly
- Alert if docs not verified in 30 days

## Deprecation Strategy

Old documents:
- `.doc/guide/*` → merged into INSTALL.md, HACKING.md
- `.doc/setup/*` → merged into INSTALL.md
- `.doc/PROJECT_ORGANIZATION.md` → replaced by this file

Process:
1. Create new consolidated docs
2. Delete old redundant files
3. Update index.yaml with `replaces:` field
4. No deprecation notices needed (files deleted)

## Maintenance Rules

### DO
- Update metadata when adding docs
- Test all commands before committing
- Link to ADRs for decisions
- Keep documents focused (single responsibility)
- Use YAML for machine-readable data

### DON'T
- Duplicate content across files
- Use emoji or excessive formatting
- Create deep directory hierarchies
- Let docs drift from code reality
- Forget to update index.yaml

## Future Extensions

When needed:
1. API documentation (Doxygen/Sphinx)
2. Performance tuning guide
3. Deployment playbooks
4. CI/CD integration docs

Add as:
- `.doc/api/` for API references
- `.doc/performance.md` for tuning
- `.doc/deploy.md` for deployment

Update `index.yaml` and `modules.yaml` accordingly.

## Version History

- 2025-10-22: Initial design
- Format: Keep this section updated with major changes

## References

- Linux kernel Documentation/
- Architecture Decision Records (ADR) pattern
- Documentation-as-Code principles

