# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Architecture Documentation Update - 2025-10-23

#### Added
- **Detailed yijinjing architecture** in ARCHITECTURE.md:
  - Three-layer design: Frame (48-byte header) / Page (1-128MB) / Journal
  - Zero-copy design explanation with code examples
  - Intelligent page sizing strategy (MD=128MB, TD/STRATEGY=4MB)
  - Event system abstraction and implementation
  - Location system: mode/category/layout classification
  - Time system: nanosecond precision details
  - Publisher/Observer pattern for IPC
- **Code structure mapping**: Documentation to actual file paths
  - yijinjing: `core/cpp/yijinjing/` with detailed file tree
  - wingchun: `core/cpp/wingchun/` with module breakdown
  - Python layer: `core/python/kungfu/` structure
  - Extensions: `core/extensions/` organization
- **Statistics**: Code size (~15K lines total)

#### Changed
- Updated ARCHITECTURE.md token estimate: 2500 → 4200 tokens
- Rewrote yijinjing section with concrete technical details from code
- Changed from abstract descriptions to struct definitions and measurements

#### Rationale
- **Code is Truth**: All information extracted from actual source code
- **High information density**: Replaced vague descriptions with specifics
- **Verifiable**: All technical details cross-referenced with code

### Documentation Restructure - 2025-10-22

#### Added
- New documentation structure following Linux principles
- `.context/` directory for AI context management
  - `DESIGN.md`: Context engineering design principles
  - `index.yaml`: Document metadata and dependencies
  - `modules.yaml`: Context loading strategies
- Architecture Decision Records (ADRs)
  - `adr/001-docker.md`: Docker development environment decision
  - `adr/002-wsl2.md`: WSL2 backend decision
  - `adr/003-dns.md`: DNS resolution strategy
- Core documentation files
  - `ORIGIN.md`: Project fork history and identity
  - `INDEX.md`: Documentation navigation
  - `INSTALL.md`: Comprehensive setup guide
  - `HACKING.md`: Development workflow
  - `ARCHITECTURE.md`: System architecture
  - `CHANGELOG.md`: This file

#### Changed
- Simplified `README.md` to 80 lines (from 176)
- Removed emoji and excessive formatting from all docs
- Consolidated environment setup docs into single `INSTALL.md`
- Flattened `.doc/` structure (removed `guide/`, `setup/` subdirectories)

#### Removed
- `.doc/README.md` (replaced by `INDEX.md`)
- `.doc/guide/quickstart.md` (merged into `INSTALL.md`)
- `.doc/guide/development-guide.md` (split into `INSTALL.md` and `HACKING.md`)
- `.doc/guide/environment-status.md` (snapshot document, outdated)
- `.doc/setup/docker-dns-fix.md` (merged into `INSTALL.md` and `adr/003-dns.md`)
- `.doc/PROJECT_ORGANIZATION.md` (replaced by `.context/DESIGN.md`)

#### Rationale
- Implement Linux kernel documentation principles
- Eliminate duplicate content (was 60% duplicated)
- Optimize for AI/LLM context management
- Make documentation testable and verifiable
- Single source of truth for each topic

### Initial Fork - ~2025-03

#### Added
- Fork of kungfu trading framework
- Custom Docker development environment
- Docker Compose configuration for WSL2
- Initial project documentation

#### Modified
- Project renamed to "godzilla-evan"
- Custom modifications by godzilla.dev team

---

## Documentation Versioning

This changelog tracks significant documentation changes and project milestones.

For code changes, see git commit history:
```bash
git log --oneline
```

---

Last Updated: 2025-10-22

