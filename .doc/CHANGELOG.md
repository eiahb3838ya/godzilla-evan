# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

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

