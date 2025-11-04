# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Binance Market Toggle Feature - 2025-11-04

#### Added
- **ADR-004 Implementation**: Binance Market Toggle configuration
  - `enable_spot` and `enable_futures` flags in account config
  - Conditional initialization of Spot/Futures REST and WebSocket clients
  - Guard clauses in constructor, `on_start()`, and `_check_status()`
  - Backward compatible (defaults to `true` for both markets)
- **DEBUGGING.md Case 3**: Two-database path confusion issue
  - Documents critical containerization pitfall
  - Root cause: `KF_HOME` environment variable vs hardcoded paths
  - Solution: Respect container environment variables
  - Best practices for path handling in Docker

#### Changed
- `core/extensions/binance/include/common.h`: Added market toggle flags to Configuration struct
- `core/extensions/binance/src/trader_binance.cpp`: Conditional market initialization logic

#### Fixed
- **Critical**: Two separate database files causing "ghost account" issues
  - Wrong DB: `/root/.config/kungfu/app/system/etc/kungfu/db/live/accounts.db`
  - Correct DB: `/app/runtime/system/etc/kungfu/db/live/accounts.db` (KF_HOME)
  - Resolution: Deleted wrong DB, all operations now use `$KF_HOME`
- Futures-only API keys no longer generate Spot login errors
- TD Gateway logs are clean when Spot market is disabled

#### Lessons Learned
1. **Never hardcode paths** in containerized environments - use environment variables
2. **Always verify database path** before operations (print `db_path` in scripts)
3. **SQLAlchemy Json column updates** may not persist - use direct SQLite for reliability
4. **Container environment variables** (`docker-compose.yml`) take precedence over defaults
5. **Test data flow end-to-end** when debugging configuration issues (Python → DB → C++)
6. **Never manually create database tables** - let ORM manage schema (manual SQL caused schema mismatch)

---

### Binance Testnet Integration & Documentation - 2025-10-28

#### Added
- **TESTNET.md** complete rewrite with step-by-step troubleshooting
  - PM2 installation guide (now required for official scripts)
  - Manual database creation method for Docker environments
  - Complete restart procedure with graceful shutdown
  - Comprehensive troubleshooting section for 6+ common errors
- **scripts/binance_test/graceful_shutdown.sh** - Automated cleanup script
  - Stops all PM2 processes gracefully
  - Cleans journal files (prevents crashes on restart)
  - Cleans socket files (`*.nn`, `*.sock`)
  - Removes old logs (7+ days)
- **DEBUGGING.md Case 2** - PM2 + Database Configuration deep dive
  - Documents 5 chained errors during system startup
  - Root cause analysis for each error
  - Step-by-step diagnostic process
  - Complete resolution with working configuration

#### Changed
- **INSTALL.md** updated to mention PM2 requirement for test scripts
- **INDEX.md** restructured with critical warnings at top
  - Emphasizes TESTNET.md for Binance users
  - Simplified navigation
- **Learning Plan** (.cursor/plans/) completely redesigned
  - Removed "Phase 0", merged into Phase 2.1
  - Condensed from 1304 lines → 252 lines
  - Incorporated all debugging lessons learned
  - Added detailed troubleshooting table

#### Fixed
- **Critical**: Documented `gz_user1` account name requirement (hardcoded in PM2 configs)
- **Critical**: Documented PM2 installation (not pre-installed in container)
- **Critical**: Documented journal file cleanup requirement
- Python symlink issue (`python` vs `python3`)
- InstrumentType mismatch (Spot vs FFuture)

#### Lessons Learned
1. PM2 is essential for official scripts but not pre-installed
2. Account name MUST be `gz_user1` (not email or custom names)
3. Journal files cause registration conflicts if not cleaned
4. Database creation may fail in non-TTY Docker environments
5. Strategy must match API type (Futures Testnet requires FFuture)

---

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

