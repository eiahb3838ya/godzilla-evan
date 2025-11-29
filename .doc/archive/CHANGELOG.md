---
title: Change History
updated_at: 2025-11-24
owner: core-dev
lang: en
tokens_estimate: 1800
layer: 00_index
tags: [changelog, history, versions, releases]
purpose: "All notable changes to this project documented"
---

# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Symbol Naming Convention Documentation - 2025-11-26

#### Added
- **New Documentation**: `.doc/40_config/symbol_naming_convention.md`
  - **Purpose**: Document critical symbol format requirements and prevent subscription/order failures
  - **Root Cause**: During debugging, discovered symbol format errors caused:
    1. `IndexError: list index out of range` when placing orders
    2. Silent subscription failure (strategy not receiving market data despite successful subscription)
    3. Required C++ rebuild to fix after config changes
  - **Content**:
    - Symbol format specification: `lowercase_base_underscore_quote` (e.g., `"btc_usdt"`)
    - Complete flow tracing: subscription → hash matching → event filtering → callback
    - Base/quote coin extraction mechanism ([book.py:122-123](../../core/python/kungfu/wingchun/book/book.py#L122-L123))
    - Subscription hash matching system ([common.h:354-365](../../core/cpp/wingchun/include/kungfu/wingchun/common.h#L354-L365))
    - Exchange format conversion ([type_convert_binance.h:111-121](../../core/extensions/binance/include/type_convert_binance.h#L111-L121))
    - Common errors with examples and fixes
    - Compilation dependency requirements
    - Token estimate: ~5,800 tokens

- **New Documentation**: `.doc/90_operations/debugging_guide.md`
  - **Purpose**: Systematic strategy debugging procedures based on real debugging experience
  - **Content**:
    - Problem 1: Strategy not receiving market data (detailed 6-step troubleshooting)
    - Problem 2: IndexError when placing orders
    - Problem 3: Strategy startup failures
    - Problem 4: Order status anomalies
    - Problem 5: Account connection failures
    - Log analysis techniques
    - Emergency reset procedures
    - Debugging checklist
    - Token estimate: ~6,500 tokens

#### Updated
- `.doc/10_modules/strategy_framework.md`
  - Fixed all symbol format examples from `"btcusdt"` to `"btc_usdt"` (lines 92, 211, 227, 238, 293)
  - Added **IMPORTANT - Symbol Format** section after subscription example (lines 217-220)
  - Added **CRITICAL - Symbol Field** section in Configuration (lines 298-304)
  - Emphasized format requirement with code references to `book.py` and subscription matching

- `.doc/40_config/config_usage_map.md`
  - Added entire **Strategy Configuration** section (lines 205-290)
  - Documented `symbol` field with complete usage locations
  - Explained why format matters (base/quote extraction, subscription matching, exchange conversion)
  - Added troubleshooting for both IndexError and silent subscription failure
  - Cross-referenced to symbol naming convention and debugging guide

#### Fixed
- **Critical Bug**: `strategies/demo_future/config.json` symbol format
  - Changed from `"symbol": "btcusdt"` to `"symbol": "btc_usdt"`
  - Required C++ rebuild to clear subscription hash cache
  - Strategy now successfully receives depth events and places orders

#### Debugging Session Highlights
- **Issue**: demo_future strategy subscribed successfully but `on_depth()` never called
- **Investigation**:
  - Traced through `runner.cpp:72` → `is_subscribed()` → `get_symbol_id()` hash matching
  - Discovered symbol hash mismatch: `hash("btcusdt") ≠ hash("btc_usdt")`
  - MD gateway correctly published with `"btc_usdt"`, but strategy subscribed to `"btcusdt"`
- **Resolution**:
  - Fixed config.json symbol format
  - Rebuilt C++ modules to clear compiled cache
  - Strategy immediately started receiving depth events
- **Lessons Learned**:
  - Symbol format errors cause silent failures (no error messages)
  - Config changes affecting hash matching require C++ rebuild
  - Hash-based subscription filtering happens at C++ level, invisible to Python

#### Code References
- `core/python/kungfu/wingchun/book/book.py:122-123` - Base/quote extraction via `split("_")`
- `core/cpp/wingchun/include/kungfu/wingchun/common.h:354-365` - Symbol hash generation
- `core/cpp/wingchun/include/kungfu/wingchun/strategy/context.h:162-171` - Subscription matching
- `core/cpp/wingchun/src/strategy/runner.cpp:68-76` - Event filtering logic
- `core/extensions/binance/include/type_convert_binance.h:111-121` - Exchange format conversion
- `core/extensions/binance/src/marketdata_binance.cpp:148` - Depth event publishing with original symbol

#### Impact
- **Documentation Completeness**: +2 new comprehensive guides (~12,300 tokens)
- **Accuracy**: Fixed misleading examples in 3 existing documents
- **Developer Experience**: Future symbol format errors will be quickly identified via docs
- **Bug Prevention**: Explicit format requirements prevent common pitfalls

---

### Account Naming Convention Documentation - 2025-11-24

#### Added
- **New Documentation**: `.doc/40_config/account_naming_convention.md`
  - **Purpose**: Clarify the dual naming system for accounts (database format vs runtime format)
  - **Root Cause**: Users encountered `invalid account` errors due to confusion about when to use `binance_gz_user1` vs `gz_user1`
  - **Content**:
    - Complete flow tracing: account creation → TD gateway → strategy
    - Database format (`account_id`): `{source}_{account}` (e.g., `binance_gz_user1`)
    - Runtime format (`account`): pure account name (e.g., `gz_user1`)
    - Common errors and corrections
    - Code references: `add.py:18`, `td.py:22-23`, `trader.cpp:27`, `context.cpp:100-103`

#### Updated
- `.doc/40_config/config_usage_map.md`
  - Added warning section: "⚠️ 重要：帳號命名機制"
  - Explained automatic prefix addition during `kfc account add`
  - Linked to new `account_naming_convention.md`

- `.doc/90_operations/cli_operations_guide.md`
  - Added "帳號命名邏輯" section in troubleshooting
  - Clarified when to use pure account name vs database format
  - Updated code references to include `td.py:22-23`

### Configuration Documentation Correction - 2025-11-21

#### Fixed
- **Critical Documentation Errors**: Corrected misleading account configuration information across multiple docs
  - **Root Cause**: Multiple documentation files incorrectly stated that account configuration is stored in JSON files at `~/.config/kungfu/app/config/td/binance/<account>.json`
  - **Actual Implementation**: Account configuration is stored in SQLite database at `runtime/system/etc/kungfu/db/live/accounts.db` (table: `account_config`)
  - **Correct Method**: Use CLI command `kfc account -s binance add` for interactive configuration

#### Updated Files
- `.doc/40_config/config_usage_map.md`
  - Replaced "How to Update Configuration" section (lines 232-285)
  - Method 1: CLI command with `kfc account -s binance add`
  - Method 2: Python script for database queries and updates
  - Method 3: Direct SQLite operations for advanced users
  - Added code references: `models.py:23-28`, `add.py:15-25`, `data_proxy.py:80-82`

- `.doc/90_operations/cli_operations_guide.md`
  - Replaced "Account Config Not Found" troubleshooting section (lines 613-631)
  - Removed incorrect JSON file creation instructions
  - Added correct CLI and database methods
  - Updated front-matter with account-related code references

- `.doc/90_operations/pm2_startup_guide.md`
  - Fixed "Verify account configuration" section (line 509)
  - Replaced `cat ~/.config/kungfu/app/config/td/binance/my_account.json` with Python database query
  - Updated troubleshooting comments (lines 471-473)

- `.doc/START.md`
  - Corrected configuration location information (lines 144-147)
  - Changed from JSON file path to database path and CLI method

#### Added
- **Demo Future Strategy**: Complete Binance Futures trading strategy setup
  - Created `strategies/demo_future/` directory with strategy, config, and documentation
  - Created `scripts/demo_future/` with PM2 launch scripts (master, ledger, md, td, strategy)
  - Full documentation in `strategies/demo_future/README.md` and `scripts/demo_future/README.md`
  - Demonstrates Futures trading with index price subscription and order management

#### Code References
- `core/python/kungfu/data/sqlite/models.py:23-28` - Account table definition
- `core/python/kungfu/command/account/add.py:15-25` - CLI account addition
- `core/python/kungfu/data/sqlite/data_proxy.py:80-82` - Database write method
- `core/extensions/binance/package.json:8-44` - Configuration schema

#### Verified
- Configuration exists only in SQLite database (no JSON files)
- Existing account `gz_user1` has correct config: `enable_spot=false`, `enable_futures=true`
- All documentation now reflects correct configuration storage mechanism

---

### Documentation Enhancements - 2025-11-19

#### Added
- **Demo Strategy: market_data_demo**
  - Comprehensive market data subscription example (Depth, Ticker, Trade, IndexPrice)
  - Educational reference for learning path Stage 4
  - Located in `strategies/market_data_demo/` with full PM2 setup in `scripts/market_data_demo/`
  - Demonstrates proper lifecycle management and statistics tracking

#### Updated
- **Binance Extension Documentation** (`.doc/10_modules/binance_extension.md`)
  - Added "Trade Stream Design: Spot vs Futures" section
  - Documents Spot using raw `trade` stream vs Futures using `aggTrade` (aggregated)
  - Explains performance implications (97.5% data reduction for BTC/USDT Futures)
  - Includes code references and strategy use case recommendations

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

#### Verified
- Services started via PM2 and all online (master, ledger, md_binance, td_binance:gz_user1)
- TD logs confirm Futures-only with Spot disabled by configuration
- No Spot `-2015` errors when using Futures-only keys
- TD runtime log path: `/app/runtime/td/binance/gz_user1/log/live/gz_user1.log`

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
- `00_index/` directory for AI context management
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
- `.doc/PROJECT_ORGANIZATION.md` (replaced by `00_index/DESIGN.md`)

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

Last Updated: 2025-11-05

