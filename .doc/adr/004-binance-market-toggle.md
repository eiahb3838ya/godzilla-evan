# ADR-004: Binance Market Toggle Configuration

Date: 2025-11-03

## Status

Implemented ✅ (2025-11-04)

## Context

### Problem

When using Binance Futures Testnet with a Futures-only API key, the TD Gateway repeatedly attempts to log into Spot market, generating continuous error logs:

```
[error] spot login failed, error_id: -2015, error_msg: Invalid API-key, IP, or permissions for action.
```

These errors occur every 5 seconds during reconnection checks, creating log noise that obscures actual issues. While this is documented as expected behavior (see `doc/TESTNET.md` line 797-798), it degrades developer experience.

### Current Behavior

From `trader_binance.cpp:88`:
```cpp
_start_userdata(InstrumentType::FFuture);  // Only Futures on startup
```

However, `_check_status()` (line 342-348) attempts both Spot and Futures reconnection unconditionally:
```cpp
if (ws_ptr_->fetch_reconnect_flag()) {
    _start_userdata(InstrumentType::Spot);  // Always attempts
}
```

### Design Constraints

1. **Backward Compatibility**: Existing configurations must continue working without modification
2. **Linus Principles**: 
   - Minimal invasiveness (change only what's necessary)
   - Data-driven (configuration, not code changes)
   - Keep it simple (avoid over-engineering)
3. **Single Responsibility**: Only control market initialization, not authentication or URL management
4. **Non-Goals**: 
   - Supporting different API keys per market (out of scope)
   - Dynamic URL configuration (already hardcoded in `common.h`)
   - Runtime market switching (requires restart)

## Decision

Add optional boolean flags `enable_spot` and `enable_futures` to the Binance configuration JSON to control market initialization.

### Implementation Strategy

**Configuration Extension (Data Layer)**:
```cpp
// common.h
struct Configuration {
    // Existing fields
    std::string user_id;
    std::string access_key;
    std::string secret_key;
    
    // New fields (defaults maintain backward compatibility)
    bool enable_spot = true;
    bool enable_futures = true;
    
    // Existing host/port fields unchanged...
};
```

**JSON Parsing (Backward Compatible)**:
```cpp
inline void from_json(const nlohmann::json &j, Configuration &c) {
    j.at("user_id").get_to(c.user_id);
    j.at("access_key").get_to(c.access_key);
    j.at("secret_key").get_to(c.secret_key);
    
    // Optional flags with safe defaults
    c.enable_spot = j.value("enable_spot", true);
    c.enable_futures = j.value("enable_futures", true);
    
    // Existing hardcoded URL initialization unchanged...
}
```

**Initialization Guards (3 Touch Points)**:

1. **Constructor** (`trader_binance.cpp:40-53`): Guard REST client creation
2. **on_start()** (`trader_binance.cpp:88`): Guard `_start_userdata()` calls
3. **_check_status()** (`trader_binance.cpp:342-348`): Guard reconnection attempts

## Consequences

### Positive

1. **Clean Logs**: Developers using Futures-only can disable Spot, eliminating `-2015` errors
2. **Zero Breaking Changes**: Default `true` values preserve existing behavior
3. **Minimal Code Impact**: Only 4 files modified, ~20 lines added
4. **Data-Driven**: Runtime behavior controlled by database configuration, no recompilation
5. **Future-Proof**: Framework supports disabling either market independently

### Negative

1. **Partial Solution**: Does not address different API keys per market (by design)
2. **Constructor Complexity**: Conditional REST client initialization adds slight complexity
3. **Documentation Burden**: Must document new flags in TESTNET.md and example configs

### Alternatives Considered

**Alternative 1: Suppress Error Logs**
- Change `SPDLOG_ERROR` to `SPDLOG_DEBUG` when login fails
- **Rejected**: Hides legitimate errors, violates fail-fast principle

**Alternative 2: Separate API Keys per Market**
- Add `spot_access_key`, `futures_access_key` fields
- **Rejected**: Requires REST client refactoring (accepts keys in constructor), scope creep

**Alternative 3: Environment Variables**
- Use `BINANCE_ENABLE_SPOT=false`
- **Rejected**: Configuration should live in database, not environment

## Implementation Details

### File Modifications

#### 0. `core/extensions/binance/package.json`

**Location**: Lines 8-30 (config array)

**Add two new config items** (after secret_key):
```json
{
    "key": "enable_spot",
    "name": "启用现货交易",
    "type": "bool",
    "errMsg": "是否启用现货市场登录？(true/false)",
    "required": false
},
{
    "key": "enable_futures",
    "name": "启用期货交易",
    "type": "bool",
    "errMsg": "是否启用期货市场登录？(true/false)",
    "required": false
}
```

**Behavior**:
- User inputs `true` or `false` as text
- `encrypt()` converts to Python boolean automatically (line 66-67 in `__init__.py`)
- Missing fields default to `true` in C++ layer (backward compatible)

#### 1. `core/extensions/binance/include/common.h`

**Location**: Lines 18-35 (Configuration struct)

**Changes**:
```cpp
struct Configuration {
    std::string user_id;
    std::string access_key;
    std::string secret_key;
    
    // Add these two lines
    bool enable_spot = true;
    bool enable_futures = true;
    
    std::string spot_rest_host;
    // ... rest unchanged
};
```

**Location**: Lines 37-60 (from_json function)

**Changes**:
```cpp
inline void from_json(const nlohmann::json &j, Configuration &c) {
    j.at("user_id").get_to(c.user_id);
    j.at("access_key").get_to(c.access_key);
    j.at("secret_key").get_to(c.secret_key);
    
    // Add these two lines
    c.enable_spot = j.value("enable_spot", true);
    c.enable_futures = j.value("enable_futures", true);
    
    // Hardcoded URLs remain unchanged...
}
```

#### 2. `core/extensions/binance/src/trader_binance.cpp`

**Location**: Lines 40-53 (Constructor)

**Current**:
```cpp
rest_ptr_ = std::make_shared<binapi::rest::api>(...);
frest_ptr_ = std::make_shared<binapi::rest::api>(...);
```

**Updated**:
```cpp
if (config_.enable_spot) {
    rest_ptr_ = std::make_shared<binapi::rest::api>(...);
} else {
    SPDLOG_INFO("Spot market disabled by configuration");
}

if (config_.enable_futures) {
    frest_ptr_ = std::make_shared<binapi::rest::api>(...);
} else {
    SPDLOG_INFO("Futures market disabled by configuration");
}
```

**Location**: Lines 69-92 (on_start)

**Current**:
```cpp
_start_userdata(InstrumentType::FFuture);
```

**Updated**:
```cpp
if (config_.enable_futures && frest_ptr_) {
    _start_userdata(InstrumentType::FFuture);
} else {
    SPDLOG_INFO("Skipping Futures initialization (disabled or client unavailable)");
}

if (config_.enable_spot && rest_ptr_) {
    _start_userdata(InstrumentType::Spot);
} else {
    SPDLOG_INFO("Skipping Spot initialization (disabled or client unavailable)");
}
```

**Location**: Lines 342-348 (_check_status)

**Current**:
```cpp
if (ws_ptr_->fetch_reconnect_flag()) {
    _start_userdata(InstrumentType::Spot);
}
if (fws_ptr_->fetch_reconnect_flag()) {
    _start_userdata(InstrumentType::FFuture);
}
```

**Updated**:
```cpp
if (config_.enable_spot && rest_ptr_ && ws_ptr_->fetch_reconnect_flag()) {
    _start_userdata(InstrumentType::Spot);
}
if (config_.enable_futures && frest_ptr_ && fws_ptr_->fetch_reconnect_flag()) {
    _start_userdata(InstrumentType::FFuture);
}
```

#### 3. Account Configuration (CLI Only)

**Interactive CLI (Recommended, default)**
```bash
python core/python/dev_run.py account -s binance add
# Follow prompts:
# - user_id: gz_user1
# - access_key: YOUR_KEY
# - secret_key: YOUR_SECRET
# - 是否启用现货市场登录？(true/false): false
# - 是否启用期货市场登录？(true/false): true
```

> Note: Direct database updates are deprecated (can cause schema/path drift). Always use the interactive CLI.

#### 4. Documentation Updates

**File**: `doc/TESTNET.md`

**Location**: After "Code Behavior" section (around line 800)

**Add Section**:
```markdown
### Disabling Spot Login (Reduce Log Noise)

If using Futures-only API keys, you can disable Spot login attempts to eliminate `-2015` errors:

**Update Database Configuration**:
```sql
UPDATE account_config 
SET config = json_set(config, '$.enable_spot', false)
WHERE user_id = 'gz_user1';
```

**Or Create New Account**:
```python
import sqlite3, json
conn = sqlite3.connect('/root/.config/kungfu/app/kungfu.db')
cursor = conn.cursor()
config = {
    'access_key': 'YOUR_FUTURES_KEY',
    'secret_key': 'YOUR_FUTURES_SECRET',
    'enable_spot': false,
    'enable_futures': true
}
cursor.execute('INSERT OR REPLACE INTO account_config VALUES (?, ?, ?, ?)',
               ('gz_user1', 'binance', 1, json.dumps(config)))
conn.commit()
```

**Verify Configuration**:
```bash
pm2 restart td_binance:gz_user1
pm2 logs td_binance:gz_user1 --lines 10
# Should see: [info] Spot market disabled by configuration
```
```

### Testing Strategy

**Unit Tests** (C++ - if test framework available):
```cpp
TEST(BinanceConfig, DefaultBothEnabled) {
    json config = {{"user_id", "test"}, {"access_key", "key"}, {"secret_key", "sec"}};
    Configuration c = config.get<Configuration>();
    ASSERT_TRUE(c.enable_spot);
    ASSERT_TRUE(c.enable_futures);
}

TEST(BinanceConfig, DisableSpot) {
    json config = {{"user_id", "test"}, {"access_key", "key"}, {"secret_key", "sec"}, {"enable_spot", false}};
    Configuration c = config.get<Configuration>();
    ASSERT_FALSE(c.enable_spot);
    ASSERT_TRUE(c.enable_futures);
}
```

**Integration Tests**:
1. **Backward Compatibility**: Start TD with old config (no flags) → Both markets should initialize
2. **Spot Disabled**: Start TD with `enable_spot: false` → No Spot REST client, no login attempts
3. **Futures Disabled**: Start TD with `enable_futures: false` → No Futures REST client
4. **Log Verification**: Check logs for "disabled by configuration" messages

**Manual Test Cases**:
```bash
# Case 1: Existing config (no flags)
python core/python/dev_run.py account -s binance show
# Should work as before

# Case 2: Disable Spot
sqlite3 /root/.config/kungfu/app/kungfu.db
UPDATE account_config SET config = json_set(config, '$.enable_spot', false);
.quit
pm2 restart td_binance:gz_user1
pm2 logs --lines 20
# Expected: [info] Spot market disabled by configuration
# Expected: No "-2015" errors for Spot

# Case 3: Re-enable
UPDATE account_config SET config = json_remove(config, '$.enable_spot');
pm2 restart td_binance:gz_user1
# Expected: Both markets initialize
```

**Interactive CLI Test Cases**:
```bash
# Test Case 1: Futures-Only Account (Interactive)
python core/python/dev_run.py account -s binance add
# 输入:
#   user_id: gz_user1
#   access_key: YOUR_FUTURES_KEY
#   secret_key: YOUR_FUTURES_SECRET
#   是否启用现货市场登录？(true/false): false
#   是否启用期货市场登录？(true/false): true

# 验证数据库
docker exec godzilla-dev sqlite3 /app/runtime/system/etc/kungfu/db/live/accounts.db \
    'SELECT config FROM account_config WHERE account_id="binance_gz_user1"' | \
python3 -m json.tool
# 预期输出包含: "enable_spot": false, "enable_futures": true

# 启动 TD Gateway
pm2 start ecosystem.config.js --only td_binance

# 检查日志
pm2 logs td_binance --lines 20 | grep -E '(Spot|Futures|disabled|enabled)'
# 预期输出:
# [info] Spot market disabled by configuration
# [info] Futures market initialized

# Test Case 2: 默认行为（向后兼容）
# 删除账户，重新创建，对 enable_spot/enable_futures 问题直接按回车（使用默认值）
# 验证 DB 中字段为 true（或不存在，C++ 使用默认值）
# 验证日志中两个市场都尝试初始化
```

## Design Patterns Applied

1. **Strategy Pattern**: Configuration determines runtime behavior without code changes
2. **Guard Clause Pattern**: Early returns via boolean checks reduce nesting
3. **Fail-Safe Defaults**: `true` defaults prevent breaking existing deployments
4. **Open-Closed Principle**: Extended configuration without modifying existing struct layout

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Null pointer dereference if REST clients not created | Low | High | Always check `rest_ptr_` / `frest_ptr_` existence before use |
| User misconfigures and disables both markets | Low | Medium | Document that at least one market should be enabled |
| Breaking change for users with custom tooling | Very Low | Low | Defaults ensure zero changes needed |

## References

- Original Issue: Spot login failures documented in `doc/TESTNET.md:797-798`
- Debugging Case: `doc/DEBUGGING.md` Case 1 - TD Gateway startup issues
- Configuration Parsing: `core/extensions/binance/include/common.h:37-60`
- TD Initialization: `core/extensions/binance/src/trader_binance.cpp:69-92`
- Binance REST API: Uses hardcoded URLs in constructor (lines 40-53)

## Approval

- [ ] Technical Review
- [ ] Code Review
- [ ] Documentation Review
- [ ] Testing Complete
- [ ] Deployed to Development

---

## Implementation Notes

### Database Schema Correction (2025-11-04)

**Problem:** Documentation contained incorrect manual SQL table creation using `user_id` as primary key, while code expects `account_id`.

**Root Cause:**  
Historical mismatch between documentation examples and actual SQLAlchemy Model definition in `models.py`:
- Documentation (TESTNET.md, DEBUGGING.md, learning plan): Used `user_id TEXT NOT NULL` with composite key `(user_id, source_name)`
- Code (`core/python/kungfu/data/sqlite/models.py`): Defined `account_id VARCHAR` as primary key

This caused TD startup failures with "JSON parse error" because `get_td_account_config()` query failed when column names didn't match.

**Solution:**  
- Removed all manual SQL table creation from documentation
- Updated all docs to use official interactive command: `python core/python/dev_run.py account -s binance add`
- This ensures table structure matches SQLAlchemy Model definition automatically

**Test-Driven Fix (TDD Approach):**
1. **RED Phase**: Created `test_db_schema_fix.sh` with 4 tests (all failed, confirming problem exists)
2. **GREEN Phase**: Fixed database by dropping old table, recreating with correct schema using SQLAlchemy
3. **REFACTOR Phase**: Updated all documentation to prevent future occurrences

**Test Results:**
- ✅ Schema test: Table now contains `account_id` column
- ✅ Config read test: SQLAlchemy can successfully query account config
- ✅ TD startup test: No JSON parse errors
- ✅ Market toggle config: Configuration successfully added to database

**Files Modified:**
- `core/extensions/binance/test/test_db_schema_fix.sh` (new regression test)
- `doc/TESTNET.md` (2 instances fixed)
- `doc/DEBUGGING.md` (3 instances fixed)
- `doc/quantitative-trading-learning-path.plan.md` (1 instance fixed)

**Impact:** Fixes TD startup failures and ensures all future users follow correct account creation method.

---

## Implementation Summary (2025-11-04)

### What We Built

Successfully implemented the Binance Market Toggle feature with full backward compatibility.

**Verified Functionality:**
- ✅ Spot disabled, Futures enabled: TD Gateway logs show "Spot market disabled by configuration"
- ✅ No Spot login errors when Futures-only API key is used
- ✅ All services (master, ledger, md_binance, td_binance) running stable
- ✅ Configuration persists in database and correctly propagates to C++ layer

### Critical Bug Discovered & Fixed

**The Two-Database Problem:**

During implementation testing, we discovered a critical infrastructure issue where two separate SQLite database files existed:
- `/root/.config/kungfu/app/system/etc/kungfu/db/live/accounts.db` (wrong)
- `/app/runtime/system/etc/kungfu/db/live/accounts.db` (correct, used by KF_HOME)

**Root Cause:**
- `docker-compose.yml` sets `KF_HOME=/app/runtime` (container environment)
- Manual scripts hardcoded `/root/.config/kungfu/app` paths
- Operations targeted different databases, causing "ghost account" phenomena

**Impact:**
- Account creation appeared to fail ("Duplicate account") despite empty database queries
- Configuration updates silently failed (written to wrong database)
- Delete operations had no effect
- Hours wasted debugging "impossible" behavior

**Resolution:**
1. Deleted the erroneous database path
2. All scripts now respect `$KF_HOME` environment variable
3. Configuration updates verified using direct SQLite queries
4. Documented in DEBUGGING.md Case 3

**Lesson:** Never hardcode paths in containerized environments. Always use environment variables.

### The Manual Database Creation Anti-Pattern

**Additional Discovery:** Manual SQL table creation was a significant design flaw that caused extensive detours.

**The Problem:**
- SQLAlchemy Model defines `account_id` as primary key (`models.py:25`)
- Manual SQL in documentation used `user_id` as primary key
- Schema mismatch caused: `sqlite3.OperationalError: no such column: account_id`

**Impact:**
- Hours wasted debugging schema inconsistencies
- Risk of missing fields or incorrect types
- Maintenance nightmare (Model changes don't sync with manual SQL)
- False assumption that database structure was correct

**Root Cause:**
Documentation showed manual table creation instead of using SQLAlchemy's built-in mechanism:
```sql
-- ❌ WRONG: Manual SQL (prone to errors)
CREATE TABLE account_config (
    user_id TEXT PRIMARY KEY,  -- ← Wrong column name!
    source_name TEXT,
    config TEXT
);
```

**Correct Approach:**
```python
# ✅ RIGHT: Let SQLAlchemy create tables from Model
from kungfu.data.sqlite.models import Base
Base.metadata.create_all(engine)
```

Or use the official interactive command:
```bash
python core/python/dev_run.py account -s binance add
```

**Why This Matters:**
1. **Schema Consistency**: Model is the single source of truth
2. **Type Safety**: `Json` column type handled correctly by SQLAlchemy
3. **Future-Proof**: Model changes automatically propagate
4. **Zero Manual Sync**: No need to update SQL when Model changes

**Resolution:**
- Removed all manual SQL table creation from documentation
- Updated all guides to use official `account add` command
- Created regression test to verify schema correctness

**Lesson:** Never manually create database tables when using an ORM. Let the ORM manage schema.

### Configuration Update Method

**Issue Found:** SQLAlchemy `session_scope` with `Json` column type does not reliably persist config updates.

**Working Solution:**
```bash
docker exec godzilla-dev bash -c "python3 << 'EOF'
import sqlite3, json
conn = sqlite3.connect('/app/runtime/system/etc/kungfu/db/live/accounts.db')
cursor = conn.cursor()
cursor.execute('SELECT config FROM account_config WHERE account_id=\"binance_gz_user1\"')
config = json.loads(cursor.fetchone()[0])
config['enable_spot'] = False
config['enable_futures'] = True
cursor.execute('UPDATE account_config SET config=? WHERE account_id=?', 
               (json.dumps(config), 'binance_gz_user1'))
conn.commit()
conn.close()
EOF
"
```

### Verification Commands

```bash
# Check TD logs for market toggle confirmation (PM2)
docker exec godzilla-dev pm2 logs td_binance:gz_user1 --lines 30 | grep -E '(Spot|Futures|disabled|enabled|login)'

# Or check the exact runtime log file
docker exec godzilla-dev tail -50 /app/runtime/td/binance/gz_user1/log/live/gz_user1.log | \
  grep -E '(Spot|Futures|disabled|enabled|login)'

# Expected output:
# [info] Spot market disabled by configuration
# [info] Connecting BINANCE TD for gz_user1 (Spot: disabled, Futures: enabled)
# [info] Skipping Spot initialization (disabled or client unavailable)
# [info] login success
```

### Files Actually Modified

**C++ Layer:**
- `core/extensions/binance/include/common.h` (lines 26-27, 51-52)
- `core/extensions/binance/src/trader_binance.cpp` (constructor, on_start, _check_status)

**Documentation:**
- `doc/DEBUGGING.md` (added Case 3: Two-Database Problem)
- `doc/adr/004-binance-market-toggle.md` (this file)

**No Changes Needed:**
- Python layer works correctly with existing code
- Database schema already supports arbitrary JSON config fields

---

## Current Status (2025-11-05)

- All core services started via official script and are online (master, ledger, md_binance, td_binance:gz_user1)
- TD confirms Futures-only login with Spot disabled by configuration
- No Spot "-2015" errors observed; Spot initialization skipped as expected
- TD log path verified: `/app/runtime/td/binance/gz_user1/log/live/gz_user1.log`

```
[info] Spot market disabled by configuration
[info] Connecting BINANCE TD for gz_user1 (Spot: disabled, Futures: enabled)
[info] Skipping Spot initialization (disabled or client unavailable)
[info] login success
```

Last Updated: 2025-11-04
