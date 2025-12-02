---
title: CLI Operations Guide
updated_at: 2025-11-24
owner: operations
lang: en
tags: [operations, cli, commands, dev_run, kungfu, account-naming]
code_refs:
  - core/python/kungfu/command/master.py:1-21
  - core/python/kungfu/command/md.py:1-29
  - core/python/kungfu/command/td.py:1-35
  - core/python/kungfu/command/strategy.py:1-110
  - core/python/kungfu/command/ledger.py:1-25
  - core/python/kungfu/command/account/add.py:15-25
  - core/python/kungfu/data/sqlite/models.py:23-28
  - core/python/kungfu/data/sqlite/data_proxy.py:80-82
purpose: "Reference guide for command-line tools to run and manage trading system services"
tokens_estimate: 5000
---

# CLI Operations Guide

## Overview

The trading system provides a unified CLI interface via the `dev_run.py` script for running all system components. This guide documents all available commands, their options, and common usage patterns.

**Entry Point**: `core/python/dev_run.py`

**Usage Pattern**:
```bash
python3 dev_run.py [OPTIONS] COMMAND [COMMAND_OPTIONS]
```

## Global Options

Available for all commands:

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--log_level` | `-l` | Logging level | `info` |
| `--home` | `-h` | KungFu home directory | `~/.config/kungfu/app/runtime` |

**Log Levels**: `trace`, `debug`, `info`, `warning`, `error`

**Example**:
```bash
# Run master with debug logging
python3 dev_run.py -l debug master

# Use custom home directory
python3 dev_run.py -h /custom/path master
```

---

## Commands

### 1. master - Master Service

**Purpose**: Central coordinator and service registry

**Source**: [master.py:12-19](../../core/python/kungfu/command/master.py)

**Syntax**:
```bash
python3 dev_run.py [OPTIONS] master [MASTER_OPTIONS]
```

**Options**:

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--low_latency` | `-x` | Flag | Run in low latency mode |

**Examples**:
```bash
# Standard mode
python3 dev_run.py -l info master

# Low latency mode (uses busy-wait polling)
python3 dev_run.py -l info master -x
```

**What it does**:
- Starts master service on port 9000
- Maintains service registry (all components register with master)
- Provides service discovery for inter-component communication
- Monitors component heartbeats

**Runs as**: Daemon process, exits only on SIGINT

**PM2 Usage**:
```json
{
  "apps": [{
    "name": "master",
    "script": "dev_run.py",
    "args": "-l info master"
  }]
}
```

---

### 2. ledger - Ledger Service

**Purpose**: Account state tracking, position management, PnL calculation

**Source**: [ledger.py:12-24](../../core/python/kungfu/command/ledger.py)

**Syntax**:
```bash
python3 dev_run.py [OPTIONS] ledger [LEDGER_OPTIONS]
```

**Options**:

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--low_latency` | `-x` | Flag | Run in low latency mode |

**Examples**:
```bash
# Standard mode
python3 dev_run.py -l info ledger

# Low latency mode
python3 dev_run.py -l info ledger -x
```

**What it does**:
- Tracks all account positions and assets
- Validates order requests (balance checks, position checks)
- Calculates realized/unrealized PnL
- Routes orders to appropriate trading gateways
- Publishes broker states (connection status)

**Dependencies**: Requires `master` to be running

**Runs as**: Daemon process

---

### 3. md - Market Data Gateway

**Purpose**: Connect to exchange WebSocket, stream market data to journal

**Source**: [md.py:14-27](../../core/python/kungfu/command/md.py)

**Syntax**:
```bash
python3 dev_run.py [OPTIONS] md -s SOURCE [MD_OPTIONS]
```

**Required Options**:

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--source` | `-s` | Choice | Exchange identifier (e.g., `binance`) |

**Optional Options**:

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--account` | `-a` | String | Account identifier (for private streams) |
| `--low_latency` | `-x` | Flag | Run in low latency mode |

**Examples**:
```bash
# Public market data (depth, trades, tickers)
python3 dev_run.py -l trace md -s binance

# Private market data (user trades, position updates)
python3 dev_run.py -l info md -s binance -a my_account

# Low latency mode
python3 dev_run.py -l info md -s binance -x
```

**What it does**:
- Connects to exchange WebSocket API
- Subscribes to market data channels (requested by strategies)
- Parses WebSocket messages into standard data structures
- Writes market data events to journal (Depth, Ticker, Trade)
- Auto-reconnects on disconnection

**Available Sources**:
- `binance` - Binance spot/futures (testnet/mainnet)
- Others as registered in `EXTENSION_REGISTRY_MD`

**Dependencies**: Requires `master` and `ledger` to be running

**Configuration**: Reads from `~/.config/kungfu/app/config/md/<source>/config.json`

---

### 4. td - Trading Gateway

**Purpose**: Execute orders at exchange via REST API

**Source**: [td.py:14-34](../../core/python/kungfu/command/td.py)

**Syntax**:
```bash
python3 dev_run.py [OPTIONS] td -s SOURCE -a ACCOUNT [TD_OPTIONS]
```

**Required Options**:

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--source` | `-s` | Choice | Exchange identifier (e.g., `binance`) |
| `--account` | `-a` | String | Account identifier |

**Optional Options**:

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--low_latency` | `-x` | Flag | Run in low latency mode |

**Examples**:
```bash
# Connect to Binance with account "my_account"
python3 dev_run.py -l info td -s binance -a my_account

# Low latency mode
python3 dev_run.py -l info td -s binance -a my_account -x
```

**What it does**:
- Connects to exchange REST API
- Reads order requests from journal (OrderInput events)
- Executes orders via exchange API (POST /order, DELETE /order)
- Writes order confirmations to journal (Order events)
- Writes trade fills to journal (MyTrade events)
- Handles order queries, position queries, balance queries

**Available Sources**:
- `binance` - Binance spot/futures (testnet/mainnet)
- Others as registered in `EXTENSION_REGISTRY_TD`

**Dependencies**: Requires `master`, `ledger`, and `md` to be running

**Configuration**: Reads from `~/.config/kungfu/app/config/td/<source>/<account>.json`

**Critical**: Account name must match configuration file name

---

### 5. strategy - Strategy Runner

**Purpose**: Execute user-defined trading strategies

**Source**: [strategy.py:19-110](../../core/python/kungfu/command/strategy.py)

**Syntax**:
```bash
python3 dev_run.py [OPTIONS] strategy -n NAME -p PATH [STRATEGY_OPTIONS]
```

**Required Options**:

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--name` | `-n` | String | Strategy name (unique identifier) |
| `--path` | `-p` | String | Path to strategy Python file |

**Optional Options**:

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--group` | `-g` | String | Strategy group (default: `default`) |
| `--config` | `-c` | String | Path to config file (can be multiple) |
| `--low_latency` | `-x` | Flag | Run in low latency mode |
| `--replay` | `-r` | Flag | Run in replay mode |
| `--session_id` | `-i` | Int | Replay session ID (required with `-r`) |
| `--backtest` | `-t` | Flag | Run in backtest mode |
| `--begin_time` | `-b` | String | Backtest start time (nanoseconds) |
| `--end_time` | `-e` | String | Backtest end time (nanoseconds) |
| `--cancel` | `-d` | Flag | Cancel active orders on startup |
| `--symbol` | `-s` | String | Cancel orders for specific symbol |

**Examples**:

#### Live Trading (Production)
```bash
python3 dev_run.py -l info strategy \
  -n demo_spot \
  -p strategies/demo_spot.py
```

#### With Configuration File
```bash
python3 dev_run.py -l info strategy \
  -n demo_spot \
  -p strategies/demo_spot.py \
  -c config/demo_spot.json
```

**Config File** (`config/demo_spot.json`):
```json
{
  "symbols": ["btcusdt", "ethusdt"],
  "threshold": 0.02,
  "max_position": 1.0
}
```

**Access in Strategy**:
```python
def pre_start(self, context):
    config = context.get_config()
    symbols = config["symbols"]
    threshold = config["threshold"]
```

#### Multiple Configurations
```bash
# Run same strategy with 3 different configs
python3 dev_run.py -l info strategy \
  -n multi_config \
  -p strategies/strategy.py \
  -c config/config1.json \
  -c config/config2.json \
  -c config/config3.json
```

Creates 3 strategy instances, each with its own config.

#### Backtest Mode
```bash
python3 dev_run.py -l info strategy \
  -n backtest_demo \
  -p strategies/demo_spot.py \
  -t \
  -b 1699564800000000000 \
  -e 1699651200000000000
```

**Time Format**: Nanoseconds since epoch
- `1699564800000000000` = 2023-11-10 00:00:00 UTC
- Use: `date -d "2023-11-10" +%s%N` (Linux)

#### Replay Mode
```bash
# Replay session 12345
python3 dev_run.py -l info strategy \
  -n replay_demo \
  -p strategies/demo_spot.py \
  -r \
  -i 12345
```

Replays events from a previous session for debugging.

#### Cancel Orders on Startup
```bash
# Cancel all active orders
python3 dev_run.py -l info strategy \
  -n demo_spot \
  -p strategies/demo_spot.py \
  -d

# Cancel orders for specific symbol
python3 dev_run.py -l info strategy \
  -n demo_spot \
  -p strategies/demo_spot.py \
  -d -s btcusdt
```

**What it does**:
- Loads strategy Python module
- Creates strategy context (access to trading API)
- Calls strategy lifecycle callbacks (pre_start, on_depth, etc.)
- Subscribes to market data as requested by strategy
- Routes order events back to strategy
- Manages strategy state across event loop

**Dependencies**: Requires `master`, `ledger`, `md`, and `td` to be running

**Runs as**: Event-driven process, exits on SIGINT

---

## Common Workflows

### Workflow 1: Start Full System (Manual)

```bash
# Terminal 1: Master
python3 dev_run.py -l info master

# Wait 5 seconds

# Terminal 2: Ledger
python3 dev_run.py -l info ledger

# Wait 5 seconds

# Terminal 3: Market Data
python3 dev_run.py -l trace md -s binance

# Wait 5 seconds

# Terminal 4: Trading Gateway
python3 dev_run.py -l info td -s binance -a my_account

# Wait 5 seconds

# Terminal 5: Strategy
python3 dev_run.py -l info strategy \
  -n demo_spot \
  -p strategies/demo_spot.py
```

**Total Startup Time**: ~25 seconds

### Workflow 2: Start Full System (PM2)

See [PM2 Startup Guide](pm2_startup_guide.md) for automated startup using PM2.

### Workflow 3: Restart Single Component

```bash
# Find and kill process
ps aux | grep "dev_run.py.*md.*binance"
kill <pid>

# Restart
python3 dev_run.py -l trace md -s binance
```

**Note**: Other components continue running, only MD gateway restarts.

### Workflow 4: Backtest Historical Data

```bash
# Step 1: Ensure system is stopped (no live trading)
pm2 stop all

# Step 2: Run backtest
python3 dev_run.py -l info strategy \
  -n backtest_strategy \
  -p strategies/my_strategy.py \
  -t \
  -b 1699564800000000000 \
  -e 1699651200000000000

# Step 3: Review results
tail -f ~/.pm2/logs/backtest_strategy-out.log
```

**Backtest Requirements**:
- Historical market data must exist in journal
- Use `--low_latency` for faster execution
- Disable live connections (master/ledger/md/td not needed)

### Workflow 5: Debug Strategy (Verbose Logging)

```bash
# Run with trace-level logging
python3 dev_run.py -l trace strategy \
  -n debug_strategy \
  -p strategies/debug_strategy.py

# Output shows:
# - Every callback invocation
# - All market data events
# - Order state transitions
# - Journal reads/writes
```

**Warning**: Trace logging generates ~10 MB/min, use only for debugging.

---

## Environment Variables

### KF_HOME

**Purpose**: Override default home directory

```bash
export KF_HOME=/custom/path/kungfu
python3 dev_run.py -l info master
```

**Default**: `~/.config/kungfu/app/runtime`

**Directory Structure**:
```
$KF_HOME/
├── journal/          # Event journals (binary)
├── log/              # Application logs
├── config/           # Configuration files
│   ├── md/
│   │   └── binance/config.json
│   └── td/
│       └── binance/my_account.json
└── db/               # SQLite databases
```

### CLEAR_JOURNAL

**Purpose**: Delete old journal files on startup (development only)

```bash
export CLEAR_JOURNAL=1
python3 dev_run.py -l info master
```

**Warning**: Deletes ALL historical data. Use only in development/testing.

**Effect**:
```bash
# Executed on startup
find $KF_HOME/journal/ -name "*.journal" -delete
```

---

## Configuration Files

### Master Configuration

**Location**: None (master has no config file)

**Port**: Hardcoded to 9000

### Ledger Configuration

**Location**: None (ledger has no config file)

### Market Data Configuration

**Location**: `$KF_HOME/config/md/<source>/config.json`

**Example** (`md/binance/config.json`):
```json
{
  "user_id": "binance",
  "spot_rest_host": "testnet.binance.vision",
  "spot_wss_host": "stream.testnet.binance.vision"
}
```

### Trading Gateway Configuration

**Location**: `$KF_HOME/config/td/<source>/<account>.json`

**Example** (`td/binance/my_account.json`):
```json
{
  "user_id": "my_account",
  "access_key": "YOUR_API_KEY",
  "secret_key": "YOUR_SECRET_KEY",
  "enable_spot": true,
  "enable_futures": false
}
```

**Critical**: Keep secret keys secure, never commit to git.

---

## Troubleshooting

### Command Not Found

**Symptom**:
```bash
python3: can't open file 'dev_run.py': [Errno 2] No such file or directory
```

**Solution**:
```bash
# Navigate to correct directory
cd core/python
python3 dev_run.py -l info master
```

### Master Already Running

**Symptom**:
```
ERROR: Master service already running on port 9000
```

**Solution**:
```bash
# Find and kill existing master
ps aux | grep "dev_run.py.*master"
kill <pid>

# Or use PM2
pm2 stop master
pm2 delete master
```

### Import Error

**Symptom**:
```
ImportError: No module named 'kungfu'
```

**Solution**:
```bash
# Install kungfu package
cd core/cpp
cmake --build build --target install

# Or set PYTHONPATH
export PYTHONPATH=/path/to/core/python:$PYTHONPATH
```

### Account Config Not Found

**Symptom**:
```
ERROR: Account config not found for account: my_account
```

**Root Cause**: Account configuration is stored in SQLite database, not JSON files.

**Database Location**: `runtime/system/etc/kungfu/db/live/accounts.db`

**Solution Method 1 - CLI Command (Recommended)**:
```bash
# Add account using interactive CLI
kfc account -s binance add

# Follow prompts to enter:
# - user_id: my_account
# - access_key: YOUR_BINANCE_API_KEY
# - secret_key: YOUR_BINANCE_SECRET_KEY
# - enable_spot: true/false
# - enable_futures: true/false
```

**⚠️ 重要：帳號命名邏輯**

當你輸入 `user_id: my_account` 時，系統會自動加上 `{source}_` 前綴：
- 資料庫中的 `account_id`：`binance_my_account`
- TD gateway 參數：`-a my_account`（使用**純帳號名稱**）
- 策略配置：`"account": "my_account"`（使用**純帳號名稱**）

**內部邏輯** ([add.py:18](../../core/python/kungfu/command/account/add.py#L18))：
```python
account_id = ctx.source + '_' + answers[ctx.schema['key']]  # "binance_my_account"
```

詳細說明請參閱 [帳號命名機制](../40_config/NAMING_CONVENTIONS.md#一帳號命名規範)。

**Solution Method 2 - Python Script**:
```python
import sqlite3
import json

db_path = '/home/huyifan/projects/godzilla-evan/runtime/system/etc/kungfu/db/live/accounts.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

config = {
    'user_id': 'my_account',
    'access_key': 'YOUR_BINANCE_API_KEY',
    'secret_key': 'YOUR_BINANCE_SECRET_KEY',
    'enable_spot': True,
    'enable_futures': False
}

cursor.execute(
    "INSERT OR REPLACE INTO account_config (account_id, source_name, receive_md, config) VALUES (?, ?, ?, ?)",
    ('binance_my_account', 'binance', 1, json.dumps(config))
)
conn.commit()
conn.close()
```

**Verify Configuration**:
```python
import sqlite3
import json

db_path = '/home/huyifan/projects/godzilla-evan/runtime/system/etc/kungfu/db/live/accounts.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT config FROM account_config WHERE account_id = 'binance_my_account'")
result = cursor.fetchone()
if result:
    print(json.dumps(json.loads(result[0]), indent=2))
else:
    print("Account not found")

conn.close()
```

### Strategy File Not Found

**Symptom**:
```
FileNotFoundError: strategies/demo_spot.py
```

**Solution**:
```bash
# Use absolute path
python3 dev_run.py -l info strategy \
  -n demo_spot \
  -p /absolute/path/to/strategies/demo_spot.py

# Or relative to current directory
cd /path/to/project
python3 core/python/dev_run.py -l info strategy \
  -n demo_spot \
  -p strategies/demo_spot.py
```

---

## Related Documentation

### Configuration
- [Config Usage Map](../40_config/CONFIG_REFERENCE.md) - Configuration file locations
- [Binance Config Contract](../30_contracts/binance_config_contract.md) - Exchange configuration

### Operations
- [PM2 Startup Guide](pm2_startup_guide.md) - Automated process management
- [Debugging Guide](DEBUGGING.md) - Troubleshooting issues
- [Log Locations](LOG_LOCATIONS.md) - Where to find logs

### Development
- [Strategy Framework](../10_modules/strategy_framework.md) - Writing strategies
- [Strategy Lifecycle Flow](../20_interactions/strategy_lifecycle_flow.md) - Execution model

---

## Quick Reference

### Start Services
```bash
python3 dev_run.py -l info master
python3 dev_run.py -l info ledger
python3 dev_run.py -l trace md -s binance
python3 dev_run.py -l info td -s binance -a my_account
python3 dev_run.py -l info strategy -n demo -p strategies/demo.py
```

### Stop Services
```bash
# Graceful (SIGINT)
kill -2 <master_pid>  # Stops all registered services

# Force (SIGKILL)
pkill -9 -f "dev_run.py"
```

### View Logs
```bash
# Real-time
tail -f ~/.pm2/logs/master-out.log

# Last 100 lines
tail -n 100 ~/.pm2/logs/md_binance-error.log

# Search for errors
grep -i error ~/.pm2/logs/*.log
```

### Check Status
```bash
# PM2 status
pm2 list

# Process status
ps aux | grep dev_run.py

# Master connectivity
curl http://localhost:9000/health  # (if health endpoint exists)
```

---

## Version History

- **2025-11-17**: Initial CLI operations guide
