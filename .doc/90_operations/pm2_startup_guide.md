---
title: PM2 Startup Guide
updated_at: 2025-11-21
owner: operations
lang: en
tags: [operations, pm2, startup, process-management, deployment]
code_refs:
  - scripts/binance_test/run.sh:1-45
  - scripts/binance_test/master.json:1-16
  - scripts/binance_test/ledger.json:1-16
  - scripts/binance_test/md_binance.json:1-16
  - scripts/binance_test/td_binance.json:1-16
  - core/python/kungfu/data/sqlite/models.py:23-28
  - core/python/kungfu/command/account/add.py:15-25
purpose: "Guide for starting and managing the trading system using PM2 process manager"
tokens_estimate: 4600
---

# PM2 Startup Guide

## Overview

The trading system uses **PM2** (Process Manager 2) to manage multiple Python processes that form the complete trading infrastructure. PM2 provides process monitoring, automatic restarts, log management, and graceful shutdowns.

**Key Components**:
1. **Master** - Central coordinator and registry
2. **Ledger** - Account state and position tracking
3. **Market Data (MD)** - Market data gateways per exchange
4. **Trader (TD)** - Trading gateways per exchange/account
5. **Strategy** - User-defined trading strategies

## Prerequisites

### Install PM2

```bash
# Install PM2 globally via npm
npm install -g pm2

# Verify installation
pm2 --version
```

### Python Environment

```bash
# Ensure Python 3.7+ is available
python3 --version

# Verify kungfu module is installed
python3 -c "import kungfu; print('OK')"
```

## Service Architecture

### Component Dependencies

```
Master (port 9000)
  ↓
  └─ Ledger (reads master registry)
      ↓
      ├─ Market Data Gateways (MD)
      │   └─ md_binance (WebSocket → Journal)
      │
      ├─ Trading Gateways (TD)
      │   └─ td_binance (REST API ← Journal)
      │
      └─ Strategies
          └─ your_strategy (reads MD, writes orders)
```

**Critical**: Components must start in this order:
1. Master first (others register with it)
2. Ledger second (tracks all accounts)
3. MD/TD gateways (connect to exchanges)
4. Strategies last (consume MD, send orders)

## Startup Scripts

### Quick Start

**Location**: [`scripts/binance_test/run.sh`](../../scripts/binance_test/run.sh)

```bash
cd scripts/binance_test
./run.sh start
```

This script:
1. Clears old journal files (development only)
2. Starts services in correct order with delays
3. Uses PM2 to manage processes

### Manual Startup (Step-by-Step)

#### Step 1: Start Master

**File**: [`scripts/binance_test/master.json`](../../scripts/binance_test/master.json)

```bash
pm2 start master.json
```

**Configuration**:
```json
{
  "apps": [{
    "name": "master",
    "cwd": "./",
    "script": "../../core/python/dev_run.py",
    "exec_interpreter": "python3",
    "args": "-l info master",
    "watch": "true",
    "env": {
      "CLEAR_JOURNAL": "1"
    }
  }]
}
```

**What it does**:
- Launches master service on port 9000
- Provides service registry (other components register here)
- Log level: `info` (can be `trace`, `debug`, `info`, `warning`, `error`)

**Wait**: 5 seconds for master to initialize

#### Step 2: Start Ledger

**File**: [`scripts/binance_test/ledger.json`](../../scripts/binance_test/ledger.json)

```bash
sleep 5  # Wait for master
pm2 start ledger.json
```

**Configuration**:
```json
{
  "apps": [{
    "name": "ledger",
    "cwd": "./",
    "script": "../../core/python/dev_run.py",
    "exec_interpreter": "python3",
    "args": "-l info ledger",
    "watch": "true",
    "env": {
      "CLEAR_JOURNAL": "1"
    }
  }]
}
```

**What it does**:
- Tracks all account positions and assets
- Coordinates order routing to trading gateways
- Publishes broker states

**Wait**: 5 seconds for ledger to register and initialize

#### Step 3: Start Market Data Gateway

**File**: [`scripts/binance_test/md_binance.json`](../../scripts/binance_test/md_binance.json)

```bash
sleep 5  # Wait for ledger
pm2 start md_binance.json
```

**Configuration**:
```json
{
  "apps": [{
    "name": "md_binance",
    "cwd": "./",
    "script": "../../core/python/dev_run.py",
    "exec_interpreter": "python3",
    "args": "-l trace md -s binance",
    "watch": "true",
    "env": {
      "CLEAR_JOURNAL": "1"
    }
  }]
}
```

**Arguments**:
- `-l trace`: Log level (trace shows WebSocket messages)
- `md`: Market data service
- `-s binance`: Source identifier (binance gateway)

**What it does**:
- Connects to Binance WebSocket API
- Subscribes to market data (depth, trades, tickers)
- Writes market data to journal

**Wait**: 5 seconds for WebSocket connection to establish

#### Step 4: Start Trading Gateway

**File**: [`scripts/binance_test/td_binance.json`](../../scripts/binance_test/td_binance.json)

```bash
sleep 5  # Wait for MD
pm2 start td_binance.json
```

**Configuration**:
```json
{
  "apps": [{
    "name": "td_binance",
    "cwd": "./",
    "script": "../../core/python/dev_run.py",
    "exec_interpreter": "python3",
    "args": "-l info td -s binance -a my_account",
    "watch": "true",
    "env": {
      "CLEAR_JOURNAL": "1"
    }
  }]
}
```

**Arguments**:
- `td`: Trading service
- `-s binance`: Source identifier
- `-a my_account`: Account identifier (must match config file)

**What it does**:
- Connects to Binance REST API
- Reads order requests from journal
- Executes orders via exchange API
- Writes order confirmations back to journal

**Wait**: 5 seconds for API authentication

#### Step 5: Start Strategy (Optional)

**Example**: `strategy_demo.json`

```bash
pm2 start strategy_demo.json
```

**Configuration**:
```json
{
  "apps": [{
    "name": "strategy_demo",
    "cwd": "./",
    "script": "../../core/python/dev_run.py",
    "exec_interpreter": "python3",
    "args": "-l info strategy -s demo_spot -p ../../strategies/demo_spot.py",
    "watch": "true"
  }]
}
```

**Arguments**:
- `strategy`: Strategy runner
- `-s demo_spot`: Strategy name
- `-p ../../strategies/demo_spot.py`: Path to strategy file

## PM2 Configuration Options

### Common Options

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Process name (shown in `pm2 list`) | `"master"` |
| `cwd` | Working directory | `"./"` |
| `script` | Python script to run | `"dev_run.py"` |
| `exec_interpreter` | Interpreter | `"python3"` |
| `args` | Command-line arguments | `"-l info master"` |
| `watch` | Restart on file changes | `"true"` (dev only) |
| `env` | Environment variables | `{"CLEAR_JOURNAL": "1"}` |

### Log Levels

Set via `-l <level>` argument:

| Level | Usage | Output |
|-------|-------|--------|
| `trace` | WebSocket debugging | All messages, very verbose |
| `debug` | Development | Function calls, internal state |
| `info` | Production (default) | Important events, startups |
| `warning` | Production errors | Warnings and errors only |
| `error` | Critical only | Errors only |

**Example**:
```bash
# Verbose MD gateway (see all WebSocket messages)
pm2 start md_binance.json --update-env -l trace

# Quiet strategy (errors only)
pm2 start strategy_demo.json --update-env -l error
```

### Environment Variables

#### CLEAR_JOURNAL

**Purpose**: Delete old journal files on startup (development only)

```json
"env": {
  "CLEAR_JOURNAL": "1"
}
```

**Warning**: NEVER use in production (loses all historical data)

**Journal Location**: `~/.config/kungfu/app/runtime/journal/`

**Manual Clear**:
```bash
find ~/.config/kungfu/app/ -name "*.journal" | xargs rm -f
```

#### KF_HOME (Advanced)

**Purpose**: Override default kungfu home directory

```json
"env": {
  "KF_HOME": "/custom/path/to/kungfu"
}
```

**Default**: `~/.config/kungfu/app/runtime`

## Process Management

### Check Status

```bash
# List all processes
pm2 list

# Output:
# ┌─────┬────────────┬─────────┬──────┬───────────┬──────────┬──────────┐
# │ id  │ name       │ status  │ cpu  │ memory    │ watching │ restarts │
# ├─────┼────────────┼─────────┼──────┼───────────┼──────────┼──────────┤
# │ 0   │ master     │ online  │ 0%   │ 45.2 MB   │ enabled  │ 0        │
# │ 1   │ ledger     │ online  │ 0%   │ 52.8 MB   │ enabled  │ 0        │
# │ 2   │ md_binance │ online  │ 1%   │ 38.5 MB   │ enabled  │ 0        │
# │ 3   │ td_binance │ online  │ 0%   │ 35.1 MB   │ enabled  │ 0        │
# └─────┴────────────┴─────────┴──────┴───────────┴──────────┴──────────┘

# Show detailed info
pm2 show master

# Monitor in real-time
pm2 monit
```

### View Logs

```bash
# Tail all logs
pm2 logs

# Tail specific process
pm2 logs master

# Last 100 lines
pm2 logs master --lines 100

# Clear logs
pm2 flush
```

**Log Files**:
- Location: `~/.pm2/logs/`
- Format: `<name>-out.log` (stdout), `<name>-error.log` (stderr)

### Restart Processes

```bash
# Restart single process
pm2 restart master

# Restart all
pm2 restart all

# Reload (zero-downtime, if supported)
pm2 reload master
```

### Stop Processes

```bash
# Stop single process
pm2 stop master

# Stop all
pm2 stop all

# Delete from PM2 registry
pm2 delete master
```

## Shutdown Sequence

### Graceful Shutdown

**Script**: [`scripts/binance_test/run.sh stop`](../../scripts/binance_test/run.sh)

```bash
./run.sh stop
```

**What it does**:
```bash
# Find master process PID
master_pid=$(ps -ef | grep python | grep master | awk '{ print $2 }')

# Send SIGINT (Ctrl+C equivalent)
kill -2 $master_pid
```

**Graceful Shutdown Order**:
1. Master receives SIGINT
2. Master broadcasts shutdown to all registered services
3. Strategies stop first (finish in-flight orders)
4. Gateways stop next (close WebSocket/API connections)
5. Ledger stops (write final positions)
6. Master exits last

**Wait Time**: 5-10 seconds for complete shutdown

### Force Shutdown

```bash
# Stop all PM2 processes immediately
pm2 stop all

# Kill all PM2 processes (last resort)
pm2 kill
```

**Warning**: Force shutdown may leave orders in inconsistent state. Always prefer graceful shutdown.

## Troubleshooting

### Process Won't Start

**Symptom**: Process shows "errored" or "stopped" in `pm2 list`

**Debug**:
```bash
# Check error log
pm2 logs <process-name> --err --lines 50

# Common issues:
# 1. Master not running (start master first)
# 2. Port already in use (kill existing process)
# 3. Python import error (check PYTHONPATH)
```

### Process Keeps Restarting

**Symptom**: `restarts` count keeps increasing in `pm2 list`

**Debug**:
```bash
# Watch logs in real-time
pm2 logs <process-name> --lines 0

# Common issues:
# 1. Configuration error (check database integrity)
# 2. Missing account config (use: kfc account -s binance add)
# 3. Exchange API key invalid (check account_config table)
```

### WebSocket Disconnects

**Symptom**: MD gateway logs "Connection closed" repeatedly

**Debug**:
```bash
pm2 logs md_binance | grep -i "disconnect\|error"

# Common issues:
# 1. Network instability (check internet connection)
# 2. Rate limiting (too many subscriptions)
# 3. Exchange maintenance (check exchange status page)
```

**Fix**: PM2 auto-restarts the process, WebSocket reconnects automatically

### Orders Not Executing

**Symptom**: Strategy sends orders but nothing happens

**Debug**:
1. Check ledger logs:
   ```bash
   pm2 logs ledger | grep -i "order\|error"
   ```

2. Check TD gateway logs:
   ```bash
   pm2 logs td_binance | grep -i "order\|error"
   ```

3. Verify account configuration (stored in SQLite database):
   ```python
   import sqlite3, json
   conn = sqlite3.connect('/home/huyifan/projects/godzilla-evan/runtime/system/etc/kungfu/db/live/accounts.db')
   cursor = conn.cursor()
   cursor.execute("SELECT config FROM account_config WHERE account_id = 'binance_my_account'")
   result = cursor.fetchone()
   if result:
       print(json.dumps(json.loads(result[0]), indent=2))
   conn.close()
   ```

**Common issues**:
- Account ID mismatch (strategy uses different account than TD gateway)
- Insufficient balance (ledger rejects order)
- Invalid API keys (TD gateway can't authenticate)

### High Memory Usage

**Symptom**: Process memory >500 MB in `pm2 list`

**Debug**:
```bash
# Check process details
pm2 show <process-name>

# Restart to clear memory
pm2 restart <process-name>
```

**Common causes**:
- Journal file growth (old events accumulate)
- Memory leak (report issue if persists)
- Too many subscriptions (reduce symbols)

## Production Deployment

### Disable Watch Mode

**Development**:
```json
"watch": "true"  // Auto-restart on file changes
```

**Production**:
```json
"watch": "false"  // Explicit restarts only
```

### Remove CLEAR_JOURNAL

**Development**:
```json
"env": {
  "CLEAR_JOURNAL": "1"  // Wipe journals on start
}
```

**Production**:
```json
"env": {}  // Preserve journal history
```

### Set Log Level to Info

**Development**: `-l trace` (verbose)
**Production**: `-l info` (balanced) or `-l warning` (quiet)

### Enable PM2 Startup

**Auto-start on server reboot**:
```bash
# Generate startup script
pm2 startup

# Save current process list
pm2 save

# Test by rebooting
sudo reboot
```

After reboot, verify:
```bash
pm2 list  # Should show all processes running
```

### Log Rotation

**Configure log rotation** to prevent disk space issues:

```bash
pm2 install pm2-logrotate

# Configure rotation
pm2 set pm2-logrotate:max_size 10M     # Rotate at 10 MB
pm2 set pm2-logrotate:retain 7         # Keep 7 days
pm2 set pm2-logrotate:compress true    # Gzip old logs
```

## Docker Deployment

### PM2 in Docker

**Dockerfile**:
```dockerfile
FROM python:3.9

# Install PM2
RUN npm install -g pm2

# Copy application
COPY . /app
WORKDIR /app

# Start services
CMD ["pm2-runtime", "start", "master.json", "ledger.json", "md_binance.json", "td_binance.json"]
```

**Notes**:
- Use `pm2-runtime` instead of `pm2` (blocks and logs to stdout)
- Combine multiple JSON files in single command
- Set `KF_HOME=/app/runtime` for persistent journals

## Related Documentation

### Configuration
- [Config Usage Map](../40_config/config_usage_map.md) - Configuration file locations
- [Binance Config Contract](../30_contracts/binance_config_contract.md) - Exchange configuration

### Debugging
- [Debugging Guide](DEBUGGING.md) - Common issues and solutions
- [Log Locations](LOG_LOCATIONS.md) - Where to find logs

### Operations
- [CLI Operations Guide](cli_operations_guide.md) - Command-line tools
- [Docker Deployment](../95_adr/001-docker.md) - Container deployment

## Appendix: Service Arguments Reference

### Master

```bash
python3 dev_run.py -l info master
```

**Arguments**:
- `-l <level>`: Log level
- `master`: Service type

**No additional arguments**

### Ledger

```bash
python3 dev_run.py -l info ledger
```

**Arguments**:
- `-l <level>`: Log level
- `ledger`: Service type

**No additional arguments**

### Market Data

```bash
python3 dev_run.py -l info md -s <source>
```

**Arguments**:
- `-l <level>`: Log level
- `md`: Service type
- `-s <source>`: Exchange identifier (e.g., `binance`, `okex`)

### Trader

```bash
python3 dev_run.py -l info td -s <source> -a <account>
```

**Arguments**:
- `-l <level>`: Log level
- `td`: Service type
- `-s <source>`: Exchange identifier
- `-a <account>`: Account identifier (must exist in config)

### Strategy

```bash
python3 dev_run.py -l info strategy -s <name> -p <path>
```

**Arguments**:
- `-l <level>`: Log level
- `strategy`: Service type
- `-s <name>`: Strategy name (unique identifier)
- `-p <path>`: Path to strategy Python file

## Version History

- **2025-11-17**: Initial PM2 startup guide
