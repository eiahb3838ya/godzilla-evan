# Binance Testnet Configuration Guide

Complete guide for configuring and troubleshooting Binance Testnet connections.

## Overview

This document covers the setup and troubleshooting of Binance Testnet connections for both **Spot** and **Futures** markets. The godzilla-evan system uses hardcoded URLs in the Binance extension C++ code, which requires special configuration for testnet environments.

## Quick Reference

### Testnet Endpoints

**Spot Testnet** (`testnet.binance.vision`):
- REST API: `https://testnet.binance.vision:443`
- WebSocket: `wss://stream.testnet.binance.vision:443`
- API Key Generation: https://testnet.binance.vision/key/generate

**Futures Testnet** (`testnet.binancefuture.com`):
- REST API: `https://testnet.binancefuture.com:443`
- WebSocket: `wss://stream.binancefuture.com:443` (USD-M) or `wss://dstream.binancefuture.com:443` (COIN-M)
- API Key Generation: Requires separate keys from Spot Testnet
- Documentation: https://developers.binance.com/docs/derivatives/

### Production Endpoints

**Production** (`api.binance.com`):
- REST API: `https://api.binance.com:443`
- WebSocket: `wss://stream.binance.com:443`
- Not compatible with testnet API keys

---

## Initial Setup

### 0. Install PM2 (Process Manager) ⚠️ **REQUIRED**

**Critical**: Official test scripts (`scripts/binance_test/run.sh`) require PM2.

```bash
# In Docker container
docker-compose exec app bash

# Install Node.js and npm
apt-get update
apt-get install -y nodejs npm

# Install PM2 globally
npm install -g pm2

# Verify installation
pm2 --version
# Should output: 6.0.13 or similar
```

**Why PM2?**
- Manages multiple services (Master, Ledger, MD, TD, Strategy)
- Auto-restart on failure
- Centralized logging (`pm2 logs`)
- Production-ready process management
- **Required by official scripts**

**Troubleshooting**:
- If `pm2: command not found` → Node.js/npm not installed
- If version warnings appear → Safe to ignore (PM2 works with older Node.js)

---

### 1. Obtain API Keys

#### Spot Testnet
```bash
# Visit in browser
https://testnet.binance.vision/key/generate

# Click "Generate HMAC_SHA256 Key"
# Save both API Key and Secret Key immediately (shown only once)
```

#### Futures Testnet
```bash
# Futures Testnet requires separate API keys
# Visit Binance official documentation for Futures Testnet access
# https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info
```

### 2. Configure Extension URLs

**Critical**: The Binance extension uses hardcoded URLs in C++ code, not from database configuration.

Edit `core/extensions/binance/include/common.h`:

```cpp
inline void from_json(const nlohmann::json &j, Configuration &c)
{
    j.at("user_id").get_to(c.user_id);
    j.at("access_key").get_to(c.access_key);
    j.at("secret_key").get_to(c.secret_key);
    
    // For Spot Testnet:
    c.spot_rest_host = "testnet.binance.vision";
    c.spot_rest_port = 443;
    c.spot_wss_host = "stream.testnet.binance.vision";
    c.spot_wss_port = 443;
    
    // For Futures Testnet:
    c.ubase_rest_host = "testnet.binancefuture.com";
    c.ubase_rest_port = 443;
    c.ubase_wss_host = "stream.binancefuture.com";
    c.ubase_wss_port = 443;
    c.cbase_rest_host = "testnet.binancefuture.com";
    c.cbase_rest_port = 443;
    c.cbase_wss_host = "dstream.binancefuture.com";
    c.cbase_wss_port = 443;
}
```

**For Production**:
```cpp
    c.spot_rest_host = "api.binance.com";
    c.spot_wss_host = "stream.binance.com";
    c.ubase_rest_host = "fapi.binance.com";
    c.ubase_wss_host = "fstream.binance.com";
    c.cbase_rest_host = "dapi.binance.com";
    c.cbase_wss_host = "dstream.binance.com";
```

### 3. Rebuild Extension

**Every time you modify `common.h`, you must rebuild**:

```bash
docker-compose exec app /bin/bash
cd /app/core/build
make kfext_binance -j$(nproc)

# Verify rebuild
ls -lah /app/core/build/build_extensions/binance/kfext_binance*.so
```

### 4. Add Account to Database

#### Method A: Interactive (Recommended for local terminal)

```bash
# Only works in interactive terminal (not in Docker background)
docker exec -it godzilla-dev bash
python core/python/dev_run.py account -s binance add

# Follow prompts:
# - User ID: gz_user1 (MUST use this for official scripts)
# - Access Key: <paste your API key>
# - Secret Key: <paste your secret key>
```

**⚠️ Important**: Use `gz_user1` as User ID to match official scripts.

---

#### Method B: Manual Database Creation (For Docker/non-interactive)

If interactive method fails with "AssertionError" or "no TTY":

```bash
docker-compose exec app bash

# Create database directory
mkdir -p /root/.config/kungfu/app

# Create database with Python script
python3 << 'EOF'
import sqlite3
import json
import os

EOF
```

### Configure Account

Use the official interactive command to add your account (refer to [official documentation](https://godzilla.dev/documentation/installation/#cloud-server-or-local-machine)):

```bash
cd ~/dev/godzilla-community
python core/python/dev_run.py account -s binance add
```

**Interactive input**:
- **user_id**: `gz_user1`
- **access_key**: Your Binance Testnet API Key
- **secret_key**: Your Binance Testnet Secret Key

This command automatically creates the correct database table structure

---

#### Verify Account

```bash
# Check account was added
python core/python/dev_run.py account -s binance show

# Expected output:
# receive_md    user_id    access_key                secret_key
# ------------  ---------  ------------------------  ------------------------
# True          gz_user1   MpFV92IITflE1iFCyzjq1n... UX9M52UeBxuQM91aJiOTi...
```

**Troubleshooting**:
- `python: command not found` → Run `ln -sf /usr/bin/python3 /usr/bin/python`
- `sqlite3.OperationalError` → Database path or table name incorrect
- Keys showing with `\n` → Strip newlines when pasting

---

## System Startup

### Method A: Official Scripts (Recommended) ⭐

**Location**: `scripts/binance_test/`

**Prerequisites**:
- PM2 installed (see step 0)
- Account added to database (see step 4)
- User ID must be `gz_user1`

**Start All Services**:
```bash
cd /app/scripts/binance_test

# Start Master, Ledger, MD, TD
bash run.sh start

# Wait ~30 seconds for all services to stabilize
sleep 30

# Check status
pm2 list
```

**Expected Output**:
```
┌────┬────────────────────────┬─────────┬─────────┬──────────┬────────┬──────┬───────────┐
│ id │ name                   │ mode    │ pid     │ uptime   │ ↺      │ status    │
├────┼────────────────────────┼─────────┼─────────┼──────────┼────────┼───────────┤
│ 0  │ master                 │ fork    │ 1234    │ 30s      │ 0      │ online    │
│ 1  │ ledger                 │ fork    │ 1235    │ 25s      │ 0      │ online    │
│ 2  │ md_binance             │ fork    │ 1236    │ 20s      │ 0      │ online    │
│ 3  │ td_binance:gz_user1    │ fork    │ 1237    │ 15s      │ 0      │ online    │
└────┴────────────────────────┴─────────┴─────────┴──────────┴────────┴───────────┘
```

**What `run.sh start` does**:
1. Clears journal files: `find ~/.config/kungfu/app/ -name "*.journal" | xargs rm -f`
2. Starts Master → waits 5s
3. Starts Ledger → waits 5s
4. Starts MD (Binance) → waits 5s
5. Starts TD (Binance) → waits 5s

**Start Strategy**:
```bash
# Create strategy PM2 config (if not exists)
cat > /app/scripts/binance_test/strategy_hello.json << 'EOF'
{
  "apps": [{
    "name": "strategy:hello",
    "cwd": "../../",
    "script": "core/python/dev_run.py",
    "exec_interpreter": "python3",
    "args": "-l info strategy -n hello -p strategies/helloworld/helloworld.py -c strategies/conf.json",
    "watch": false
  }]
}
EOF

# Start strategy
pm2 start strategy_hello.json

# View real-time market data
pm2 logs strategy:hello --lines 20
```

**PM2 Commands**:
```bash
pm2 list                          # List all processes
pm2 logs md_binance --lines 50    # View MD logs
pm2 logs td_binance:gz_user1      # View TD logs (live)
pm2 restart strategy:hello        # Restart strategy
pm2 stop all                      # Stop all services
pm2 delete all                    # Remove all services
```

---

### Method B: Manual Startup (For debugging)

**Critical**: Services must start in this order:

```bash
# 0. Clean old state
pkill -f 'python.*dev_run.py'
find /app/runtime -name '*.journal' -delete
find /app/runtime -name '*.nn' -type s -delete

# 1. Master (system controller)
cd /app
python core/python/dev_run.py -l info master > /tmp/master.log 2>&1 &
sleep 5

# 2. Ledger (order routing)
python core/python/dev_run.py -l info ledger > /tmp/ledger.log 2>&1 &
sleep 5

# 3. MD Gateway (market data)
python core/python/dev_run.py -l info md -s binance > /tmp/md.log 2>&1 &
sleep 5

# 4. TD Gateway (trading)
python core/python/dev_run.py -l info td -s binance -a gz_user1 > /tmp/td.log 2>&1 &
sleep 8

# 5. Strategy (your trading logic)
python core/python/dev_run.py -l info strategy -n hello \
  -p strategies/helloworld/helloworld.py \
  -c strategies/conf.json > /tmp/strategy.log 2>&1 &

# 6. Check logs
tail -f /tmp/md.log
tail -f /tmp/td.log
tail -f /tmp/strategy.log
```

---

### Complete Restart (Clean Slate)

#### Option A: Use Graceful Shutdown Script (Recommended)

```bash
cd /app/scripts/binance_test
bash graceful_shutdown.sh

# Wait a few seconds, then restart
bash run.sh start
```

**What it does**:
- Gracefully stops all PM2 processes
- Kills remaining Python processes
- Cleans journal files (`*.journal`)
- Cleans socket files (`*.nn`, `*.sock`)
- Removes old logs (>7 days)

#### Option B: Manual Cleanup

```bash
# Stop all services
pm2 delete all
pkill -9 python

# Clean all state
find /app/runtime -name '*.journal' -delete
find /app/runtime -name '*.nn' -type s -delete
rm -rf /app/runtime/md
rm -rf /app/runtime/td
rm -rf /app/runtime/system/master/*/journal

# Restart
cd /app/scripts/binance_test
bash run.sh start
```

---

## Troubleshooting

### Error: "bash: pm2: command not found"

**Symptom**:
```bash
$ bash run.sh start
run.sh: line 9: pm2: command not found
```

**Root Cause**: PM2 not installed

**Solution**:
```bash
# Install Node.js and PM2
apt-get update && apt-get install -y nodejs npm
npm install -g pm2

# Verify
pm2 --version
```

See [Step 0: Install PM2](#0-install-pm2-process-manager-️-required)

---

### Error: "bash: python: command not found"

**Symptom**:
```bash
$ python core/python/dev_run.py
bash: python: command not found
```

**Root Cause**: Container only has `python3`, some scripts expect `python`

**Solution**:
```bash
# Create symlink
ln -sf /usr/bin/python3 /usr/bin/python

# Verify
python --version
```

---

### Error: JSON Parse Error / Database Not Found

**Symptom**:
```
RuntimeError: [json.exception.parse_error.101] parse error at line 1, column 1: 
syntax error while parsing value - unexpected end of input; expected '[', '{', or a literal
```

**Root Causes**:
1. Database doesn't exist: `/root/.config/kungfu/app/kungfu.db`
2. Account config is empty or malformed
3. Interactive `account add` failed (no TTY)

**Solution**:
```bash
# Check if database exists
ls -lh /root/.config/kungfu/app/kungfu.db

# If not exists, create manually (see Method B in step 4)
# Add account using official command
cd ~/dev/godzilla-community
python core/python/dev_run.py account -s binance add

# Verify
python core/python/dev_run.py account -s binance show
```

---

### Error: "app register timeout" + "segmentation fault"

**Symptom**:
```
[error] app register timeout
[critical] segmentation violation
```

**Root Cause**: Old journal files causing Master to reject re-registration

**Solution (Complete Clean Restart)**:
```bash
# Stop all services
pm2 delete all
pkill -9 python
sleep 3

# Clean ALL runtime state
find /app/runtime -name '*.journal' -delete
find /app/runtime -name '*.nn' -type s -delete

# Restart services
cd /app/scripts/binance_test
bash run.sh start
```

**Why this happens**:
- Previous service crashed, leaving registration in Master's journal
- Master thinks service is already registered
- Refuses to send `RequestStart` message
- Service waits forever → timeout → crashes

**Prevention**: Always clean journals when restarting after crashes

---

### Error: Account Name Mismatch

**Symptom**:
```
# PM2 config has:
"args": "-l trace td -s binance -a eiahb3838ya@ntu.im"

# But database has:
user_id: gz_user1
```

**Root Cause**: PM2 config file not updated after database creation

**Solution**:
```bash
# Check database account
python core/python/dev_run.py account -s binance show

# Update PM2 config to match
nano scripts/binance_test/td_binance.json
# Change: "args": "-l trace td -s binance -a gz_user1"

# Restart TD
pm2 delete td_binance:gz_user1
pm2 start scripts/binance_test/td_binance.json
```

**Best Practice**: Always use `gz_user1` for consistency with official scripts

---

### Error: Strategy No Output

**Symptom**:
```bash
$ python strategies/helloworld/helloworld.py
# No output, nothing happens
```

**Root Cause**: Strategies MUST be run via `kfc strategy` command, not directly

**Solution**:
```bash
# Wrong:
python strategies/helloworld/helloworld.py

# Correct:
python core/python/dev_run.py strategy -n hello \
  -p strategies/helloworld/helloworld.py \
  -c strategies/conf.json

# Or use PM2:
pm2 start scripts/binance_test/strategy_hello.json
```

---

### Error: "Invalid API-key, IP, or permissions for action" (-2015)

**Symptom**:
```
[error] spot login failed, error_id: -2015
[error] future login failed, error_id: -2015
```

**Root Causes**:

1. **Wrong URL Configuration**
   - Problem: `common.h` still uses production URLs
   - Solution: Edit `common.h` to use testnet URLs and rebuild
   
2. **API Key Mismatch**
   - Spot Testnet keys ≠ Futures Testnet keys
   - Testnet keys ≠ Production keys
   - Solution: Use correct key for your environment
   
3. **Database vs Code Mismatch**
   - Problem: Database has testnet key, but code uses production URLs
   - Solution: URLs in `common.h` must match key type

**Diagnostic Steps**:

```bash
# Test Spot Testnet API key
API_KEY="your_key"
SECRET_KEY="your_secret"
TIMESTAMP=$(date +%s000)
SIGNATURE=$(echo -n "timestamp=$TIMESTAMP" | openssl dgst -sha256 -hmac "$SECRET_KEY" | awk '{print $2}')

curl -H "X-MBX-APIKEY: $API_KEY" \
  "https://testnet.binance.vision/api/v3/account?timestamp=$TIMESTAMP&signature=$SIGNATURE"

# Success: Returns account info
# Failure: {"code":-2015,"msg":"Invalid API-key..."}
```

```bash
# Test Futures Testnet API key
curl -H "X-MBX-APIKEY: $API_KEY" \
  "https://testnet.binancefuture.com/fapi/v2/account?timestamp=$TIMESTAMP&signature=$SIGNATURE"
```

### Error: "stream truncated" or WebSocket Connection Failed

**Symptom**:
```
[error] stream truncated
```

**Root Causes**:

1. **Wrong WebSocket Port**
   - Problem: Using port 9443 for Futures Testnet
   - Solution: Change to port 443 in `common.h`

2. **Wrong WebSocket Host**
   - Spot: `stream.testnet.binance.vision:443`
   - Futures USD-M: `stream.binancefuture.com:443`
   - Futures COIN-M: `dstream.binancefuture.com:443`

**Diagnostic Steps**:

```bash
# Test WebSocket port connectivity
curl -I https://stream.binancefuture.com:443
# Should return: HTTP/2 404 (port is open)

curl -I https://stream.binancefuture.com:9443
# Should fail: connection refused (port is closed)
```

### Error: "Address already in use"

**Symptom**:
```
RuntimeError: Address already in use
```

**Root Cause**: Previous service crashed, leaving socket files

**Solution**:
```bash
# Kill all kfc processes
pkill -f kfc
sleep 3

# Delete socket files
find /app/runtime -name '*.nn' -type s -delete

# Restart services
```

### Error: "app register timeout"

**Symptom**:
```
[error] app register timeout
```

**Root Cause**: Communication issue between gateways and Master

**Solution**:
```bash
# Full container restart
docker-compose restart app
sleep 10

# Re-enter container and start services
docker-compose exec app bash
# Then follow startup sequence
```

### Error: "bus error" or "segmentation fault"

**Symptom**:
```
Bus error (core dumped)
Segmentation fault
```

**Root Causes**:

1. **Corrupted Journal Files**
   - Problem: Previous crash left 0-byte journal files
   - Solution: Delete runtime directory
   
   ```bash
   rm -rf /app/runtime
   mkdir -p /app/runtime/journal
   ```

2. **Missing MD Gateway**
   - Problem: Strategy subscribes to data source that isn't running
   - Solution: Start MD gateway before strategy

3. **Memory Mapping Failure**
   - Problem: Invalid journal file structure
   - Solution: Clean restart with fresh journals

---

## Verification

### Check Running Services

```bash
ps aux | grep kfc | grep -v grep
# Should show: master, md, td, ledger
```

### View Logs

**Live Logs** (in container `/tmp` when using nohup):
```bash
tail -f /tmp/master.log
tail -f /tmp/md.log
tail -f /tmp/td.log
tail -f /tmp/ledger.log
```

**Runtime Logs**:
```bash
# Master
tail -f /app/runtime/system/*/master/log/live/*.log

# MD Gateway
tail -f /app/runtime/md/binance/binance/log/live/*.log

# TD Gateway
tail -f /app/runtime/td/binance/*/log/live/*.log

# Ledger
tail -f /app/runtime/system/*/ledger/log/live/*.log
```

### Success Indicators

**TD Gateway (Spot Testnet)**:
- No `-2015` errors
- Logs show "spot login successful" or similar

**TD Gateway (Futures Testnet)**:
- No `-2015` errors
- No "stream truncated" errors
- Logs show "future login successful" or similar

**MD Gateway**:
- WebSocket connected
- Receiving market data (depth/ticker)
- Log shows exchange info updates

---

## Important Notes

### API Key Compatibility Matrix

| Key Type | Spot Testnet | Futures Testnet | Production |
|----------|--------------|-----------------|------------|
| Spot Testnet | ✅ | ❌ | ❌ |
| Futures Testnet | ❌ | ✅ | ❌ |
| Production | ❌ | ❌ | ✅ |
| Demo Account | ❌ | ❌ | ❌ (web UI only) |

### Code Behavior

**Default Initialization** (`trader_binance.cpp` line 97-107):
```cpp
// Both markets enabled by default
if (config_.enable_futures && frest_ptr_) {
    _start_userdata(InstrumentType::FFuture);
}
if (config_.enable_spot && rest_ptr_) {
    _start_userdata(InstrumentType::Spot);
}
```

- **NEW (ADR-004)**: You can now disable Spot or Futures markets via configuration
- Spot connection only attempted during reconnection logic if enabled
- Expected: "spot login failed" if using Futures-only key and Spot is enabled

---

### Disabling Spot Login (Reduce Log Noise)

**Problem**: When using Futures-only API keys, the system attempts Spot login every 5 seconds, generating `-2015` error logs.

**Solution**: Disable Spot market in your database configuration.

#### Update Existing Account

```bash
# Method 1: Using SQLite command line
sqlite3 /root/.config/kungfu/app/kungfu.db
UPDATE account_config 
SET config = json_set(config, '$.enable_spot', json('false'))
WHERE user_id = 'gz_user1';
.quit

# Method 2: Using Python script
python3 << 'EOF'
import sqlite3, json
conn = sqlite3.connect('/root/.config/kungfu/app/kungfu.db')
cursor = conn.cursor()

# Get current config
cursor.execute("SELECT config FROM account_config WHERE user_id = ?", ('gz_user1',))
row = cursor.fetchone()
if row:
    config = json.loads(row[0])
    config['enable_spot'] = False  # Disable Spot
    cursor.execute("UPDATE account_config SET config = ? WHERE user_id = ?",
                   (json.dumps(config), 'gz_user1'))
    conn.commit()
    print("✅ Spot market disabled")
else:
    print("❌ Account not found")
conn.close()
EOF
```

#### Create New Account with Spot Disabled

```python
import sqlite3, json

conn = sqlite3.connect('/root/.config/kungfu/app/kungfu.db')
cursor = conn.cursor()

config = {
    'access_key': 'YOUR_FUTURES_KEY',
    'secret_key': 'YOUR_FUTURES_SECRET',
    'enable_spot': False,      # Disable Spot
    'enable_futures': True     # Enable Futures (optional, default is true)
}

cursor.execute('INSERT OR REPLACE INTO account_config VALUES (?, ?, ?, ?)',
               ('gz_user1', 'binance', 1, json.dumps(config)))
conn.commit()
conn.close()
print("✅ Account created with Spot disabled")
```

#### Verify Configuration

```bash
# Restart TD Gateway
pm2 restart td_binance:gz_user1

# Check logs
pm2 logs td_binance:gz_user1 --lines 20

# Expected output:
# [info] Connecting BINANCE TD for gz_user1 (Spot: disabled, Futures: enabled)
# [info] Skipping Spot initialization (disabled or client unavailable)
# [info] login success
# 
# You should NOT see:
# [error] spot login failed, error_id: -2015
```

#### Configuration Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_spot` | boolean | `true` | Enable Spot market initialization |
| `enable_futures` | boolean | `true` | Enable Futures market initialization |

**Examples**:

```json
// Futures only (recommended for Futures Testnet)
{
    "access_key": "FUTURES_KEY",
    "secret_key": "FUTURES_SECRET",
    "enable_spot": false
}

// Spot only
{
    "access_key": "SPOT_KEY",
    "secret_key": "SPOT_SECRET",
    "enable_futures": false
}

// Both enabled (default, backward compatible)
{
    "access_key": "KEY",
    "secret_key": "SECRET"
}
```

**Note**: At least one market should be enabled for the TD Gateway to function properly.

---

### Port Reference

| Endpoint | Production Port | Testnet Port |
|----------|----------------|--------------|
| Spot REST | 443 | 443 |
| Spot WSS | 443 | 443 |
| Futures REST | 443 | 443 |
| Futures WSS (USD-M) | 443 | 443 ✅ (NOT 9443) |
| Futures WSS (COIN-M) | 443 | 443 ✅ (NOT 9443) |

---

## Lessons Learned

### 1. Hardcoded URLs Are Intentional

The extension uses hardcoded URLs in `common.h`, not database configuration. This requires:
- Manual code editing for environment switching
- Rebuild after every URL change
- Careful tracking of which environment is compiled

### 2. Testnet Keys Are Environment-Specific

- Spot and Futures testnets require separate API keys
- Demo account keys (`demo.binance.com`) are web UI only, not for API access
- Always verify key with direct `curl` test before debugging system

### 3. Port 443 Is Standard for All Testnet WebSockets

Binance documentation may be ambiguous, but testing confirms:
- Production Futures: `stream.binancefuture.com:443`
- Testnet Futures: `stream.binancefuture.com:443` (same port)
- Port 9443 is not used for Futures Testnet

### 4. Clean Environment Is Critical

When troubleshooting:
1. Stop all services (`pkill -f kfc`)
2. Delete socket files (`find /app/runtime -name '*.nn' -type s -delete`)
3. Consider deleting entire runtime (`rm -rf /app/runtime`)
4. Restart in proper sequence

---

## Related Documentation

- [INSTALL.md](INSTALL.md) - Environment setup
- [HACKING.md](HACKING.md) - Development workflow
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- Binance Spot Testnet: https://testnet.binance.vision/
- Binance Derivatives Docs: https://developers.binance.com/docs/derivatives/

---

Last Updated: 2025-10-28  
Based on: PM2 + Database configuration complete system startup debugging session
