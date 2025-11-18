# Log Locations and Monitoring

Quick reference for finding and analyzing system logs.

## Log Locations

### PM2 Managed Services (Recommended)

When using official scripts (`run.sh`), services are managed by PM2:

```bash
# Inside container
docker-compose exec app bash

# View real-time logs
pm2 logs master              # Master service
pm2 logs md_binance          # Market Data gateway
pm2 logs td_binance:gz_user1 # Trading gateway
pm2 logs strategy:hello      # Strategy
pm2 logs --lines 100         # Last 100 lines from all services

# Log files location (PM2 managed)
~/.pm2/logs/
```

### Manual Startup Logs (Legacy)

If starting services manually with `nohup` (NOT recommended):

```bash
tail -f /tmp/master.log
tail -f /tmp/md.log
tail -f /tmp/td.log
```

**Note**: PM2 is the official method. Manual startup is for debugging only.

### Persistent Logs (Runtime Directory)

Permanent logs are stored in `/app/runtime/` directory structure:

```
/app/runtime/
├── system/
│   └── <mode>/              # Usually "default"
│       ├── master/
│       │   └── log/live/*.log
│       └── ledger/
│           └── log/live/*.log
├── md/
│   └── <source>/            # e.g., "binance"
│       └── <source>/
│           └── log/live/*.log
└── td/
    └── <source>/            # e.g., "binance"
        └── <account>/       # e.g., "your_email@example.com"
            └── log/live/*.log
```

### Accessing Persistent Logs

```bash
# Master logs
tail -f /app/runtime/system/*/master/log/live/*.log

# MD Gateway logs (Binance)
tail -f /app/runtime/md/binance/binance/log/live/*.log

# TD Gateway logs (Binance)
tail -f /app/runtime/td/binance/*/log/live/*.log

# Example (Binance TD for account gz_user1):
# /app/runtime/td/binance/gz_user1/log/live/gz_user1.log

# Ledger logs
tail -f /app/runtime/system/*/ledger/log/live/*.log

# All logs at once
tail -f /app/runtime/*/log/live/*.log
```

## What to Look For

### Successful Startup

**Master**:
```
[info] Master started
[info] Ready to accept connections
```

**MD Gateway**:
```
[info] WebSocket connected
[info] Receiving depth data
[info] NOTIONAL  # Exchange info updates
```

**TD Gateway (Futures Testnet)**:
```
[info] future login successful
# OR no "-2015 Invalid API-key" errors
```

**Ledger**:
```
[info] Ledger service started
[info] Registered broker: binance
```

### Common Errors

**Error -2015: Invalid API Key**:
```
[error] spot login failed, error_id: -2015
```
- **Action**: Check API key in database matches testnet/production
- **Action**: Verify URL configuration in `common.h`

**Stream Truncated**:
```
[error] stream truncated
```
- **Action**: Check WebSocket port (should be 443, not 9443)
- **Action**: Verify WebSocket host in `common.h`

**Address Already in Use**:
```
RuntimeError: Address already in use
```
- **Action**: Kill existing processes: `pkill -f kfc`
- **Action**: Delete socket files: `find /app/runtime -name '*.nn' -type s -delete`

**Bus Error / Segmentation Fault**:
```
Bus error (core dumped)
```
- **Action**: Delete corrupted journals: `rm -rf /app/runtime`
- **Action**: Ensure MD gateway is running before strategy

## Log Analysis Tips

### Filter for Errors Only

```bash
# TD Gateway errors only
grep -i error /app/runtime/td/binance/*/log/live/*.log

# Recent errors (last 30 lines)
tail -30 /app/runtime/td/binance/*/log/live/*.log | grep -i error

# Specific error codes
grep "\-2015" /app/runtime/td/binance/*/log/live/*.log
```

### Monitor Multiple Services

```bash
# Watch all services simultaneously (requires multitail or tmux)
# Option 1: Simple approach
tail -f /tmp/master.log /tmp/md.log /tmp/td.log /tmp/ledger.log

# Option 2: With labels
tail -f /tmp/*.log | awk '{print FILENAME": "$0}'
```

### Search for Specific Events

```bash
# Find login attempts
grep -r "login" /app/runtime/td/binance/*/log/live/

# Find WebSocket connections
grep -r "WebSocket" /app/runtime/md/binance/binance/log/live/

# Find order activity
grep -r "order" /app/runtime/td/binance/*/log/live/
```

## Log Rotation

Logs are organized by mode (live/backtest/replay):

```
log/
├── live/           # Real-time trading logs
├── backtest/       # Backtesting logs
└── replay/         # Replay mode logs
```

## Quick Diagnostic Script

Create a diagnostic script to check all logs at once:

```bash
#!/bin/bash
# /app/scripts/check_logs.sh

echo "=== Master Log (Last 10 lines) ==="
tail -10 /tmp/master.log 2>/dev/null || echo "No master.log"

echo -e "\n=== MD Gateway Log (Last 10 lines) ==="
tail -10 /tmp/md.log 2>/dev/null || echo "No md.log"

echo -e "\n=== TD Gateway Log (Last 10 lines) ==="
tail -10 /tmp/td.log 2>/dev/null || echo "No td.log"

echo -e "\n=== Errors in Last 5 Minutes ==="
find /app/runtime -name "*.log" -type f -mmin -5 -exec grep -i "error" {} \; | tail -20
```

## Clearing Old Logs

```bash
# Clear temporary logs
rm -f /tmp/*.log

# Clear runtime logs (CAUTION: Deletes all historical data)
rm -rf /app/runtime/*/log/*
rm -rf /app/runtime/*/*/log/*
rm -rf /app/runtime/*/*/*/log/*

# Clear everything and start fresh
rm -rf /app/runtime
mkdir -p /app/runtime/journal
```

---

## Related

- [TESTNET.md](TESTNET.md) - Testnet configuration
- [HACKING.md](HACKING.md) - Development workflow
- Learning Plan Section 3.2 - Understanding log output
