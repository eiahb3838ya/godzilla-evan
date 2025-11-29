# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context Loading

**This project has a comprehensive documentation system in `.doc/`**. Before starting any work:

```
follow .doc/START.md
```

This loads ~113k tokens of structured documentation covering:
- System architecture (Yijinjing event sourcing + Wingchun trading framework)
- API contracts and data structures
- Module interactions and lifecycle flows
- Configuration management
- Operational procedures

**Language**: User prefers 繁體中文 (zh-TW) unless otherwise specified. The `.doc/START.md` enforces this.

**Documentation Priority**: `.doc/` > `core/` source code > `strategies/` examples

## Architecture Overview

This is a **low-latency cryptocurrency trading system** with three layers:

```
Python Strategy Layer (User Logic)
         ↓ pybind11
Wingchun (C++) - Strategy runtime, order management, position tracking
         ↓
Yijinjing (C++) - Event sourcing journal (~50μs latency)
         ↓
Exchange Gateways - Binance REST/WebSocket
```

**Key Concepts**:
- **Journal**: Append-only event log for all trading events (orders, fills, market data)
- **Event Sourcing**: Complete audit trail with time-travel debugging capability
- **Strategy Callbacks**: `pre_start()` → `on_depth()` / `on_order()` / `on_transaction()` → `pre_stop()`
- **Single-threaded**: All callbacks execute sequentially (<1ms each)

## Build Commands

**In Docker Container**:
```bash
# Enter container
docker-compose exec app /bin/bash

# Initial build
cd /app/core/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Rebuild after changes
make -j$(nproc)

# Clean build
rm -rf /app/core/build/*
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

**Build Types**:
- `Release`: -O3 optimization (production)
- `Debug`: -O0 -g (debugging)
- `RelWithDebInfo`: -O3 -g (profiling)

**Python bindings**: Automatically built with C++ targets, output to `build/kfc/python/`

## Running Strategies

### ⚠️ CRITICAL: Always Use Docker + PM2

**This project MUST run inside Docker container with PM2 process manager.**

DO NOT run services directly on host machine. DO NOT use manual terminal commands for production/testing.

### Standard Startup Procedure

#### Step 1: Verify Docker Container is Running

```bash
# Check container status
docker ps | grep godzilla-dev

# If not running, start it
docker-compose up -d
```

#### Step 2: Use PM2 Inside Docker Container

**Method 1: Quick Start (Recommended)**

From **host machine**:
```bash
# Start all services (master, ledger, md, td)
docker exec -it godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"

# Check status
docker exec godzilla-dev pm2 list

# View logs
docker exec -it godzilla-dev pm2 logs

# Stop all services
docker exec -it godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh stop"
```

**Method 2: Step-by-Step (For Learning)**

Enter container first:
```bash
docker exec -it godzilla-dev bash
```

Then inside container:
```bash
cd /app/scripts/binance_test

# Start services in order
pm2 start master.json    # Wait 5 seconds
pm2 start ledger.json    # Wait 5 seconds
pm2 start md_binance.json    # Wait 5 seconds
pm2 start td_binance.json    # Wait 5 seconds

# Start your strategy (example)
cd /app/scripts/demo_future
pm2 start strategy_demo_future.json
```

#### Step 3: Monitor Services

```bash
# List all processes
docker exec godzilla-dev pm2 list

# View specific service logs
docker exec -it godzilla-dev pm2 logs master
docker exec -it godzilla-dev pm2 logs td_binance
docker exec -it godzilla-dev pm2 logs strategy_demo_future

# Monitor real-time
docker exec -it godzilla-dev pm2 monit
```

### Service Startup Order (Critical)

**Must follow this order with 5-second delays**:

```
1. Master      (service registry, must start first)
   ↓ wait 5s
2. Ledger      (account/position tracking)
   ↓ wait 5s
3. MD Gateway  (market data from exchanges)
   ↓ wait 5s
4. TD Gateway  (trading execution)
   ↓ wait 5s
5. Strategy    (your trading logic)
```

### Running Custom Strategies

To run your own strategy, create PM2 config in `scripts/<your_strategy>/`:

```json
{
  "apps": [{
    "name": "strategy_<name>",
    "cwd": "/app",
    "script": "/app/core/python/dev_run.py",
    "exec_interpreter": "python3",
    "args": "-l info strategy -n <name> -p /app/strategies/<name>/<name>.py -c /app/strategies/<name>/config.json",
    "watch": false,
    "env": {
      "KF_HOME": "/app/runtime"
    }
  }]
}
```

Then start it:
```bash
docker exec godzilla-dev pm2 start /app/scripts/<your_strategy>/strategy_<name>.json
```

### Cleanup Before Fresh Start

```bash
# Stop all PM2 processes
docker exec godzilla-dev pm2 stop all
docker exec godzilla-dev pm2 delete all

# Clear journal history (development only!)
docker exec godzilla-dev bash -c "find ~/.config/kungfu/app/ -name '*.journal' | xargs rm -f"
```

### Why Docker + PM2?

1. **Isolation**: Consistent environment across development/production
2. **Process Management**: PM2 auto-restarts on crashes, manages logs
3. **Correct Paths**: All paths use `/app` prefix inside container
4. **Dependencies**: All C++ libraries and Python packages pre-installed
5. **Monitoring**: PM2 provides CPU/memory metrics and log aggregation

### Reference

- PM2 Guide: `.doc/90_operations/pm2_startup_guide.md`
- Docker Setup: `.doc/00_index/INSTALL.md`
- Service Architecture: `.doc/00_index/ARCHITECTURE.md`

## Code Structure Notes

**C++ Core** (`core/cpp/`):
- `yijinjing/`: Journal system (no strategy logic)
- `wingchun/strategy/`: Strategy execution engine (`runner.cpp` routes events to Python callbacks)
- `wingchun/broker/`: Order management and routing
- `wingchun/book/`: Position and PnL tracking
- `wingchun/pybind/`: pybind11 bindings (maps C++ structs/enums to Python)

**Extensions** (`core/extensions/`):
- Each exchange has its own directory (e.g., `binance/`)
- Must implement `MarketData` and `Trader` abstract classes
- Registered in `EXTENSION_REGISTRY_MD` / `EXTENSION_REGISTRY_TD`

**Python Layer** (`core/python/kungfu/`):
- `command/`: CLI entry points (master, ledger, md, td, strategy)
- User strategies import from `kungfu.wingchun` (bound C++ classes)

## Critical Files

When modifying these, update corresponding `.doc/` files (see `.doc/UPDATE.md`):

**Data structures** (`core/cpp/wingchun/include/kungfu/wingchun/msg.h`):
- `Order` (line 666-730): Order state machine
- `Depth` (line 242-302): Market depth (10 levels)
- `Position` (line 1000-1071): Holdings and PnL
- `Asset` (line 947-998): Cash balances

**Strategy lifecycle** (`core/cpp/wingchun/src/strategy/runner.cpp`):
- Lines 55-194: Event subscriptions and callback routing
- Lines 66-76: Depth events
- Lines 124-141: Order events (routed by `strategy_id`)

**Python bindings** (`core/cpp/wingchun/pybind/pybind_wingchun.cpp`):
- Lines 264-319: Enum bindings (Side, OrderStatus, etc.)
- Lines 516-547: Order struct binding
- Lines 719-743: Context API binding

## Configuration

**Location**: `~/.config/kungfu/app/runtime/config/`

**Structure**:
```
config/
├── md/
│   └── binance/config.json          # Market data config
└── td/
    └── binance/<account>.json       # Trading account config (API keys)
```

**NEVER commit**: `access_key`, `secret_key`, `passphrase` (see `.doc/40_config/dangerous_keys.md`)

**Testnet vs Mainnet**: Hardcoded in `core/extensions/binance/include/common.h:18-71` (not runtime configurable)

## Documentation Updates

After code changes, run:
```
follow .doc/UPDATE.md
```

This automates:
1. Detecting which `.doc/` files need updates
2. Updating contracts (`30_contracts/`) for API changes
3. Updating modules (`10_modules/`) for implementation changes
4. Updating interactions (`20_interactions/`) for flow changes
5. Updating operations (`90_operations/`) for CLI changes

**Verification scripts**:
```bash
# Validate code references (file:line format)
python3 .doc/90_operations/scripts/verify_code_refs.py

# Check markdown links
python3 .doc/90_operations/scripts/check_links.py

# Update token estimates
python3 .doc/90_operations/scripts/estimate_tokens.py
```

## Development Workflow

1. **Load context**: `follow .doc/START.md`
2. **Make changes**: Edit on host, build in container
3. **Test**: Run affected services (master → ledger → md → td → strategy)
4. **Update docs**: `follow .doc/UPDATE.md`
5. **Verify**: Run validation scripts

## Key Constraints

- **Single-threaded strategy execution**: Callbacks must complete in <1ms
- **Journal is append-only**: Cannot delete events (only replay from specific time)
- **Order IDs are local**: `order_id` is strategy-local, `ex_order_id` is exchange-assigned
- **Depth arrays are fixed-size**: 10 levels max, sparse levels filled with zeros
- **Python/C++ boundary**: pybind11 overhead ~5-10μs per callback

## Common Pitfalls

1. **❌ Running outside Docker**: NEVER run `python3 dev_run.py` directly on host. ALWAYS use `docker exec godzilla-dev`
2. **❌ Not using PM2**: NEVER manage processes manually. ALWAYS use PM2 for process management
3. **❌ Wrong startup order**: Master must start first, then Ledger, then gateways, then strategies (5-second delays)
4. **❌ Wrong paths**: Inside container, all paths start with `/app/` not `/home/huyifan/projects/godzilla-evan/`
5. **Depth indexing**: `bid_price[0]` is **best** bid (highest), not worst
6. **Order state**: `ex_order_id` is empty until exchange confirms (status=Submitted)
7. **Config files**: Must match CLI `-a` account name exactly
8. **Journal pollution**: `CLEAR_JOURNAL=1` deletes all history (dev only)

## Quick Reference Commands

### Essential Commands (Copy-Paste Ready)

```bash
# Start all services
docker exec -it godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"

# Check status
docker exec godzilla-dev pm2 list

# View logs
docker exec -it godzilla-dev pm2 logs

# Stop all services
docker exec godzilla-dev pm2 stop all && docker exec godzilla-dev pm2 delete all

# Enter container
docker exec -it godzilla-dev bash
```

### Troubleshooting

```bash
# Container not running?
docker-compose up -d

# Services keep restarting?
docker exec -it godzilla-dev pm2 logs --err --lines 100

# Clear everything and start fresh
docker exec godzilla-dev pm2 stop all
docker exec godzilla-dev pm2 delete all
docker exec godzilla-dev bash -c "find ~/.config/kungfu/app/ -name '*.journal' | xargs rm -f"
```

## Documentation Reference

- Architecture: `.doc/00_index/ARCHITECTURE.md`
- Strategy development: `.doc/10_modules/strategy_framework.md`
- API reference: `.doc/30_contracts/strategy_context_api.md`
- Order lifecycle: `.doc/20_interactions/order_lifecycle_flow.md`
- PM2 operations: `.doc/90_operations/pm2_startup_guide.md`
- CLI commands: `.doc/90_operations/cli_operations_guide.md`
