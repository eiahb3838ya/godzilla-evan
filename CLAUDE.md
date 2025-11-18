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

**Startup sequence** (critical - must follow order):
```bash
# Terminal 1: Master service (registry)
python3 core/python/dev_run.py -l info master

# Wait 5 seconds, then Terminal 2: Ledger (position tracking)
python3 core/python/dev_run.py -l info ledger

# Wait 5 seconds, then Terminal 3: Market data gateway
python3 core/python/dev_run.py -l trace md -s binance

# Wait 5 seconds, then Terminal 4: Trading gateway
python3 core/python/dev_run.py -l info td -s binance -a my_account

# Wait 5 seconds, then Terminal 5: Strategy
python3 core/python/dev_run.py -l info strategy -n demo -p strategies/demo_spot.py
```

**Or use PM2** (see `.doc/90_operations/pm2_startup_guide.md`):
```bash
cd scripts/binance_test
./run.sh start
```

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

1. **Wrong startup order**: Master must start first, then Ledger, then gateways, then strategies
2. **Depth indexing**: `bid_price[0]` is **best** bid (highest), not worst
3. **Order state**: `ex_order_id` is empty until exchange confirms (status=Submitted)
4. **Config files**: Must match CLI `-a` account name exactly
5. **Journal pollution**: `CLEAR_JOURNAL=1` deletes all history (dev only)

## Reference

- Architecture: `.doc/00_index/ARCHITECTURE.md`
- Strategy development: `.doc/10_modules/strategy_framework.md`
- API reference: `.doc/30_contracts/strategy_context_api.md`
- Order lifecycle: `.doc/20_interactions/order_lifecycle_flow.md`
- CLI commands: `.doc/90_operations/cli_operations_guide.md`
