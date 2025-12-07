# Godzilla hf-live Project Overview

## Project Purpose
Low-latency cryptocurrency trading system with factor engine and model calculation support.
The hf-live component integrates a factor calculation framework for real-time market data analysis.

## Tech Stack
- **Language**: C++17
- **Data Types**: Market data (Depth, Trade, Bar) from `include/market_data_types.h`
- **Architecture**: Factor-based computation with event-driven design

## Key Files and Structure
- `include/market_data_types.h` - Market data structures (Depth, Trade, Bar)
- `factors/_comm/` - Common factor infrastructure (base classes, registry)
  - `factor_entry_base.h` - Base class for all factors
  - `factor_entry_registry.h` - Factor registration and creation system
  - `core.h` - Type definitions and interfaces
- `factors/demo/` - Demo factor implementation (to be created)
- `app_live/engine/` - Factor and model calculation engines
- `adapter/signal_api.cpp` - Signal API implementation

## Build System
- CMake 3.15+
- Current CMakeLists.txt located at `hf-live/CMakeLists.txt`
- Factor modules can be dynamically discovered (currently commented out)
- Build output: shared library `libsignal.so`

## Coding Conventions
- Namespace: `factors::demo` for demo factor
- Class inheritance: Must inherit from `factors::comm::FactorEntryBase`
- Method overrides: `DoOnAddQuote()`, `DoOnAddTrans()`, `DoOnUpdateFactors()`
- Factor values: Stored in `fvals_` vector (type: float)
- Registration: Use `REGISTER_FACTOR_AUTO()` macro or `REGISTER_FACTOR_WITH_STATIC_INIT()`

## Data Types
- `hf::Depth` - 10-level order book with bid/ask prices and volumes
- `hf::Trade` - Market trade with price, volume, and trade ID
- `hf::Bar` - Candlestick data with OHLCV
- All timestamps are int64_t in nanoseconds or milliseconds

## Build/Test Commands
- Build: `cd hf-live && cmake -B build && cmake --build build`
- Compile test: `cd hf-live && ./test_compile.sh`

## Demo Factor Requirements
- 2 simple factors to validate the framework
- Factor 1: `bid_ask_ratio` = bid_volume[0] / ask_volume[0]
- Factor 2: `trade_volume_ma` = moving average of last 10 trade volumes
- Files needed: factor_entry.h, factor_entry.cpp, meta_config.h, CMakeLists.txt
