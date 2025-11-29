---
title: Strategy Framework - Trading Strategy Development System
updated_at: 2025-11-17
owner: core-dev
lang: en
tags: [module, strategy, trading, lifecycle, callbacks, context-api, python]
code_refs:
  - core/python/kungfu/wingchun/strategy.py:35-184
  - core/cpp/wingchun/src/strategy/context.cpp:473
  - core/cpp/wingchun/src/strategy/runner.cpp:215
  - strategies/demo_spot.py
purpose: "Complete guide to writing trading strategies: lifecycle, context API, event callbacks, and best practices"
---

# Strategy Framework Module

## Overview

The Strategy Framework provides a Python-first development experience for writing automated trading strategies. It handles event-driven architecture, account management, order execution, and market data subscriptions through a simple callback-based API.

**Key Features:**
- Event-driven callbacks (`on_depth`, `on_order`, `on_trade`, etc.)
- Context object with complete trading API
- Automatic account book management and PnL tracking
- Configuration hot-reload without restart
- In-memory object caching for strategy state
- Backtesting engine integration

**Languages:** Python (user code) → C++ (runtime via pybind11)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           User Strategy (Python)                    │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────┐│
│  │ pre_start()  │  │ on_depth()  │  │ on_order() ││
│  │ post_start() │  │ on_ticker() │  │ on_trade() ││
│  │ pre_stop()   │  │ on_bar()    │  └────────────┘│
│  └──────────────┘  └─────────────┘                 │
└──────────────┬──────────────────────────────────────┘
               │ context.insert_order()
               │ context.subscribe()
               │ context.get_config()
               ▼
┌──────────────────────────────────────────────────────┐
│         Strategy Context (Python Wrapper)            │
│     core/python/kungfu/wingchun/strategy.py          │
└──────────────┬───────────────────────────────────────┘
               │ pywingchun bindings
               ▼
┌──────────────────────────────────────────────────────┐
│      C++ Strategy Runtime (Event Loop)               │
│   core/cpp/wingchun/src/strategy/runner.cpp          │
│                                                       │
│  ┌──────────────┐  Journal  ┌──────────────┐        │
│  │ Market Data  │ ────────→ │   Strategy   │        │
│  │   Reader     │           │  Event Loop  │        │
│  └──────────────┘           └──────┬───────┘        │
│                                    │                 │
│  ┌──────────────┐                  ▼                 │
│  │    Trader    │ ←──────── Python Callback         │
│  │   Gateway    │                                    │
│  └──────────────┘                                    │
└──────────────────────────────────────────────────────┘
```

## Strategy Lifecycle

**Source:** [core/python/kungfu/wingchun/strategy.py:144-186](../../core/python/kungfu/wingchun/strategy.py)

### Phase 1: Pre-Start (Initialization)

```python
def pre_start(context):
    """
    Called ONCE before event loop starts.
    Use for:
    - Add accounts
    - Subscribe to market data
    - Load configuration
    - Initialize strategy state
    """
    config = context.get_config()

    # Add trading account
    context.add_account(config["td_source"], config["account"])

    # Subscribe to market depth
    context.subscribe(
        config["md_source"],       # Data source (e.g., "binance")
        [config["symbol"]],         # Symbols (e.g., ["btc_usdt"] - MUST use lowercase_lowercase)
        InstrumentType.Spot,        # Instrument type
        Exchange.BINANCE            # Exchange
    )

    # Subscribe to ticker (best bid/ask)
    context.subscribe_ticker(
        config["md_source"],
        [config["symbol"]],
        InstrumentType.Spot,
        Exchange.BINANCE
    )

    # Initialize strategy state
    context.set_object("order_count", 0)
    context.set_object("last_depth_time", 0)

    context.log().info("Strategy initialized")
```

**Execution:** Runs in main thread BEFORE event loop starts.
**Blocking:** Safe to perform expensive operations (database loading, API calls).
**Errors:** Exceptions here will prevent strategy from starting.

### Phase 2: Post-Start (Optional)

```python
def post_start(context):
    """
    Called ONCE after event loop starts.
    Use for:
    - Query initial account state
    - Warm-up calculations
    - Post-initialization setup
    """
    book = context.get_account_book("binance", "my_account")
    context.log().info(f"Starting with {len(book.active_orders)} active orders")
```

**Execution:** Runs after event loop starts, before first market data callback.
**Rarely Used:** Most strategies only need `pre_start()`.

### Phase 3: Event Loop (Main Operation)

Once started, strategy enters event-driven mode. Callbacks fire on journal events:

```python
def on_depth(context, depth):
    """Market depth update (10 levels)"""
    if depth.bid_price[0] > context.get_object("threshold"):
        context.insert_order(...)

def on_ticker(context, ticker):
    """Best bid/ask update (faster than depth)"""
    spread = ticker.ask_price - ticker.bid_price
    context.log().info(f"Spread: {spread}")

def on_order(context, order):
    """Order status change (Pending → Submitted → Filled/Cancelled)"""
    if order.status == OrderStatus.Filled:
        context.log().info(f"Order {order.order_id} filled at {order.avg_price}")

def on_trade(context, trade):
    """Individual fill/execution"""
    context.log().info(f"Filled {trade.volume} @ {trade.price}")
```

**Execution:** Callbacks run in event loop thread (single-threaded).
**Blocking:** MUST complete quickly (<1ms). No blocking I/O, no infinite loops.
**Order:** Callbacks fire in timestamp order from journal.

### Phase 4: Shutdown

```python
def pre_stop(context):
    """Called BEFORE event loop stops."""
    context.log().info("Shutting down...")
    # Cancel all active orders
    book = context.get_account_book("binance", "my_account")
    for order in book.active_orders:
        if order['ex_order_id']:
            context.cancel_order(...)

def post_stop(context):
    """Called AFTER event loop stops."""
    # Cleanup resources
    context.log().info("Strategy stopped")
```

**Trigger:** SIGTERM, SIGINT, or explicit stop command.
**Graceful:** Pre-stop allows cleanup before event loop terminates.

## Context API Reference

The `context` object provides the complete trading API. See [Strategy Context API Contract](../30_contracts/strategy_context_api.md) for full specification.

### Market Data Subscription

```python
# Subscribe to order book depth (10 levels)
context.subscribe(source, symbols, instrument_type, exchange)
# → Triggers on_depth() callback

# Subscribe to ticker (best bid/ask only)
context.subscribe_ticker(source, symbols, instrument_type, exchange)
# → Triggers on_ticker() callback

# Subscribe to trade feed (not commonly used)
context.subscribe_trade(source, symbols, instrument_type, exchange)
# → Triggers on_transaction() callback

# Subscribe to all symbols (use sparingly)
context.subscribe_all(source, instrument_type, exchange)
```

**Example:**
```python
context.subscribe(
    "binance",                      # Market data source
    ["btc_usdt", "eth_usdt"],      # Multiple symbols (MUST use lowercase_lowercase format)
    InstrumentType.Spot,            # Spot market
    Exchange.BINANCE                # Exchange enum
)
```

**IMPORTANT - Symbol Format:**
- ✓ Correct: `"btc_usdt"`, `"eth_usdt"` (lowercase with underscore)
- ✗ Wrong: `"btcusdt"`, `"BTCUSDT"`, `"BTC_USDT"`, `"btc-usdt"`
- See [Symbol Naming Convention](../40_config/symbol_naming_convention.md) for details

### Order Management

```python
# Place order
order_id = context.insert_order(
    symbol,           # "btc_usdt" (MUST use lowercase_lowercase format)
    instrument_type,  # InstrumentType.Spot
    exchange,         # Exchange.BINANCE
    account,          # "my_account"
    price,            # 50000.0 (0 for market order)
    volume,           # 0.1
    order_type,       # OrderType.Limit
    side              # Side.Buy or Side.Sell
)
# Returns: Local order_id (uint64)
# Triggers: on_order() callback when status changes

# Cancel order
context.cancel_order(
    account,          # "my_account"
    order_id,         # Local order ID
    symbol,           # "btc_usdt" (same format as insert_order)
    ex_order_id,      # Exchange order ID (from on_order callback)
    instrument_type   # InstrumentType.Spot
)
# Triggers: on_order() with status=Cancelled

# Query order status
order = context.query_order(account, order_id, ex_order_id, instrument_type)
```

**Important:** `order_id` is assigned immediately but order is asynchronous. Monitor `on_order()` for status.

### Account Management

```python
# Add account (call in pre_start)
context.add_account(source, account_id)
# Example: context.add_account("binance", "my_account")

# Get account book (positions, orders, PnL)
book = context.get_account_book(source, account_id)
# Returns: AccountBook object
# book.active_orders - List of pending/partial orders
# book.get_asset(currency) - Get balance for currency

# Set cash limit (for simulated accounts)
context.set_account_cash_limit(source, exchange, account, coin, limit)
# Example: context.set_account_cash_limit("binance", "BINANCE", "my_account", "USDT", 10000.0)
```

### Configuration

```python
# Get strategy config (from JSON file)
config = context.get_config()
# Returns: dict loaded from strategies/<name>/conf.json
# Example: config["symbol"], config["account"], etc.

# Reload config without restart
context.reload_config()
config = context.get_config()  # Now has updated values
```

**Config File Example:** `strategies/demo_spot/conf.json`
```json
{
  "name": "demo_spot",
  "md_source": "binance",
  "td_source": "binance",
  "account": "my_account",
  "symbol": "btc_usdt",
  "action": "single"
}
```

**CRITICAL - Symbol Field:**
- The `"symbol"` field MUST use `lowercase_base_underscore_quote` format (e.g., `"btc_usdt"`)
- Wrong format causes:
  1. `IndexError` when placing orders ([book.py:122-123](../../core/python/kungfu/wingchun/book/book.py#L122-L123))
  2. Silent subscription failure (strategy won't receive market data)
  3. Requires C++ rebuild to fix
- See [Symbol Naming Convention](../40_config/symbol_naming_convention.md) for detailed explanation

### State Management

```python
# Store object in memory (per-strategy instance)
context.set_object("key", value)
# Example: context.set_object("order_count", 10)

# Retrieve object
value = context.get_object("key")
# Returns: None if key doesn't exist

# Common pattern: counters
count = context.get_object("order_count") or 0
count += 1
context.set_object("order_count", count)
```

**Storage:** In-memory only (lost on restart). Use for temporary state.
**Isolation:** Objects are per-strategy instance (multiple strategies don't collide).

### Market Information

```python
# Get symbol metadata (lot size, price precision, etc.)
info = context.get_market_info(symbol, exchange, instrument_type)
# Returns: dict with:
#   - price_tick (minimum price increment)
#   - contract_multiplier
#   - instrument_type
#   - etc.
```

### Logging

```python
# Get logger for current strategy
logger = context.log()

# Log messages
logger.info("Order placed")
logger.warning("Low liquidity detected")
logger.error("Failed to place order")
logger.debug("Depth update received")

# Logs written to: KF_HOME/strategy/<strategy_name>/strategy.log
```

**Log Levels:** trace, debug, info, warning, error, critical
**Configuration:** Set `KF_LOG_LEVEL` environment variable.

### Timers

```python
# Add one-time timer (nanoseconds)
def timer_callback(context, event):
    context.log().info("Timer fired!")

context.add_timer(
    nanotime,        # Absolute timestamp (nanoseconds since epoch)
    timer_callback   # Callback function
)

# Add recurring timer (interval)
context.add_time_interval(
    duration,        # Interval in nanoseconds (e.g., 1e9 = 1 second)
    timer_callback   # Callback function
)
```

**Example:** Run callback every 5 seconds
```python
def pre_start(context):
    def heartbeat(context, event):
        context.log().info("Heartbeat")

    context.add_time_interval(5 * 1e9, heartbeat)  # 5 seconds
```

## Event Callbacks

### on_depth(context, depth)

**Trigger:** Order book snapshot received (every 100ms for Binance).
**Object:** [Depth Contract](../30_contracts/depth_object_contract.md)

```python
def on_depth(context, depth):
    # Check symbol (if multiple subscriptions)
    if depth.symbol != "btcusdt":
        return

    # Best bid/ask
    best_bid = depth.bid_price[0]
    best_ask = depth.ask_price[0]
    spread = best_ask - best_bid

    # Calculate liquidity
    bid_liquidity = sum(depth.bid_volume[i] for i in range(10) if depth.bid_price[i] > 0)

    # Trading logic
    if spread < 0.01 * best_bid:  # Tight spread
        context.insert_order(...)
```

**Frequency:** High (10/sec for Binance).
**Use For:** Order book analysis, spread trading, liquidity detection.

### on_ticker(context, ticker)

**Trigger:** Best bid/ask update (faster than depth, less detailed).
**Object:** [Ticker Contract](../30_contracts/ticker_object_contract.md)

```python
def on_ticker(context, ticker):
    mid_price = (ticker.bid_price + ticker.ask_price) / 2
    context.log().info(f"Mid: {mid_price}")
```

**Frequency:** Very high (realtime).
**Use For:** Price monitoring without full depth.

### on_order(context, order)

**Trigger:** Order status change (Pending → Submitted → Filled/Cancelled).
**Object:** [Order Contract](../30_contracts/order_object_contract.md)

```python
def on_order(context, order):
    if order.status == OrderStatus.Submitted:
        context.log().info(f"Order {order.order_id} acknowledged by exchange")
        context.log().info(f"Exchange ID: {order.ex_order_id}")

    elif order.status == OrderStatus.PartialFilledActive:
        context.log().info(f"Partial fill: {order.volume_traded}/{order.volume}")

    elif order.status == OrderStatus.Filled:
        context.log().info(f"Order filled! Avg price: {order.avg_price}, Fee: {order.fee}")

    elif order.status == OrderStatus.Cancelled:
        context.log().warning(f"Order cancelled")

    elif order.status == OrderStatus.Error:
        context.log().error(f"Order rejected: {order.error_code}")
```

**Frequency:** Per order lifecycle (1-10 callbacks per order).
**Use For:** Order tracking, post-trade analysis, error handling.

### on_trade(context, trade)

**Trigger:** Individual fill/execution (subset of order updates).
**Object:** Trade object (similar to Order but represents single fill)

```python
def on_trade(context, trade):
    context.log().info(f"Trade executed: {trade.volume} @ {trade.price}")
```

**Frequency:** Per fill (may be multiple for large orders).
**Use For:** Fill analysis, slippage calculation.

## Best Practices

### 1. Non-Blocking Callbacks

```python
# ❌ WRONG - Blocking operation
def on_depth(context, depth):
    time.sleep(1)  # Blocks event loop!
    response = requests.get("https://api.example.com")  # Network I/O blocks!

# ✅ CORRECT - Quick processing
def on_depth(context, depth):
    context.set_object("last_depth", depth)  # Fast in-memory store
```

**Rule:** Callbacks MUST complete in <1ms. No blocking I/O.

### 2. State Management

```python
# ✅ CORRECT - Use set_object for state
def pre_start(context):
    context.set_object("threshold", 50000.0)
    context.set_object("order_count", 0)

def on_depth(context, depth):
    threshold = context.get_object("threshold")
    if depth.bid_price[0] > threshold:
        count = context.get_object("order_count")
        context.set_object("order_count", count + 1)
```

**Avoid:** Global variables (multiple strategy instances share same process).

### 3. Error Handling

```python
def on_order(context, order):
    if order.status == OrderStatus.Error:
        context.log().error(f"Order failed: {order.error_code}")
        # Implement retry logic or alert
        if "INSUFFICIENT_BALANCE" in order.error_code:
            context.log().error("Out of funds!")
        elif "INVALID_PRICE" in order.error_code:
            # Adjust price and retry
            pass
```

### 4. Configuration Management

```python
def pre_start(context):
    config = context.get_config()

    # Validate config
    required_keys = ["symbol", "account", "md_source", "td_source"]
    for key in required_keys:
        if key not in config:
            context.log().error(f"Missing config key: {key}")
            raise ValueError(f"Missing required config: {key}")

    # Use config values
    context.subscribe(config["md_source"], [config["symbol"]], ...)
```

## Running Strategies

### CLI Execution

```bash
# Run strategy
kfc strategy --name demo_spot --config ./strategies/demo_spot/conf.json

# With specific log level
KF_LOG_LEVEL=debug kfc strategy --name demo_spot --config ./strategies/demo_spot/conf.json

# Backtest mode (not fully implemented)
kfc strategy --name demo_spot --backtest --start 2025-01-01 --end 2025-01-31
```

### File Structure

```
strategies/
└── demo_spot/
    ├── demo_spot.py      # Strategy implementation
    └── conf.json         # Configuration

# Or simpler:
strategies/
└── demo_spot.py          # Strategy + config
```

## Related Documentation

### Contracts
- [Order Object Contract](../30_contracts/order_object_contract.md) - Order structure and state machine
- [Depth Object Contract](../30_contracts/depth_object_contract.md) - Market depth data
- [Strategy Context API](../30_contracts/strategy_context_api.md) - Complete API reference

### Modules
- [Yijinjing](yijinjing.md) - Event sourcing system (journal)
- [Wingchun](wingchun.md) - Trading gateway framework
- [Binance Extension](binance_extension.md) - Exchange connector

### Flows
- [Trading Flow](../20_interactions/trading_flow.md) - Complete order lifecycle
- [Strategy Lifecycle Flow](../20_interactions/strategy_lifecycle_flow.md) - Initialization to shutdown

### Examples
- [demo_spot.py](../../strategies/demo_spot.py) - Minimal working example
- [helloworld](../../strategies/helloworld/) - Tutorial strategy
- [triangular_arbitrage](../../strategies/triangular_arbitrage/) - Advanced example

## Version History

- **2025-11-17:** Initial module card documentation
- **2025-03-03:** Modified from original kungfu (Keren Dong) by kx@godzilla.dev
