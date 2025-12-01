---
title: Strategy Context API Contract
updated_at: 2025-11-17
owner: core-dev
lang: en
tags: [contract, api, strategy, context, trading, reference]
code_refs:
  - core/cpp/wingchun/include/kungfu/wingchun/strategy/context.h:25-131
  - core/python/kungfu/wingchun/strategy.py:144-184
purpose: "Complete API specification for the strategy context object available in all callbacks"
---

# Strategy Context API Contract

## Purpose

The `context` object is the primary interface between user strategy code and the trading system. It provides all methods needed for market data subscription, order management, account handling, configuration, and state management. This contract defines the complete API surface with parameter specifications, return values, preconditions, and side effects.

## API Categories

1. [Time & Lifecycle](#time--lifecycle)
2. [Market Data Subscription](#market-data-subscription)
3. [Order Management](#order-management)
4. [Account Management](#account-management)
5. [Configuration & State](#configuration--state)
6. [Logging](#logging)
7. [Advanced Features](#advanced-features)

---

## Time & Lifecycle

### now()

**Purpose:** Get current system time in nanoseconds.

**Signature:**
```cpp
int64_t now() const
```

**Python:**
```python
timestamp = context.now()
```

**Returns:** `int64_t` - Current time in nanoseconds since Unix epoch.

**Use Cases:**
- Timestamp calculations
- Timeout detection
- Performance measurement

**Example:**
```python
start_time = context.now()
# ... do work ...
elapsed_ns = context.now() - start_time
elapsed_ms = elapsed_ns / 1e6
context.log().info(f"Processing took {elapsed_ms:.2f} ms")
```

---

### add_timer()

**Purpose:** Schedule one-time callback at specific timestamp.

**Signature:**
```cpp
void add_timer(int64_t nanotime, const std::function<void(event_ptr)> &callback)
```

**Python:**
```python
def timer_callback(context, event):
    context.log().info("Timer fired!")

context.add_timer(nanotime, timer_callback)
```

**Parameters:**
- `nanotime` (int64): Absolute timestamp in nanoseconds
- `callback` (function): Callback function(context, event)

**Preconditions:**
- `nanotime` must be in the future
- Callback must accept (context, event) parameters

**Side Effects:**
- Callback registered in event loop
- Fires exactly once at specified time

**Example:**
```python
def pre_start(context):
    # Fire in 5 seconds
    future_time = context.now() + 5 * 1e9
    context.add_timer(future_time, lambda ctx, evt: ctx.log().info("5 seconds elapsed"))
```

---

### add_time_interval()

**Purpose:** Schedule recurring callback at fixed interval.

**Signature:**
```cpp
void add_time_interval(int64_t duration, const std::function<void(event_ptr)> &callback)
```

**Python:**
```python
def interval_callback(context, event):
    context.log().info("Heartbeat")

context.add_time_interval(duration_ns, interval_callback)
```

**Parameters:**
- `duration` (int64): Interval in nanoseconds (e.g., 1e9 = 1 second)
- `callback` (function): Callback function(context, event)

**Side Effects:**
- Callback fires repeatedly every `duration` nanoseconds
- Continues until strategy stops

**Example:**
```python
def pre_start(context):
    # Log every 10 seconds
    context.add_time_interval(10 * 1e9, lambda ctx, evt: ctx.log().info("Heartbeat"))
```

---

## Market Data Subscription

### subscribe()

**Purpose:** Subscribe to market depth (Level 2 order book).

**Signature:**
```cpp
void subscribe(const std::string &source,
               const std::vector<std::string> &instruments,
               InstrumentType inst_type = InstrumentType::Spot,
               const std::string &exchange = "")
```

**Python:**
```python
context.subscribe(source, symbols, instrument_type, exchange)
```

**Parameters:**
- `source` (string): Market data source ID (e.g., "binance")
- `symbols` (list[string]): List of symbols (e.g., ["btcusdt", "ethusdt"])
- `instrument_type` (InstrumentType): Spot, FFuture, etc.
- `exchange` (string): Exchange ID (e.g., "BINANCE")

**Triggers:** `on_depth(context, depth)` callback

**Preconditions:**
- Must be called in `pre_start()` or `post_start()`
- Market data source must be configured and running

**Side Effects:**
- WebSocket subscription initiated
- `on_depth()` fires on every depth update (~10/sec for Binance)

**Example:**
```python
def pre_start(context):
    config = context.get_config()
    context.subscribe(
        config["md_source"],      # "binance"
        [config["symbol"]],        # ["btcusdt"]
        InstrumentType.Spot,       # Spot market
        Exchange.BINANCE           # "BINANCE"
    )
```

**See Also:** [Depth Object Contract](depth_object_contract.md)

---

### unsubscribe()

**Purpose:** Unsubscribe from market depth.

**Signature:**
```cpp
void unsubscribe(const std::string &source,
                 const std::vector<std::string> &instruments,
                 InstrumentType inst_type = InstrumentType::Spot,
                 const std::string &exchange = "")
```

**Python:**
```python
context.unsubscribe(source, symbols, instrument_type, exchange)
```

**Side Effects:**
- WebSocket subscription cancelled
- `on_depth()` stops firing for those symbols

---

### subscribe_ticker()

**Purpose:** Subscribe to best bid/ask ticker (faster, less detailed than depth).

**Signature:**
```cpp
void subscribe_ticker(const std::string &source,
                      const std::vector<std::string> &instruments,
                      InstrumentType inst_type = InstrumentType::Spot,
                      const std::string &exchange = "")
```

**Python:**
```python
context.subscribe_ticker(source, symbols, instrument_type, exchange)
```

**Triggers:** `on_ticker(context, ticker)` callback

**Use Cases:**
- Price monitoring without full depth
- Spread calculation
- Lower bandwidth than depth

**See Also:** [Ticker Object Contract](ticker_object_contract.md)

---

### subscribe_trade()

**Purpose:** Subscribe to trade feed (public trades on exchange).

**Signature:**
```cpp
void subscribe_trade(const std::string &source,
                     const std::vector<std::string> &instruments,
                     InstrumentType inst_type = InstrumentType::Spot,
                     const std::string &exchange = "")
```

**Python:**
```python
context.subscribe_trade(source, symbols, instrument_type, exchange)
```

**Triggers:** `on_transaction(context, transaction)` callback

**Note:** This is for **public market trades**, not your own fills (use `on_trade()` for that).

---

### subscribe_index_price()

**Purpose:** Subscribe to index price updates (for futures).

**Signature:**
```cpp
void subscribe_index_price(const std::string &source,
                           const std::vector<std::string> &instruments,
                           InstrumentType inst_type = InstrumentType::Spot,
                           const std::string &exchange = "")
```

**Python:**
```python
context.subscribe_index_price(source, symbols, instrument_type, exchange)
```

**Triggers:** `on_index_price(context, index_price)` callback

**Use Cases:**
- Futures trading (index vs mark price)
- Arbitrage detection

---

### subscribe_all()

**Purpose:** Subscribe to ALL symbols on an exchange (use sparingly).

**Signature:**
```cpp
void subscribe_all(const std::string &source)
```

**Python:**
```python
context.subscribe_all(source)
```

**Warning:**
- Very high bandwidth
- May overwhelm strategy with callbacks
- Use only for market-making or multi-symbol strategies

---

## Order Management

### insert_order()

**Purpose:** Place a new order.

**Signature:**
```cpp
uint64_t insert_order(
    const std::string &symbol,
    InstrumentType inst_type,
    const std::string &exchange,
    const std::string &account,
    double limit_price,
    double volume,
    OrderType type,
    Side side,
    TimeCondition time = TimeCondition::GTC,
    Direction position_side = Direction::Long,
    bool reduce_only = false
)
```

**Python:**
```python
order_id = context.insert_order(
    symbol,           # "btcusdt"
    instrument_type,  # InstrumentType.Spot
    exchange,         # Exchange.BINANCE
    account,          # "my_account"
    price,            # 50000.0 (0 for market)
    volume,           # 0.1
    order_type,       # OrderType.Limit
    side              # Side.Buy or Side.Sell
)
```

**Parameters:**
- `symbol` (string): Trading pair (lowercase, e.g., "btcusdt")
- `instrument_type` (InstrumentType): Spot, FFuture, etc.
- `exchange` (string): Exchange ID ("BINANCE")
- `account` (string): Account identifier
- `limit_price` (double): Limit price (0 for market orders)
- `volume` (double): Order quantity
- `type` (OrderType): Limit, Market, etc.
- `side` (Side): Buy or Sell
- `time` (TimeCondition): GTC, IOC, FOK, etc. (default: GTC)
- `position_side` (Direction): Long or Short (futures only, default: Long)
- `reduce_only` (bool): Only reduce position (futures, default: false)

**Returns:** `uint64_t` - Local order ID (immediately assigned)

**Preconditions:**
- Account must be added via `add_account()` in `pre_start()`
- Symbol must exist in exchange
- Price/volume must be within exchange limits

**Side Effects:**
- Order written to journal
- Order sent to trader gateway asynchronously
- `on_order()` callback fires with status updates

**Example:**
```python
def on_depth(context, depth):
    order_id = context.insert_order(
        symbol="btcusdt",
        instrument_type=InstrumentType.Spot,
        exchange=Exchange.BINANCE,
        account="my_account",
        price=depth.bid_price[0],  # Sell at best bid
        volume=0.1,
        order_type=OrderType.Limit,
        side=Side.Sell
    )
    context.log().info(f"Placed order {order_id}")
```

**Error Handling:**
- Returns order_id even if order will fail
- Errors reported via `on_order()` with status=Error

**See Also:** [Order Object Contract](order_object_contract.md)

---

### cancel_order()

**Purpose:** Cancel an existing order.

**Signature:**
```cpp
uint64_t cancel_order(
    const std::string &account,
    uint64_t order_id,
    std::string &symbol,
    std::string &ex_order_id,
    InstrumentType inst_type = InstrumentType::Spot
)
```

**Python:**
```python
context.cancel_order(account, order_id, symbol, ex_order_id, instrument_type)
```

**Parameters:**
- `account` (string): Account identifier
- `order_id` (uint64): Local order ID (from insert_order)
- `symbol` (string): Trading symbol
- `ex_order_id` (string): Exchange order ID (from on_order callback)
- `instrument_type` (InstrumentType): Spot, FFuture, etc.

**Returns:** `uint64_t` - Cancellation action ID

**Preconditions:**
- Order must exist
- `ex_order_id` must be populated (wait for on_order with status=Submitted)

**Side Effects:**
- Cancellation request sent to exchange
- `on_order()` fires with status=Cancelled if successful

**Example:**
```python
def on_order(context, order):
    # Save exchange ID when order is submitted
    if order.status == OrderStatus.Submitted:
        context.set_object(f"ex_id_{order.order_id}", order.ex_order_id)

def some_callback(context, data):
    # Later, cancel the order
    order_id = context.get_object("my_order_id")
    ex_order_id = context.get_object(f"ex_id_{order_id}")
    if ex_order_id:
        context.cancel_order("my_account", order_id, "btcusdt", ex_order_id, InstrumentType.Spot)
```

---

### query_order()

**Purpose:** Query current order status from exchange.

**Signature:**
```cpp
uint64_t query_order(
    const std::string &account,
    uint64_t order_id,
    std::string &ex_order_id,
    InstrumentType inst_type,
    const std::string &symbol = {}
)
```

**Python:**
```python
context.query_order(account, order_id, ex_order_id, instrument_type, symbol)
```

**Side Effects:**
- Query sent to exchange
- `on_order()` fires with current status

**Use Cases:**
- Refresh stale order status
- Reconcile state after reconnection

---

## Account Management

### add_account()

**Purpose:** Register a trading account for use by strategy.

**Signature:**
```cpp
void add_account(const std::string &source, const std::string &account)
```

**Python:**
```python
context.add_account(source, account_id)
```

**Parameters:**
- `source` (string): Trading gateway ID (e.g., "binance")
- `account` (string): Account identifier

**Preconditions:**
- MUST be called in `pre_start()` before any order operations
- Account must be configured in system database

**Side Effects:**
- Account book initialized
- Trading gateway connection established
- Position and balance loaded

**Example:**
```python
def pre_start(context):
    config = context.get_config()
    context.add_account(config["td_source"], config["account"])
```

---

### list_accounts()

**Purpose:** Get list of all registered accounts.

**Signature:**
```cpp
std::vector<location_ptr> list_accounts()
```

**Python:**
```python
accounts = context.list_accounts()
```

**Returns:** List of account location objects

---

### set_account_cash_limit()

**Purpose:** Set available cash limit for an account (simulated accounts).

**Signature:**
```cpp
void set_account_cash_limit(
    const std::string &account,
    const std::string &coin,
    double limit
)
```

**Python:**
```python
context.set_account_cash_limit(account, coin, limit)
```

**Parameters:**
- `account` (string): Account identifier
- `coin` (string): Currency (e.g., "USDT", "BTC")
- `limit` (double): Available balance

**Use Cases:**
- Backtesting with simulated balances
- Risk management (hard limit on capital)

**Example:**
```python
def pre_start(context):
    context.add_account("binance", "my_account")
    context.set_account_cash_limit("my_account", "USDT", 10000.0)
    context.set_account_cash_limit("my_account", "BTC", 1.0)
```

---

### get_account_cash_limit()

**Purpose:** Get current cash limit for account.

**Signature:**
```cpp
double get_account_cash_limit(const std::string &account, const std::string &coin)
```

**Python:**
```python
limit = context.get_account_cash_limit(account, coin)
```

**Returns:** `double` - Current limit

---

### get_account_book()

**Purpose:** Get account book with positions, orders, and balances.

**Signature:**
```python
book = context.get_account_book(source, account_id)
```

**Returns:** `AccountBook` object

**Properties:**
- `book.active_orders` - List of pending/partial orders
- `book.get_asset(coin)` - Get balance for currency

**Example:**
```python
def on_depth(context, depth):
    config = context.get_config()
    book = context.get_account_book(config["td_source"], config["account"])

    context.log().info(f"Active orders: {len(book.active_orders)}")

    # Cancel all active orders
    for order in book.active_orders:
        if order['ex_order_id']:
            context.cancel_order(config["account"], order['order_id'], order['symbol'], order['ex_order_id'], InstrumentType.Spot)
```

---

### get_market_info()

**Purpose:** Get symbol metadata (price precision, lot size, etc.).

**Signature:**
```cpp
const std::string get_market_info(const std::string &symbol, const std::string &exchange)
```

**Python:**
```python
info = context.get_market_info(symbol, exchange, instrument_type)
```

**Returns:** Dictionary with:
- `price_tick` - Minimum price increment
- `contract_multiplier` - Contract size
- `instrument_type` - Spot, Futures, etc.

**Example:**
```python
def pre_start(context):
    info = context.get_market_info("btcusdt", Exchange.BINANCE, InstrumentType.Spot)
    context.log().info(f"Price tick: {info['price_tick']}")
```

---

## Configuration & State

### get_config()

**Purpose:** Load strategy configuration from JSON file.

**Signature:**
```python
config = context.get_config()
```

**Returns:** `dict` - Configuration loaded from `conf.json`

**Example:**
```python
def pre_start(context):
    config = context.get_config()
    symbol = config["symbol"]
    account = config["account"]
    md_source = config["md_source"]
```

**Configuration File:** `strategies/<strategy_name>/conf.json`

---

### reload_config()

**Purpose:** Reload configuration without restarting strategy.

**Signature:**
```python
context.reload_config()
```

**Use Cases:**
- Hot-reload parameters during runtime
- A/B testing different settings

**Example:**
```python
def on_ticker(context, ticker):
    context.reload_config()
    config = context.get_config()
    threshold = config.get("threshold", 50000.0)

    if ticker.bid_price > threshold:
        # ... trading logic ...
```

---

### set_object()

**Purpose:** Store object in memory (per-strategy instance).

**Signature:**
```python
context.set_object(key, value)
```

**Parameters:**
- `key` (string): Object key (unique per strategy)
- `value` (any): Object to store

**Storage:** In-memory only (lost on restart)

**Example:**
```python
def pre_start(context):
    context.set_object("order_count", 0)
    context.set_object("last_price", 0.0)
    context.set_object("threshold", 50000.0)
```

---

### get_object()

**Purpose:** Retrieve stored object from memory.

**Signature:**
```python
value = context.get_object(key)
```

**Returns:** Stored value or `None` if key doesn't exist

**Example:**
```python
def on_depth(context, depth):
    count = context.get_object("order_count") or 0
    count += 1
    context.set_object("order_count", count)

    if count > 100:
        context.log().warning("Placed 100 orders, stopping")
```

---

## Logging

### log()

**Purpose:** Get logger for current strategy instance.

**Signature:**
```python
logger = context.log()
```

**Returns:** Logger object

**Methods:**
- `logger.info(msg)` - Info level
- `logger.warning(msg)` - Warning level
- `logger.error(msg)` - Error level
- `logger.debug(msg)` - Debug level
- `logger.trace(msg)` - Trace level (verbose)

**Example:**
```python
def on_order(context, order):
    if order.status == OrderStatus.Error:
        context.log().error(f"Order {order.order_id} rejected: {order.error_code}")
    elif order.status == OrderStatus.Filled:
        context.log().info(f"Order {order.order_id} filled at {order.avg_price}")
```

**Log Location:** `KF_HOME/strategy/<strategy_name>/strategy.log`

---

## Advanced Features

### adjust_leverage()

**Purpose:** Change leverage for futures position.

**Signature:**
```cpp
void adjust_leverage(
    const std::string &account,
    const std::string &symbol,
    Direction &position_side,
    const int leverage
)
```

**Python:**
```python
context.adjust_leverage(account, symbol, position_side, leverage)
```

**Use Cases:** Futures trading only

---

### merge_positions()

**Purpose:** Merge hedged positions (futures).

**Signature:**
```cpp
void merge_positions(
    const std::string &account,
    const std::string &symbol,
    const int amount,
    InstrumentType inst_type = InstrumentType::FFuture
)
```

**Use Cases:** Futures position management

---

### query_positions()

**Purpose:** Query current positions from exchange.

**Signature:**
```cpp
bool query_positions(
    const std::string &account,
    const std::string &symbol,
    InstrumentType inst_type = InstrumentType::FFuture
)
```

**Returns:** `bool` - Success/failure

**Side Effects:** Position data refreshed

---

## Related Documentation

### Contracts
- [Order Object Contract](order_object_contract.md) - Order structure
- [Depth Object Contract](depth_object_contract.md) - Market depth
- [Binance Config Contract](binance_config_contract.md) - Configuration schema

### Modules
- [Strategy Framework](../10_modules/strategy_framework.md) - Complete strategy guide
- [Yijinjing](../10_modules/yijinjing.md) - Event system
- [Wingchun](../10_modules/wingchun.md) - Trading gateway

### Examples
- [demo_spot.py](../../strategies/demo_spot.py) - Usage examples
- [helloworld](../../strategies/helloworld/) - Basic tutorial

## Version History

- **2025-11-17:** Initial API contract documentation
- **2025-03-03:** Modified from original kungfu (Keren Dong) by kx@godzilla.dev
