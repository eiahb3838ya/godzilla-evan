---
title: Ledger System (Account Book & PnL)
updated_at: 2025-11-17
owner: core-dev
lang: en
tags: [module, ledger, account-book, positions, pnl, risk-management]
dependencies:
  - yijinjing_journal
  - gateway_architecture
code_refs:
  - core/cpp/wingchun/include/kungfu/wingchun/service/ledger.h:1-115
  - core/cpp/wingchun/include/kungfu/wingchun/book/book.h:1-88
  - core/cpp/wingchun/include/kungfu/wingchun/msg.h:947-998
  - core/cpp/wingchun/include/kungfu/wingchun/msg.h:1000-1071
purpose: "Manages account state, positions, assets, and PnL tracking across trading strategies"
tokens_estimate: 5000
---

# Ledger System (Account Book & PnL)

## Overview

The **Ledger System** is responsible for tracking account state, positions, assets (cash balances), and profit/loss (PnL) across all trading strategies. It consumes order fills, trades, and market data to maintain an accurate real-time view of portfolio state.

**Core Concepts**:
- **Book**: Abstract interface for position/asset tracking (one book per account)
- **BookContext**: Manages multiple books, routes events to appropriate books
- **Ledger**: Service-level component that coordinates books and publishes state
- **Asset**: Cash balance per currency (e.g., USDT available, margin, frozen)
- **Position**: Holdings per symbol (e.g., 0.5 BTC long, 1.2 ETH short)

## Architecture

### Component Hierarchy

```
┌────────────────────────────────────────────────────┐
│               Strategy Context                      │
│   • Queries positions via get_account_book()       │
│   • Receives position updates via callbacks        │
└────────────────┬───────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────┐
│              BookContext                           │
│   • Owns multiple Books (one per account)          │
│   • Routes events to appropriate Book              │
│   • Aggregates instrument info                     │
└────────────────┬───────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────┐
│                Book (per account)                  │
│   • on_order()      - Update frozen positions      │
│   • on_trade()      - Update fills, PnL            │
│   • on_depth()      - Update mark prices           │
│   • on_position()   - Sync from exchange           │
│   • on_asset()      - Sync cash balances           │
└────────────────────────────────────────────────────┘
```

### Data Flow

```
Exchange Gateway → Journal → Ledger → Book → Strategy
    (Order)                    ↓
                          Update Position
                               ↓
    (Trade)              Calculate PnL
                               ↓
    (Depth)              Update Mark Price
                               ↓
                        Publish Position Event → Strategy Callback
```

## Core Components

### 1. Book (Abstract Base)

**File**: [core/cpp/wingchun/include/kungfu/wingchun/book/book.h:22-49](../../core/cpp/wingchun/include/kungfu/wingchun/book/book.h)

**Purpose**: Abstract interface for account state tracking

**Virtual Methods**:
```cpp
class Book {
public:
    // Market data (for mark-to-market valuation)
    virtual void on_depth(event_ptr event, const Depth &depth) = 0;

    // Order/trade updates (for position tracking)
    virtual void on_order_input(event_ptr event, const OrderInput &input) = 0;
    virtual void on_order(event_ptr event, const Order &order) = 0;
    virtual void on_trade(event_ptr event, const MyTrade &trade) = 0;

    // Exchange sync (query responses)
    virtual void on_position(event_ptr event, const Position& position) = 0;
    virtual void on_asset(event_ptr event, const Asset& asset) = 0;

    bool is_ready() const { return ready_; }  // True after initial sync
};
```

**Implementation Pattern**:
Each account has a concrete `Book` subclass that implements position/asset logic specific to that account type (spot, futures, margin, etc.).

### 2. BookContext

**File**: [core/cpp/wingchun/include/kungfu/wingchun/book/book.h:51-82](../../core/cpp/wingchun/include/kungfu/wingchun/book/book.h)

**Purpose**: Manages multiple books and routes events

**Key Methods**:
```cpp
class BookContext {
public:
    // Book management
    void add_book(const location_ptr& location, const Book_ptr& book);
    void pop_book(uint32_t location_uid);

    // Instrument registry
    const Instrument& get_inst_info(const std::string &symbol, const std::string &exchange_id) const;
    std::vector<Instrument> all_inst_info() const;

    // Internal monitoring
    void monitor_instruments();
    void monitor_positions(const location_ptr& location, const Book_ptr& book);
};
```

**Lifetime**:
- Created once per strategy/ledger instance
- Persists across entire session
- Maintains `instruments_` registry (symbol metadata)
- Routes events to books based on `location_uid`

### 3. Ledger (Service)

**File**: [core/cpp/wingchun/include/kungfu/wingchun/service/ledger.h:22-109](../../core/cpp/wingchun/include/kungfu/wingchun/service/ledger.h)

**Purpose**: Service-level coordinator for account state

**Key Responsibilities**:
- Publishes broker states (connection status)
- Forwards order requests to traders
- Handles instrument/asset queries
- Provides BookContext access

**Key Methods**:
```cpp
class Ledger : public practice::apprentice {
public:
    BookContext_ptr get_book_context();

    // Order forwarding
    void new_order_single(const event_ptr &event, uint32_t account_location_uid, OrderInput &order_input);
    void cancel_order(const event_ptr &event, uint32_t account_location_uid, uint64_t order_id);

    // State publishing
    void publish(const std::string &msg);
    void publish_broker_states(int64_t trigger_time);

    // Virtual callbacks (implemented by subclasses)
    virtual void on_depth(event_ptr event, const Depth &depth) = 0;
    virtual void on_order(event_ptr event, const Order &order) = 0;
    virtual void on_trade(event_ptr event, const Trade &trade) = 0;
    virtual void on_transaction(event_ptr event, const MyTrade &trade) = 0;
    virtual void on_instruments(const std::vector<Instrument> &instruments) = 0;
};
```

## Data Structures

### Asset (Cash Balance)

**File**: [core/cpp/wingchun/include/kungfu/wingchun/msg.h:947-998](../../core/cpp/wingchun/include/kungfu/wingchun/msg.h)

**Purpose**: Represents cash balance for a single currency/coin

**Fields**:
```cpp
struct Asset {
    int64_t update_time;              // Last update timestamp (nanoseconds)
    uint32_t holder_uid;              // Location UID of holder
    LedgerCategory ledger_category;   // Account type (spot, futures, etc.)

    char coin[64];                    // Currency (e.g., "USDT", "BTC")
    char account_id[64];              // Account identifier
    char exchange_id[64];             // Exchange identifier

    double avail;                     // Available balance
    double margin;                    // Used as margin (futures)
    double frozen;                    // Frozen in pending orders
};
```

**Invariants**:
- `avail` >= 0 (cannot go negative in normal operation)
- `margin` >= 0 (futures only, 0 for spot)
- `frozen` >= 0 (locked in open orders)
- **Total Balance** = `avail + margin + frozen`

**Example**:
```json
{
  "coin": "USDT",
  "account_id": "my_binance_account",
  "exchange_id": "BINANCE",
  "avail": 9500.00,       // Can trade with this
  "margin": 0.00,         // No futures positions
  "frozen": 500.00        // 500 USDT locked in open limit orders
}
```

### Position (Holdings)

**File**: [core/cpp/wingchun/include/kungfu/wingchun/msg.h:1000-1071](../../core/cpp/wingchun/include/kungfu/wingchun/msg.h)

**Purpose**: Represents holdings for a single symbol

**Fields**:
```cpp
struct Position {
    uint32_t strategy_id;             // Strategy owning this position
    int64_t update_time;              // Last update timestamp

    char symbol[64];                  // Trading symbol (e.g., "btcusdt")
    InstrumentType instrument_type;   // Spot, Futures, etc.
    char exchange_id[64];             // Exchange identifier
    char source_id[64];               // Gateway identifier
    char account_id[64];              // Account identifier

    uint32_t holder_uid;              // Location UID
    LedgerCategory ledger_category;   // Account type

    Direction direction;              // Long or Short
    double volume;                    // Quantity held
    int64_t frozen_total;             // Frozen in orders

    // Valuation
    double last_price;                // Most recent market price
    double avg_open_price;            // Average entry price
    double settlement_price;          // Settlement price (futures)

    // PnL
    double margin;                    // Margin used (futures)
    double realized_pnl;              // Closed position PnL
    double unrealized_pnl;            // Open position PnL
};
```

**Invariants**:
- `volume` >= 0 (quantity held)
- `frozen_total` >= 0 and <= `volume` (cannot freeze more than held)
- `realized_pnl` accumulates over time (never decreases unless reset)
- `unrealized_pnl` = `(last_price - avg_open_price) * volume` (long)
- `unrealized_pnl` = `(avg_open_price - last_price) * volume` (short)

**Example (Long Position)**:
```json
{
  "symbol": "btcusdt",
  "direction": "Long",
  "volume": 0.5,                  // Own 0.5 BTC
  "frozen_total": 0,              // None frozen
  "avg_open_price": 48000.00,     // Bought at avg $48k
  "last_price": 50000.00,         // Current market price
  "realized_pnl": 0.00,           // No closed trades
  "unrealized_pnl": 1000.00       // (50000 - 48000) * 0.5 = +$1000
}
```

## Position Tracking Logic

### On Order Insert

**Trigger**: `on_order_input(event, OrderInput)`

**Actions**:
1. **Freeze Assets** (for buy orders):
   - Spot: Freeze `price * volume` in quote currency (e.g., USDT)
   - Futures: Calculate required margin, freeze margin

2. **Freeze Position** (for sell orders):
   - Freeze `volume` in base currency position
   - Update `frozen_total`

**Example**:
```cpp
// Buy 0.1 BTC at 50000 USDT (spot)
// Before: USDT avail = 10000, frozen = 0
// After:  USDT avail = 5000, frozen = 5000  (freeze 0.1 * 50000 = 5000 USDT)
```

### On Order Fill (Partial or Full)

**Trigger**: `on_trade(event, MyTrade)`

**Actions**:
1. **Unfreeze Assets/Position**:
   - Reduce `frozen` by filled amount
   - Increase `avail` if fill was worse than limit price (refund)

2. **Update Position**:
   - Add to `volume` (buy) or subtract from `volume` (sell)
   - Update `avg_open_price` using weighted average:
     ```cpp
     new_avg = (old_avg * old_volume + fill_price * fill_volume) / (old_volume + fill_volume)
     ```

3. **Calculate Realized PnL** (if closing position):
   ```cpp
   // Closing a long position (sell)
   realized_pnl += (sell_price - avg_open_price) * sell_volume
   ```

4. **Deduct Fees**:
   - Reduce `avail` by `trade.fee` in `trade.fee_currency`

**Example (Buy Fill)**:
```cpp
// Initial: BTC position = 0.4, avg_price = 48000
// Fill: Buy 0.1 BTC at 49000
// Result:
//   volume = 0.5
//   avg_open_price = (48000 * 0.4 + 49000 * 0.1) / 0.5 = 48200
```

### On Depth Update

**Trigger**: `on_depth(event, Depth)`

**Actions**:
1. **Update Mark Price**:
   - Set `last_price = depth.ask_price[0]` (for long positions)
   - Set `last_price = depth.bid_price[0]` (for short positions)

2. **Recalculate Unrealized PnL**:
   ```cpp
   // Long position
   unrealized_pnl = (last_price - avg_open_price) * volume

   // Short position
   unrealized_pnl = (avg_open_price - last_price) * volume
   ```

**Frequency**: Every depth update (~10 per second), so unrealized PnL is near real-time.

### On Order Cancel

**Trigger**: `on_order(event, Order)` with `status = Cancelled`

**Actions**:
1. **Unfreeze Remaining**:
   - Unfreeze `order.volume_left` (quantity that didn't fill)
   - Restore to `avail` (cash) or unfrozen position (holdings)

**Example**:
```cpp
// Order: Buy 0.1 BTC, filled 0.04, cancelled
// Unfreeze: 50000 * 0.06 = 3000 USDT (return to avail)
```

## Strategy Access

### Query Positions

**Python API**:
```python
def pre_start(self, context):
    # Get book context
    book = context.get_account_book("binance", "my_account")

    # Query positions (future API - not in current context.h)
    positions = context.query_positions("my_account", "btcusdt", InstrumentType.Spot)
    for pos in positions:
        print(f"{pos.symbol}: {pos.volume} @ {pos.avg_open_price}")
        print(f"Unrealized PnL: {pos.unrealized_pnl}")
```

### Position Callbacks

**Python API**:
```python
def on_position(self, context, position):
    """Called when position changes (after fills)"""
    self.log.info(f"Position update: {position.symbol}")
    self.log.info(f"  Volume: {position.volume}")
    self.log.info(f"  Avg Price: {position.avg_open_price}")
    self.log.info(f"  Unrealized PnL: {position.unrealized_pnl}")
    self.log.info(f"  Realized PnL: {position.realized_pnl}")
```

### Asset Callbacks

**Python API**:
```python
def on_asset(self, context, asset):
    """Called when cash balance changes"""
    self.log.info(f"Asset update: {asset.coin}")
    self.log.info(f"  Available: {asset.avail}")
    self.log.info(f"  Frozen: {asset.frozen}")
    self.log.info(f"  Margin: {asset.margin}")
```

## Ledger Categories

**Enum**: `LedgerCategory` (defined in [common.h](../../core/cpp/wingchun/include/kungfu/wingchun/common.h))

| Category | Description | Asset Behavior | Position Behavior |
|----------|-------------|----------------|-------------------|
| **Account** | Spot trading | Separate balances per coin | Net position per symbol |
| **Margin** | Margin trading | Shared collateral | Long/short positions |
| **Futures** | Futures/swaps | Collateral in margin | Mark-to-market PnL |

**Example (Spot - Account)**:
- Asset: USDT (quote currency)
- Asset: BTC (base currency)
- Position: Long 0.5 BTC (no direction, just holdings)

**Example (Futures)**:
- Asset: USDT (margin collateral)
- Position: Long 1.0 BTCUSDT (directional, can be short)
- Unrealized PnL added to margin

## Performance Characteristics

### Update Frequency

| Event Type | Frequency | Overhead |
|------------|-----------|----------|
| Depth update | ~10/sec per symbol | ~5 μs (mark price update) |
| Order fill | Per trade | ~50 μs (position update) |
| Order insert | Per order | ~20 μs (freeze logic) |
| Position query | On-demand | ~1 μs (in-memory lookup) |

### Memory Usage

**Per Account**:
- Book object: ~1 KB
- Position map: ~200 bytes per symbol
- Asset map: ~150 bytes per currency

**Example**: 10 accounts × 50 symbols each = ~100 KB total

### Concurrency

- **Thread-Safety**: Ledger runs in its own event loop (single-threaded)
- **Isolation**: Each strategy has its own BookContext instance
- **Coordination**: Journal guarantees event ordering across components

## Error Handling

### Insufficient Balance

```python
# Strategy attempts to buy with insufficient USDT
order_id = context.insert_order(
    symbol="btcusdt",
    side=Side.Buy,
    volume=100,  # Requires 5M USDT
    price=50000
)

# Ledger checks avail < required_amount
# → Order rejected with error_code = "INSUFFICIENT_BALANCE"
# → on_order(order) callback with status = Error
```

### Position Not Found

```python
# Strategy tries to sell without holding position
order_id = context.insert_order(
    symbol="btcusdt",
    side=Side.Sell,
    volume=1.0  # Don't have 1.0 BTC
)

# Ledger checks position.volume < required_volume
# → Order rejected
```

## Related Documentation

### Contracts
- [Order Object Contract](../30_contracts/order_object_contract.md) - Order state tracking
- [Strategy Context API](../30_contracts/strategy_context_api.md) - Account/book access methods

### Modules
- [Strategy Framework](strategy_framework.md) - How strategies interact with ledger
- [Gateway Architecture](gateway_architecture.md) - Order routing through ledger

### Future Documentation
- **Asset Management Flow** (20_interactions/) - Cash balance lifecycle
- **Position Lifecycle Flow** (20_interactions/) - Position open → close → PnL
- **Risk Management** (30_contracts/) - Position limits, margin checks

## Implementation Notes

### Freezing Logic

**Critical**: Freezing must happen BEFORE order is sent to exchange:
```cpp
// Correct order:
1. Check available balance
2. Freeze required amount
3. Send order to exchange
4. If exchange rejects, unfreeze

// Incorrect (race condition):
1. Send order to exchange
2. Freeze after acknowledgment  ← TOO LATE, could double-spend
```

### PnL Calculation Precision

**Floating-Point Issues**:
- Use `double` for prices/volumes (standard in finance)
- Accumulate PnL in high precision (don't round intermediate steps)
- Display PnL with 2-8 decimal places depending on currency

**Example**:
```cpp
// Correct
double pnl = (sell_price - buy_price) * volume;

// Incorrect (loses precision)
double pnl = round((sell_price - buy_price) * volume);
```

### Exchange Synchronization

**On Strategy Start**:
1. Ledger queries positions from exchange (`query_positions()`)
2. Exchange responds with `Position` events
3. Book processes `on_position()` to sync internal state
4. Mark book as `ready_ = true`
5. Strategy `pre_start()` executes after books ready

**Periodic Sync** (optional):
- Query positions every 5-10 minutes
- Detect drift between ledger and exchange
- Log warnings if discrepancies detected

## Version History

- **2025-11-17**: Initial ledger system documentation
- **2025-03-03**: Modified from original kungfu by kx@godzilla.dev
