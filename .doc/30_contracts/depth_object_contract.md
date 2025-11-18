---
title: Depth Object Contract
updated_at: 2025-11-17
owner: core-dev
lang: en
tags: [contract, market-data, depth, orderbook, message]
code_refs:
  - core/cpp/wingchun/include/kungfu/wingchun/msg.h:242-302
  - core/extensions/binance/src/marketdata_binance.cpp:145-220
purpose: "Defines the market depth (order book) data structure for Level 2 market data"
---

# Depth Object Contract

## Purpose

The `Depth` struct represents Level 2 market depth data (order book snapshots) with up to 10 price levels on each side (bid/ask). This is the primary market data structure consumed by trading strategies for order book analysis.

## Structure Definition

**Source:** [core/cpp/wingchun/include/kungfu/wingchun/msg.h:242-302](../../core/cpp/wingchun/include/kungfu/wingchun/msg.h)

### Fields

| Field | Type | Size | Constraints | Description |
|-------|------|------|-------------|-------------|
| `source_id` | `char[]` | 64 bytes | null-terminated | Market data source identifier (e.g., "binance") |
| `symbol` | `char[]` | 64 bytes | lowercase | Trading symbol (e.g., "btcusdt") |
| `exchange_id` | `char[]` | 64 bytes | uppercase | Exchange identifier (e.g., "BINANCE") |
| `data_time` | `int64_t` | 8 bytes | nanoseconds | Exchange timestamp of data generation |
| `instrument_type` | `InstrumentType` | 1 byte | enum | Spot, Futures, etc. |
| `bid_price[10]` | `double[]` | 80 bytes | descending | Best bid prices (bid[0] = highest bid) |
| `ask_price[10]` | `double[]` | 80 bytes | ascending | Best ask prices (ask[0] = lowest ask) |
| `bid_volume[10]` | `double[]` | 80 bytes | >0 | Quantity available at each bid level |
| `ask_volume[10]` | `double[]` | 80 bytes | >0 | Quantity available at each ask level |

**Total Size:** ~538 bytes (packed struct)

### Array Indexing

**Critical:** Arrays are indexed from 0 (best price) to 9 (worst price):

```
bid_price[0]   <-- Highest bid (best price to sell)
bid_price[1]   <-- Second best bid
...
bid_price[9]   <-- 10th best bid

ask_price[0]   <-- Lowest ask (best price to buy)
ask_price[1]   <-- Second best ask
...
ask_price[9]   <-- 10th best ask
```

## Invariants

### 1. Price Ordering
- `bid_price[0]` >= `bid_price[1]` >= ... >= `bid_price[9]` (descending)
- `ask_price[0]` <= `ask_price[1]` <= ... <= `ask_price[9]` (ascending)
- `bid_price[0]` < `ask_price[0]` (bid < ask, no crossed market)

### 2. Volume Constraints
- `bid_volume[i]` > 0 when `bid_price[i]` > 0
- `ask_volume[i]` > 0 when `ask_price[i]` > 0
- Zero prices indicate no more levels (sparse array)

### 3. Timestamp
- `data_time` is exchange-generated timestamp in nanoseconds
- NOT system receive time (use journal timestamp for latency measurement)
- Monotonic within a symbol stream (usually, but not guaranteed)

### 4. Symbol Format
- `symbol` is lowercase normalized (e.g., "btcusdt", not "BTCUSDT")
- `exchange_id` is uppercase (e.g., "BINANCE")
- `source_id` matches market data gateway name

### 5. Sparse Levels
If exchange provides <10 levels, remaining entries are zeros:
```cpp
// Example: 5 bid levels, 8 ask levels
bid_price = {50000, 49999, 49998, 49997, 49996, 0, 0, 0, 0, 0}
bid_volume = {1.5, 2.3, 0.8, 1.2, 0.5, 0, 0, 0, 0, 0}

ask_price = {50001, 50002, 50003, 50004, 50005, 50006, 50007, 50008, 0, 0}
ask_volume = {2.1, 1.8, 3.2, 0.9, 1.1, 0.6, 0.4, 0.3, 0, 0}
```

## Usage Examples

### Python (Strategy - Common Pattern)

```python
# strategies/demo_spot.py:65
def on_depth(self, context, depth):
    """
    Callback when order book depth update received.
    """
    # Best bid and ask (Level 1)
    best_bid = depth.bid_price[0]
    best_ask = depth.ask_price[0]
    spread = best_ask - best_bid

    self.log.info(f"{depth.symbol} | Bid: {best_bid} x {depth.bid_volume[0]}")
    self.log.info(f"{depth.symbol} | Ask: {best_ask} x {depth.ask_volume[0]}")
    self.log.info(f"Spread: {spread} ({spread / best_bid * 100:.2f}%)")

    # Check if enough liquidity at best bid
    if depth.bid_volume[0] >= 10.0:
        # Place sell order into best bid
        context.insert_order(
            symbol=depth.symbol,
            exchange_id=depth.exchange_id,
            source_id=depth.source_id,
            account_id=self.account,
            price=best_bid,  # Sell at bid (immediate execution)
            volume=5.0,
            side=Side.Sell,
            offset=Offset.Close,
            price_type=PriceType.Limit
        )
```

### Python (Advanced - Order Book Analysis)

```python
def on_depth(self, context, depth):
    """
    Calculate cumulative volume and find support/resistance.
    """
    # Calculate bid-side liquidity (support levels)
    bid_liquidity = sum(depth.bid_volume[i] for i in range(10) if depth.bid_price[i] > 0)

    # Calculate ask-side liquidity (resistance levels)
    ask_liquidity = sum(depth.ask_volume[i] for i in range(10) if depth.ask_price[i] > 0)

    # Find volume-weighted mid price (VWMP)
    if bid_liquidity > 0 and ask_liquidity > 0:
        vwmp = (depth.bid_price[0] * ask_liquidity + depth.ask_price[0] * bid_liquidity) / (bid_liquidity + ask_liquidity)
        self.log.info(f"VWMP: {vwmp:.2f}, Bid Liq: {bid_liquidity:.2f}, Ask Liq: {ask_liquidity:.2f}")

    # Detect imbalance (whale walls)
    if depth.bid_volume[0] > 3 * depth.ask_volume[0]:
        self.log.warning("Large bid wall detected - potential support")
    elif depth.ask_volume[0] > 3 * depth.bid_volume[0]:
        self.log.warning("Large ask wall detected - potential resistance")
```

### C++ (Market Data Gateway - Binance Example)

```cpp
// core/extensions/binance/src/marketdata_binance.cpp:180
void MarketDataBinance::on_depth_message(const nlohmann::json& data)
{
    msg::data::Depth depth = {};

    // Set metadata
    depth.set_source_id("binance");
    depth.set_symbol(symbol);  // normalized lowercase
    depth.set_exchange_id("BINANCE");
    depth.data_time = parse_timestamp(data["T"].get<int64_t>());
    depth.instrument_type = InstrumentType::Spot;

    // Parse bids (highest to lowest)
    auto bids = data["bids"].get<std::vector<std::vector<std::string>>>();
    for (size_t i = 0; i < std::min(bids.size(), (size_t)10); ++i) {
        depth.bid_price[i] = std::stod(bids[i][0]);
        depth.bid_volume[i] = std::stod(bids[i][1]);
    }

    // Parse asks (lowest to highest)
    auto asks = data["asks"].get<std::vector<std::vector<std::string>>>();
    for (size_t i = 0; i < std::min(asks.size(), (size_t)10); ++i) {
        depth.ask_price[i] = std::stod(asks[i][0]);
        depth.ask_volume[i] = std::stod(asks[i][1]);
    }

    // Write to journal (broadcast to strategies)
    writer_->write_data(msg::type::Depth, depth, sizeof(Depth));
}
```

## Data Source Specifics

### Binance Spot
- **WebSocket:** `wss://stream.binance.com/ws/<symbol>@depth10@100ms`
- **Update Frequency:** 100ms snapshots
- **Levels:** Always 10 levels (may have zeros if <10 available)
- **Timestamp:** Exchange-generated (`T` field in JSON)

### Binance Futures
- **WebSocket:** `wss://fstream.binance.com/ws/<symbol>@depth10@100ms`
- **Update Frequency:** 100ms snapshots
- **Levels:** Always 10 levels
- **Note:** Futures may have wider spreads than spot

### Generic Exchange Mapping
When implementing a new exchange connector:
1. Sort bids descending, asks ascending
2. Take top 10 levels (or fewer if unavailable)
3. Normalize symbol to lowercase
4. Use exchange timestamp, not system time
5. Zero-fill unused levels

## Performance Considerations

### Memory Layout
Struct is packed to 538 bytes for efficient cache access:
```cpp
#ifndef _WIN32
} __attribute__((packed));
#else
};  // Windows allows padding
#endif
```

### Update Frequency
- Binance: 100ms snapshots = 10 updates/second
- High-frequency strategies receive ~600 callbacks/minute per symbol
- Use `context.set_object()` for expensive calculations to avoid recomputation

### Latency Chain
```
Exchange → WebSocket → Parser → Journal Write → Strategy Callback
  <1ms      <1ms        <100μs      ~500ns         function call
```

Total latency (exchange to strategy): typically **<5ms** for local deployment.

## Related Contracts

- [Ticker Object Contract](ticker_object_contract.md) - Best bid/ask only (faster, less detailed)
- [Trade Object Contract](trade_object_contract.md) - Individual trades (execution data)
- [Strategy Context API](strategy_context_api.md) - `on_depth()` callback specification

## Related Documentation

### Modules
- [Binance Extension](../10_modules/binance_extension.md) - Depth data parsing implementation
- [Strategy Framework](../10_modules/strategy_framework.md) - How to use depth in strategies

### Flows
- [Market Data Flow](../20_interactions/market_data_flow.md) - WebSocket → Strategy propagation
- [Event Flow](../20_interactions/event_flow.md) - Journal-based event distribution

## JSON Serialization

The system provides automatic JSON conversion:

```cpp
inline void to_json(nlohmann::json &j, const Depth &depth)
{
    j["data_time"] = depth.data_time;
    j["instrument_type"] = depth.instrument_type;
    j["source_id"] = depth.get_source_id();
    j["symbol"] = depth.get_symbol();
    j["exchange_id"] = depth.get_exchange_id();

    j["bid_price"] = depth.get_bid_price();    // Returns std::vector<double>
    j["ask_price"] = depth.get_ask_price();
    j["bid_volume"] = depth.get_bid_volume();
    j["ask_volume"] = depth.get_ask_volume();
}
```

Python strategies receive this as native attributes:
```python
depth.bid_price   # List[float] of length 10
depth.ask_price   # List[float] of length 10
```

## Common Pitfalls

### 1. Index Confusion
```python
# ❌ WRONG - bid_price[9] is NOT the best bid
if depth.bid_price[9] > threshold:
    pass

# ✅ CORRECT - bid_price[0] is the best bid
if depth.bid_price[0] > threshold:
    pass
```

### 2. Assuming 10 Levels
```python
# ❌ WRONG - may divide by zero if <10 levels
avg_bid = sum(depth.bid_price) / 10

# ✅ CORRECT - filter out zeros
valid_bids = [p for p in depth.bid_price if p > 0]
avg_bid = sum(valid_bids) / len(valid_bids) if valid_bids else 0
```

### 3. Timestamp Misuse
```python
# ❌ WRONG - depth.data_time is exchange time, not local time
if depth.data_time > time.time_ns():
    self.log.error("Clock skew!")

# ✅ CORRECT - use for ordering, not wall-clock comparison
if depth.data_time > last_depth_time:
    process_depth(depth)
    last_depth_time = depth.data_time
```

### 4. Bid/Ask Confusion
```python
# ❌ WRONG - buy at ask, sell at bid (paying spread)
buy_price = depth.ask_price[0]   # You PAY the ask
sell_price = depth.bid_price[0]  # You RECEIVE the bid

# ✅ CORRECT - this is how markets work
# To buy: you lift the ask (take from sellers)
# To sell: you hit the bid (give to buyers)
```

## Version History

- **2025-11-17:** Initial contract documentation based on code analysis
- **2025-03-03:** Modified from original kungfu (Keren Dong) by kx@godzilla.dev
