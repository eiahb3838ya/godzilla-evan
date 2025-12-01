---
title: Binance Extension - Cryptocurrency Exchange Connector
updated_at: 2025-11-19
owner: core-dev
lang: en
tokens_estimate: 10200
layer: 10_modules
tags: [binance, exchange, gateway, websocket, rest-api, market-data, order-execution, spot, futures]
purpose: "Binance-specific MD/TD gateway implementation with dual-market support and configurable market toggle"
code_refs:
  - core/extensions/binance/include/marketdata_binance.h
  - core/extensions/binance/include/trader_binance.h
  - core/extensions/binance/include/common.h
  - core/extensions/binance/include/type_convert_binance.h
  - core/extensions/binance/src/marketdata_binance.cpp
  - core/extensions/binance/src/trader_binance.cpp
---

# Binance Extension - Cryptocurrency Exchange Connector

## Overview

The Binance extension provides a complete cryptocurrency exchange integration for the kungfu trading framework, implementing both Market Data (MD) and Trader (TD) gateways for Binance's Spot and Futures markets.

**Architecture**: Extends [wingchun](wingchun.md) broker interfaces, runs on [yijinjing](yijinjing.md) event system

**Key Capabilities**:
- Dual-market support (Spot + USDT-margined Futures)
- WebSocket-based real-time market data streaming
- REST API order execution and account queries
- Configurable market toggle (enable/disable per market)
- Type conversion between Binance API and kungfu data structures

**Implementation**: 20 source files (~3,000 lines C++), binapi client library

## System Context

```
┌─────────────────────────────────────────────────────────────┐
│ Strategy (Python/C++)                                       │
│   context.subscribe(), context.insert_order()              │
└───────────────────┬─────────────────────────────────────────┘
                    │ wingchun::Context API
┌───────────────────▼─────────────────────────────────────────┐
│ Wingchun Gateway Layer                                      │
│   broker::MarketData / broker::Trader interfaces            │
└───────┬─────────────────────────────────────────────────────┘
        │ Virtual function calls
┌───────▼─────────────────────────────────────────────────────┐
│ Binance Extension (this module)                             │
│ ┌─────────────────┐  ┌─────────────────┐                   │
│ │ MarketDataBinance│  │ TraderBinance   │                   │
│ │ - WebSocket (WS) │  │ - REST API      │                   │
│ │ - REST API       │  │ - WebSocket (WS)│                   │
│ │ - Depth/Ticker   │  │ - Order/Position│                   │
│ └────────┬────────┘  └────────┬────────┘                   │
│          │                     │                             │
│          │ binapi library      │                             │
│          ├─────────────────────┤                             │
│          │ TypeConvert (utils) │                             │
│          │ Configuration       │                             │
│          └─────────────────────┘                             │
└──────────────────┬──────────────────────────────────────────┘
                   │ HTTPS/WSS (Boost.Asio)
┌──────────────────▼──────────────────────────────────────────┐
│ Binance Testnet/Mainnet                                     │
│ - testnet.binance.vision (Spot)                             │
│ - testnet.binancefuture.com (Futures)                       │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. MarketDataBinance (MD Gateway)

**Purpose**: Subscribe to and distribute real-time market data from Binance

**File**: `core/extensions/binance/src/marketdata_binance.cpp` (600+ lines)

**Class Hierarchy**:
```cpp
kungfu::wingchun::broker::MarketData (wingchun base class)
    └── kungfu::wingchun::binance::MarketDataBinance
```

**Key Members**:
```cpp
class MarketDataBinance : public broker::MarketData {
private:
    Configuration config_;                              // Connection settings
    boost::asio::io_context ioctx_;                     // Async I/O context
    std::shared_ptr<binapi::ws::websockets> ws_ptr_;    // Spot WebSocket
    std::shared_ptr<binapi::ws::websockets> fws_ptr_;   // Futures WebSocket
    std::shared_ptr<binapi::ws::websockets> dws_ptr_;   // Coin-margined Futures WS
    std::shared_ptr<binapi::rest::api> rest_ptr_;       // Spot REST client
    std::shared_ptr<binapi::rest::api> frest_ptr_;      // Futures REST client
    std::shared_ptr<std::thread> task_thread_;          // I/O thread
    std::map<std::string, OrderBook> depths_cache_;     // Order book snapshots
    std::map<uint32_t, ChannelInfo> channel_cache_;     // Subscription tracking
};
```

**Subscription Flow**:
```
Strategy calls context.subscribe(instruments)
    ↓
MarketDataBinance::subscribe()
    ↓
Generate symbol_id (hash of symbol + sub_type + inst_type)
    ↓
Check channel_cache_ for duplicates
    ↓
ws_ptr_->part_depth() / fws_ptr_->part_depth()  (WebSocket subscription)
    ↓
Lambda callback receives binapi::ws::part_depths_t
    ↓
Convert to msg::data::Depth, write to yijinjing journal
    ↓
Strategy receives on_depth() event
```

**Supported Data Types**:
- **Depth** (`subscribe()`): Order book L2 updates (20 levels)
- **Trade** (`subscribe_trade()`): Public trade executions stream
  - **Spot**: Raw trade stream (每筆成交單獨推送)
  - **Futures**: Aggregated trade stream (aggTrade，短時間內同價格同方向成交聚合)
- **Ticker** (`subscribe_ticker()`): 24hr statistics
- **Index Price** (`subscribe_index_price()`): Futures index price

**Example** (`marketdata_binance.cpp:122-180`):
```cpp
bool MarketDataBinance::subscribe(const std::vector<Instrument>& instruments) {
    for (const auto& inst : instruments) {
        std::string symbol = to_binance_symbol(inst.symbol);  // "ltc_usdt" → "LTCUSDT"

        auto cb = [this, instrument_type=inst.instrument_type, orig_symbol](
            const char* fl, int ec, std::string errmsg, binapi::ws::part_depths_t msg) {
            if (ec) {
                SPDLOG_ERROR("fail to get depth: ec({}), errmsg({})", ec, errmsg);
                return false;
            }

            // Convert binapi depth to kungfu Depth message
            msg::data::Depth& depth = this->get_writer(0)->open_data<msg::data::Depth>(0, msg::type::Depth);
            strcpy(depth.symbol, orig_symbol.c_str());
            depth.instrument_type = instrument_type;

            // Copy bid/ask levels (up to 20 levels)
            for (size_t i = 0; i < msg.bids.size() && i < 20; ++i) {
                depth.BidPrice[i] = std::stod(msg.bids[i].price);
                depth.BidVolume[i] = std::stod(msg.bids[i].amount);
            }
            // ... asks similar ...

            this->get_writer(0)->close_data();  // Write to journal
            return true;
        };

        // Subscribe via WebSocket (Spot vs Futures routing)
        if (inst.instrument_type == InstrumentType::Spot) {
            auto handle = ws_ptr_->part_depth(cb, symbol.c_str(), "20", binapi::ws::e_levels::_20);
            channel_cache_[symbol_id] = {symbol, "depth", inst.instrument_type, handle, true};
        } else if (inst.instrument_type == InstrumentType::FFuture) {
            auto handle = fws_ptr_->part_depth(cb, symbol.c_str(), "20", binapi::ws::e_levels::_20);
            channel_cache_[symbol_id] = {symbol, "depth", inst.instrument_type, handle, true};
        }
    }
    return true;
}
```

### Trade Stream Design: Spot vs Futures

**Implementation** (`marketdata_binance.cpp:186-264`):

The Binance extension uses **different trade stream types** for Spot and Futures markets:

| Market Type | Stream Type | Callback Type | Frequency | Reason |
|-------------|-------------|---------------|-----------|--------|
| **Spot** | Raw `trade` | `binapi::ws::trade_t` | ~100-1000/sec | 現貨成交頻率適中，原始數據更有分析價值 |
| **USDT Futures** | Aggregated `aggTrade` | `binapi::ws::agg_trade_t` | ~10-50/sec | 期貨高頻成交，聚合後減少 90% 數據量 |
| **Coin Futures** | Aggregated `aggTrade` | `binapi::ws::agg_trade_t` | ~10-50/sec | 同上 |

**Code Evidence**:
```cpp
// marketdata_binance.cpp:231-264
if (inst.instrument_type == InstrumentType::Spot) {
    // Spot: Raw trade stream
    h = ws_ptr_->trade(symbol.c_str(),
        [this, orig_symbol](const char* fl, int ec, std::string errmsg, binapi::ws::trade_t msg) {
            // msg contains individual trade
            trade.trade_id = msg.t;        // Unique trade ID
            trade.ask_id = msg.a;          // Maker order ID
            trade.bid_id = msg.b;          // Taker order ID
            // ... emit Trade event
        });
} else if (inst.instrument_type == InstrumentType::FFuture) {
    // Futures: Aggregated trade stream
    h = fws_ptr_->agg_trade(symbol.c_str(), future_cb);
    // future_cb receives binapi::ws::agg_trade_t
    // Multiple trades at same price/direction aggregated into one
    trade.trade_id = msg.a;        // Aggregate trade ID (not individual)
    trade.ask_id = 0;              // Not available in aggTrade
    trade.bid_id = 0;
}
```

**aggTrade Aggregation Rules** (Binance API):
- 在 100ms 時間窗口內
- 相同價格
- 相同方向 (買/賣)
- 相同 taker
→ 聚合為單筆 aggTrade，累加 volume

**Performance Impact**:
- **BTC/USDT Futures**: 原始 trade ~2000/sec → aggTrade ~50/sec (減少 97.5%)
- **ETH/USDT Spot**: 原始 trade ~500/sec (保留所有)

**Strategy Implications**:
- ✅ **價格與成交量數據完整**：aggTrade 包含所有關鍵信息
- ✅ **適合大多數量化策略**：VWAP、成交量分析等不需要逐筆數據
- ⚠️ **訂單簿微觀結構分析受限**：無法追蹤單筆成交的 maker/taker order ID
- ⚠️ **Market microstructure 研究不適用**：需要原始 trade 數據才能分析訂單流

**Switching to Raw Trade (Futures)**:
如需在期貨市場使用原始 trade stream，需修改 `marketdata_binance.cpp:262`:
```cpp
// 當前: aggTrade
h = fws_ptr_->agg_trade(symbol.c_str(), future_cb);

// 修改為: raw trade (需調整 callback 簽名)
h = fws_ptr_->trade(symbol.c_str(), future_trade_cb);
```
⚠️ **警告**: 高頻市場可能產生 10-100x 數據量，需評估系統 I/O 能力

**Startup Sequence** (`on_start()`, line 65-120):
1. Connect to REST API (Spot + Futures)
2. Fetch exchange info (`exchange_info()`, `future_exchange_info()`)
3. Populate MarketInfoDB (shared memory DB for instrument metadata)
4. Start I/O thread (`ioctx_.run()`)
5. Register 5-second health check timer (`_check_status()`)

### 2. TraderBinance (TD Gateway)

**Purpose**: Execute orders and query account state on Binance

**File**: `core/extensions/binance/src/trader_binance.cpp` (800+ lines)

**Class Hierarchy**:
```cpp
kungfu::wingchun::broker::Trader (wingchun base class)
    └── kungfu::wingchun::binance::TraderBinance
```

**Key Members**:
```cpp
class TraderBinance : public broker::Trader {
private:
    Configuration config_;                              // Connection + auth settings
    boost::asio::io_context ioctx_;                     // Async I/O context
    std::shared_ptr<binapi::rest::api> rest_ptr_;       // Spot REST client
    std::shared_ptr<binapi::rest::api> frest_ptr_;      // Futures REST client
    std::shared_ptr<binapi::ws::websockets> ws_ptr_;    // Spot WebSocket (userdata)
    std::shared_ptr<binapi::ws::websockets> fws_ptr_;   // Futures WebSocket (userdata)
    std::map<std::size_t, order_map_record> order_data_;// Order tracking
    std::list<std::string> listenKeys;                  // WebSocket listen keys
    std::mutex order_mtx_;                              // Thread-safe order access
};
```

**Order Execution Flow**:
```
Strategy calls context.insert_order(order_input)
    ↓
TraderBinance::insert_order(event)
    ↓
Extract OrderInput from event
    ↓
Create Order record, store in order_data_ (with mutex lock)
    ↓
Route to Spot/Futures REST API based on instrument_type
    ↓
rest_ptr_->new_order() / frest_ptr_->new_order()
    ↓
Lambda callback receives binapi::rest::new_order_resp_full_t
    ↓
Convert to msg::data::Order, write to journal (on_order event)
    ↓
Update order_data_ with exchange order_id
    ↓
If filled, also emit on_trade event
```

**Example Order Insertion** (`trader_binance.cpp:123-200`):
```cpp
bool TraderBinance::insert_order(const yijinjing::event_ptr& event) {
    const OrderInput& input = event->data<OrderInput>();
    msg::data::Order order{};
    order_from_input(input, order);  // Convert input to order

    {
        std::lock_guard<std::mutex> lock(order_mtx_);
        order_data_.emplace(input.order_id, order_map_record(input.order_id, "", source, now(), order));
    }

    if (input.instrument_type == InstrumentType::Spot) {
        auto order_ptr = std::make_shared<binapi::rest::api>(...);  // Spot client
        order_ptr->new_order(
            to_binance_symbol(input.symbol),      // "ltc_usdt" → "LTCUSDT"
            to_binance(input.side),               // Buy/Sell enum conversion
            to_binance(input.order_type),         // Limit/Market enum conversion
            binapi::e_time::GTC,                  // Time in force
            binapi::e_trade_resp_type::FULL,      // Full response
            to_string(input.volume),              // Quantity
            to_string(input.limit_price),         // Limit price (if Limit order)
            [this, order_id=input.order_id](const char* fl, int ec, std::string errmsg, auto resp) {
                if (ec) {
                    SPDLOG_ERROR("Order failed: ec({}), errmsg({})", ec, errmsg);
                    // Emit error order update
                    return;
                }

                // Update order record with exchange order_id
                {
                    std::lock_guard<std::mutex> lock(order_mtx_);
                    order_data_[order_id].exchange_order_id = std::to_string(resp.orderId);
                }

                // Emit on_order event
                msg::data::Order& order = get_writer(0)->open_data<msg::data::Order>(0, msg::type::Order);
                order.order_id = order_id;
                order.exchange_order_id = resp.orderId;
                order.status = from_binance(resp.status);  // NEW/FILLED/etc.
                // ... fill other fields ...
                get_writer(0)->close_data();

                // If filled, emit on_trade event
                if (resp.status == binapi::e_status::filled) {
                    // ... emit Trade message ...
                }
            }
        );
    } else if (input.instrument_type == InstrumentType::FFuture) {
        // Similar logic with frest_ptr_->new_order()
    }

    return true;
}
```

**Account Query Methods**:
- `req_position()`: Fetch current positions (Spot balances or Futures positions)
- `req_account()`: Fetch account equity and margin info
- `query_order()`: Query specific order status
- `cancel_order()`: Cancel pending order

**User Data Stream** (`_start_userdata()`, line 220-280):
- Creates listen key via REST API
- Subscribes to WebSocket user data stream
- Receives real-time order updates, fills, position changes
- Automatically emits events (no polling needed)

### 3. Configuration System

**File**: `core/extensions/binance/include/common.h`

**Configuration Struct**:
```cpp
struct Configuration {
    std::string user_id;
    std::string access_key;       // Binance API key
    std::string secret_key;       // Binance secret key

    // Market toggle flags (ADR-004) - backward compatible defaults
    bool enable_spot = true;
    bool enable_futures = true;

    // Connection endpoints (hardcoded for Testnet)
    std::string spot_rest_host;   // "testnet.binance.vision"
    int spot_rest_port;           // 443
    std::string spot_wss_host;    // "stream.testnet.binance.vision"
    int spot_wss_port;            // 443
    std::string ubase_rest_host;  // "testnet.binancefuture.com" (USDT-margined)
    int ubase_rest_port;          // 443
    std::string ubase_wss_host;   // "stream.binancefuture.com"
    int ubase_wss_port;           // 443
    std::string cbase_rest_host;  // "testnet.binancefuture.com" (Coin-margined)
    int cbase_rest_port;          // 443
    std::string cbase_wss_host;   // "dstream.binancefuture.com"
    int cbase_wss_port;           // 443
};
```

**JSON Parsing** (`common.h:43-71`):
```cpp
inline void from_json(const nlohmann::json &j, Configuration &c) {
    j.at("user_id").get_to(c.user_id);
    j.at("access_key").get_to(c.access_key);
    j.at("secret_key").get_to(c.secret_key);

    // Market toggle flags (backward compatible - defaults to true if missing)
    c.enable_spot = j.value("enable_spot", true);
    c.enable_futures = j.value("enable_futures", true);

    // Hardcoded Testnet URLs (not configurable via JSON)
    c.spot_rest_host = "testnet.binance.vision";
    c.spot_rest_port = 443;
    // ... rest hardcoded ...
}
```

**Market Toggle Feature (ADR-004)**:
- **Problem**: Futures-only API keys generate continuous Spot login errors
- **Solution**: Guard clause pattern - skip initialization if market disabled
- **Implementation**: 3 touch points in `trader_binance.cpp`:
  1. Constructor (lines 50-80): Conditional REST client creation
  2. `on_start()` (lines 105-116): Conditional `_start_userdata()` calls
  3. `_check_status()` (lines 342-350): Conditional reconnection attempts

**Example** (`trader_binance.cpp:50-64`):
```cpp
// Constructor - Guard Clause Pattern (ADR-004)
if (config_.enable_spot) {
    rest_ptr_ = std::make_shared<binapi::rest::api>(
        ioctx_, config_.spot_rest_host, std::to_string(config_.spot_rest_port),
        config_.access_key, config_.secret_key, 10000);
    ws_ptr_ = std::make_shared<binapi::ws::websockets>(
        ioctx_, config_.spot_wss_host, std::to_string(config_.spot_wss_port));
} else {
    SPDLOG_INFO("Spot market disabled by configuration");
}

if (config_.enable_futures) {
    frest_ptr_ = std::make_shared<binapi::rest::api>(
        ioctx_, config_.ubase_rest_host, std::to_string(config_.ubase_rest_port),
        config_.access_key, config_.secret_key, 10000);
    fws_ptr_ = std::make_shared<binapi::ws::websockets>(
        ioctx_, config_.ubase_wss_host, std::to_string(config_.ubase_wss_port));
} else {
    SPDLOG_INFO("Futures market disabled by configuration");
}
```

**Config Example**: `.doc/40_config/examples/binance_market_toggle_config.json`

### 4. Type Conversion Utilities

**File**: `core/extensions/binance/include/type_convert_binance.h`

**Purpose**: Bidirectional type conversion between Binance API enums and kungfu enums

**Key Functions**:

```cpp
// Order Type Conversion
OrderType from_binance(const binapi::e_type &binance_order_type);
binapi::e_type to_binance(const OrderType &order_type);
// Limit ↔ binapi::e_type::limit
// Market ↔ binapi::e_type::market

// Order Status Conversion
OrderStatus from_binance(const binapi::e_status &status);
// binapi::e_status::newx → OrderStatus::Submitted
// binapi::e_status::partially_filled → OrderStatus::PartialFilledActive
// binapi::e_status::filled → OrderStatus::Filled
// binapi::e_status::canceled → OrderStatus::Cancelled
// binapi::e_status::rejected → OrderStatus::Error

// Side Conversion
Side from_binance(const binapi::e_side &binance_side);
binapi::e_side to_binance(const Side &side);
// Buy ↔ binapi::e_side::buy
// Sell ↔ binapi::e_side::sell

// Direction (Futures only)
binapi::e_position_side to_binance(const Direction &direction);
// Long → binapi::e_position_side::p_long
// Short → binapi::e_position_side::p_short

// Symbol Conversion
const std::string to_binance_symbol(const std::string &symbol);
const std::string from_binance_symbol(const std::string &symbol);
// "ltc_usdt" ↔ "LTCUSDT"
// "btc_usdt" ↔ "BTCUSDT"
```

**Symbol Conversion Logic** (`type_convert_binance.h:111-142`):
```cpp
inline const std::string to_binance_symbol(const std::string &symbol) {
    // "ltc_usdt" → "LTCUSDT"
    std::vector<std::string> result;
    boost::split(result, symbol, boost::is_any_of("_"));
    std::string res;
    for (auto &coin: result) {
        std::transform(coin.begin(), coin.end(), coin.begin(), ::toupper);
        res.append(coin);
    }
    return res;
}

inline const std::string from_binance_symbol(const std::string &symbol) {
    // "LTCUSDT" → "ltc_usdt"
    std::vector<std::string> quote_coins{"BTC", "USDT", "BNB", "BUSD", "ETH"};
    std::string res = symbol;
    for (auto &quote: quote_coins) {
        if (symbol.ends_with(quote)) {
            res.insert(res.rfind(quote), "_");  // Insert underscore before quote currency
            break;
        }
    }
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
}
```

## Usage Examples

### Example 1: Futures-Only Configuration (Testnet)

**Scenario**: Use Futures Testnet API key, disable Spot to avoid `-2015` errors

**Configuration**:
```bash
# Interactive account creation (recommended)
python core/python/dev_run.py account -s binance add
# Prompts:
#   user_id: gz_user1
#   access_key: YOUR_FUTURES_TESTNET_KEY
#   secret_key: YOUR_FUTURES_SECRET
#   是否启用现货市场登录？(true/false): false
#   是否启用期货市场登录？(true/false): true
```

**Expected Logs** (`/app/runtime/td/binance/gz_user1/log/live/gz_user1.log`):
```
[info] Spot market disabled by configuration
[info] Connecting BINANCE TD for gz_user1 (Spot: disabled, Futures: enabled)
[info] Skipping Spot initialization (disabled or client unavailable)
[info] login success
```

**Result**: No Spot login errors, clean logs

### Example 2: Market Data Subscription (Python Strategy)

```python
from kungfu.wingchun.constants import *

class MyStrategy(Strategy):
    def on_start(self):
        # Subscribe to Futures depth data
        self.ctx.subscribe(
            source=SOURCE_BINANCE,
            symbols=["ltc_usdt", "btc_usdt"],
            exchange_id=EXCHANGE_BINANCE,
            instrument_type=InstrumentType.FFuture,
            data_type=MarketDataType.Depth
        )

    def on_depth(self, context, depth, location, dest):
        if depth.symbol == "ltc_usdt":
            self.ctx.log_info(
                f"LTC Depth - Best Bid: {depth.BidPrice[0]} @ {depth.BidVolume[0]}, "
                f"Best Ask: {depth.AskPrice[0]} @ {depth.AskVolume[0]}"
            )
```

**Underlying Execution**:
1. Strategy calls `context.subscribe()`
2. Wingchun routes to `MarketDataBinance::subscribe()`
3. Extension converts "ltc_usdt" → "LTCUSDT"
4. `fws_ptr_->part_depth()` subscribes to `ltcusdt@depth20@100ms` stream
5. WebSocket callback writes `msg::data::Depth` to journal
6. Strategy receives `on_depth()` callback via yijinjing event loop

### Example 3: Order Execution (Python Strategy)

```python
def on_depth(self, context, depth, location, dest):
    if depth.symbol == "ltc_usdt":
        # Insert limit buy order
        order_id = context.insert_limit_order(
            symbol="ltc_usdt",
            exchange_id=EXCHANGE_BINANCE,
            instrument_type=InstrumentType.FFuture,
            side=Side.Buy,
            price=depth.AskPrice[0] - 0.01,  # Below best ask
            volume=1.0
        )
        self.ctx.log_info(f"Inserted order {order_id}")

def on_order(self, context, order, location, dest):
    self.ctx.log_info(
        f"Order {order.order_id} status: {order.status}, "
        f"filled: {order.volume_traded}/{order.volume}"
    )

def on_trade(self, context, trade, location, dest):
    self.ctx.log_info(
        f"Trade {trade.trade_id} executed: {trade.volume} @ {trade.price}"
    )
```

**Underlying Execution**:
1. `context.insert_limit_order()` creates `msg::data::OrderInput`
2. Wingchun routes to `TraderBinance::insert_order()`
3. Extension converts types and calls `frest_ptr_->new_order()`
4. Binance REST API responds with order confirmation
5. Extension writes `msg::data::Order` to journal → `on_order()` callback
6. When filled, writes `msg::data::Trade` to journal → `on_trade()` callback

## Performance Characteristics

**Market Data Latency**:
- WebSocket subscription: ~50-200ms from exchange event to strategy callback
- Order book update frequency: 100ms (Binance depth stream default)
- Thread model: Single I/O thread per gateway instance (Boost.Asio async)

**Order Execution Latency**:
- REST API round-trip: ~100-500ms (network dependent)
- User data stream latency: ~50-200ms for order updates
- Thread safety: Mutex-protected order_data_ map

**Resource Usage**:
- Memory: ~50MB per gateway instance (WebSocket buffers + order book cache)
- CPU: Minimal (<5% single core) during normal trading
- Network: ~100KB/s per market per 10 subscribed symbols

**Scalability**:
- Max subscriptions per MD instance: ~100 symbols (Binance WebSocket limit)
- Max concurrent orders: Limited by Binance rate limits (50 orders/10s Spot, 300/10s Futures)
- Multiple accounts: Spawn separate TD gateway instances

## Hotspots and Pitfalls

### Hotspot 1: Order Book Cache Management

**Location**: `MarketDataBinance::depths_cache_` (`marketdata_binance.h:118`)

**Issue**: Order book snapshots stored in memory can grow unbounded if many symbols subscribed

**Mitigation**:
- Cache only subscribed symbols
- Clear cache on unsubscribe
- Consider snapshot expiry (not currently implemented)

**Example Failure**:
```cpp
// Subscribe to 500 symbols → depths_cache_ grows to ~500MB
for (auto& symbol : all_symbols) {
    subscribe({symbol});  // Each adds to cache
}
```

### Hotspot 2: Thread-Safe Order Access

**Location**: `TraderBinance::order_data_` + `order_mtx_` (`trader_binance.h:64-66`)

**Issue**: High-frequency order updates can cause mutex contention

**Current Design**:
```cpp
{
    std::lock_guard<std::mutex> lock(order_mtx_);
    order_data_[order_id] = new_record;  // Lock held during map update
}
```

**Mitigation**:
- Keep critical sections small
- Avoid I/O operations inside lock
- Consider lock-free data structures for extreme HFT

### Hotspot 3: WebSocket Reconnection Logic

**Location**: `_check_status()` health check timer (5-second interval)

**Issue**: Reconnection flag checked every 5 seconds, potential 5s gap if connection drops

**Current Implementation** (`trader_binance.cpp:342-350`):
```cpp
void TraderBinance::_check_status(kungfu::yijinjing::event_ptr) {
    if (config_.enable_spot && rest_ptr_ && ws_ptr_->fetch_reconnect_flag()) {
        _start_userdata(InstrumentType::Spot);  // Reconnect Spot
    }
    if (config_.enable_futures && frest_ptr_ && fws_ptr_->fetch_reconnect_flag()) {
        _start_userdata(InstrumentType::FFuture);  // Reconnect Futures
    }
}
```

**Mitigation**:
- Reduce timer interval for faster recovery (trade-off: CPU usage)
- Implement exponential backoff for reconnection attempts (not currently implemented)

### Pitfall 1: Symbol Case Sensitivity

**Problem**: kungfu uses lowercase with underscore (`ltc_usdt`), Binance uses uppercase no separator (`LTCUSDT`)

**Failure Example**:
```cpp
// ❌ WRONG: Using kungfu symbol directly with Binance API
rest_ptr_->new_order("ltc_usdt", ...);  // Binance rejects: invalid symbol
```

**Correct Usage**:
```cpp
// ✅ RIGHT: Always convert with type_convert utilities
std::string binance_symbol = to_binance_symbol("ltc_usdt");  // "LTCUSDT"
rest_ptr_->new_order(binance_symbol, ...);
```

**Detection**:
- Binance API returns error `-1121`: "Invalid symbol"
- Check logs for symbol conversion before API calls

### Pitfall 2: Market Toggle Configuration Mismatch

**Problem**: Disabling a market but still trying to trade it causes null pointer dereference

**Failure Scenario**:
```json
// Config: Spot disabled
{"enable_spot": false, "enable_futures": true}
```

```cpp
// Strategy tries to insert Spot order
context.insert_order(
    symbol="btc_usdt",
    instrument_type=InstrumentType.Spot,  // ← Spot disabled!
    ...
);
```

**Result**:
```cpp
// trader_binance.cpp:136 - rest_ptr_ is nullptr
rest_ptr_->new_order(...);  // ← Segmentation fault
```

**Mitigation**:
- Always check pointer validity before use
- Add guard clause: `if (!rest_ptr_) { SPDLOG_ERROR("Spot disabled"); return false; }`
- Validate strategy config matches gateway config

**Detection**:
- Segfault in `insert_order()` / `req_position()` / `cancel_order()`
- Check logs for "market disabled by configuration" on startup

### Pitfall 3: Testnet vs Mainnet URL Hardcoding

**Problem**: URLs are hardcoded in `from_json()`, switching to mainnet requires code change

**Current Limitation** (`common.h:54-70`):
```cpp
// Hardcoded Testnet URLs (cannot override via JSON)
c.spot_rest_host = "testnet.binance.vision";
c.ubase_rest_host = "testnet.binancefuture.com";
```

**Failure**: Deploying to production with Testnet URLs → all orders rejected

**Mitigation**:
- **Development**: Use Testnet URLs (current default)
- **Production**: Manually edit `common.h` and recompile:
  ```cpp
  c.spot_rest_host = "api.binance.com";
  c.ubase_rest_host = "fapi.binance.com";
  ```
- **Future Enhancement**: Add URL fields to JSON config (requires ADR)

**Detection**:
- Orders succeed on Testnet but fail on Mainnet (or vice versa)
- API returns `-2015`: "Invalid API-key" (if using wrong environment key)

### Pitfall 4: API Rate Limits

**Problem**: Binance enforces strict rate limits, excessive requests cause temporary bans

**Limits** (Spot):
- 1200 requests/minute per IP
- 50 orders/10 seconds per account
- Weight-based limits for different endpoints

**Failure Example**:
```python
# ❌ WRONG: Rapid order insertion in loop
for i in range(100):
    context.insert_order(...)  # Exceeds 50 orders/10s
```

**Result**:
- HTTP 429 error: "Too Many Requests"
- 2-minute IP ban (Spot) or 5-minute ban (Futures)

**Mitigation**:
- Rate-limit order insertion in strategy logic
- Use batch endpoints where available (not exposed in current extension)
- Monitor `order_ptr_->get_used_weight()` (not currently implemented)

**Detection**:
- Logs show error `-1003`: "Too much request weight used"
- Orders fail with "Service unavailable"

### Pitfall 5: Incomplete Order Fill Events

**Problem**: Partially filled orders may not emit `on_trade()` for each fill

**Current Implementation** (`trader_binance.cpp:150-200`):
- Single `on_trade()` event emitted when order fully filled
- Partial fills only update `on_order()` with `PartialFilledActive` status
- Individual fill executions not tracked separately

**Failure Scenario**:
```python
# Order inserted for 10 LTC
order_id = context.insert_order(volume=10.0)

# Order fills in 3 chunks: 3 LTC, 5 LTC, 2 LTC
# Expected: 3 on_trade() events
# Actual: 1 on_order() with status=PartialFilledActive (qty=3)
#         1 on_order() with status=PartialFilledActive (qty=8)
#         1 on_order() with status=Filled (qty=10)
#         1 on_trade() with volume=10
```

**Mitigation**:
- Parse user data stream for `executionReport` events (requires enhancement)
- Track `order.volume_traded` increments in `on_order()` callback
- Do NOT rely on `on_trade()` count for fill analysis

**Detection**:
- Strategy logic assumes `on_trade()` called per fill → incorrect position tracking
- Verify by checking Binance web interface trade history vs strategy logs

## Testing

**Configuration Test**:
```bash
# Test market toggle feature
docker exec godzilla-dev python3 << 'EOF'
import json
config = {
    "user_id": "test_user",
    "access_key": "test_key",
    "secret_key": "test_secret",
    "enable_spot": False,
    "enable_futures": True
}
print(json.dumps(config, indent=2))
EOF
```

**Symbol Conversion Test**:
```cpp
#include "type_convert_binance.h"
#include <cassert>

void test_symbol_conversion() {
    assert(to_binance_symbol("ltc_usdt") == "LTCUSDT");
    assert(from_binance_symbol("LTCUSDT") == "ltc_usdt");
    assert(to_binance_symbol("btc_bnb") == "BTCBNB");
    assert(from_binance_symbol("BTCBNB") == "btc_bnb");
}
```

**Integration Test**:
```bash
# Start TD gateway with Futures-only config
pm2 start ecosystem.config.js --only td_binance

# Check logs for correct initialization
pm2 logs td_binance --lines 20 | grep -E '(Spot|Futures|disabled|enabled)'
# Expected:
# [info] Spot market disabled by configuration
# [info] Connecting BINANCE TD for gz_user1 (Spot: disabled, Futures: enabled)
```

## Dependencies

**External Libraries**:
- **binapi**: Binance API C++ client (WebSocket + REST)
  - Location: Likely vendored or linked externally
  - Provides `binapi::ws::websockets`, `binapi::rest::api`
- **Boost.Asio**: Async I/O and networking
- **nlohmann/json**: JSON parsing
- **spdlog**: Logging

**Internal Dependencies**:
- [yijinjing](yijinjing.md): Event sourcing, journal I/O
- [wingchun](wingchun.md): Broker interfaces, message types

**Build**:
```bash
# Compile Binance extension (requires C++17)
cd /app/core/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make binance -j$(nproc)

# Install Python bindings
cd /app/core/python
pip3 install -e .
```

## References

**Code Locations**:
- Extension root: `core/extensions/binance/`
- Headers: `core/extensions/binance/include/`
- Sources: `core/extensions/binance/src/`
- Python bindings: `core/extensions/binance/src/pybind_binance.cpp`

**Documentation**:
- Market toggle feature: [.doc/95_adr/004-binance-market-toggle.md](../95_adr/004-binance-market-toggle.md)
- Testnet setup: [.doc/00_index/TESTNET.md](../00_index/TESTNET.md)
- Config examples: [.doc/40_config/examples/binance_market_toggle_config.json](../40_config/examples/binance_market_toggle_config.json)
- Debugging cases: [.doc/85_memory/DEBUGGING.md](../85_memory/DEBUGGING.md)

**External Resources**:
- Binance Spot Testnet: https://testnet.binance.vision/
- Binance Futures Testnet: https://testnet.binancefuture.com/
- Binance API Docs: https://developers.binance.com/

## Change History

- **2025-11-17**: Initial module card created
- **2025-11-04**: Market toggle feature implemented (ADR-004)
- **2025-03-03**: Extension created by kx@godzilla.dev
