---
title: Wingchun (咏春) - Trading Gateway System
updated_at: 2025-11-17
owner: core-dev
lang: en
tokens_estimate: 7500
layer: 10_modules
tags: [wingchun, trading, gateway, market-data, order-execution, strategy]
purpose: "Trading abstraction layer providing MD/TD gateways and strategy framework built on yijinjing"
code_refs:
  - core/cpp/wingchun/include/kungfu/wingchun/broker/marketdata.h
  - core/cpp/wingchun/include/kungfu/wingchun/broker/trader.h
  - core/cpp/wingchun/include/kungfu/wingchun/strategy/strategy.h
  - core/cpp/wingchun/src/strategy/context.cpp
---

# Wingchun (咏春) - Trading Gateway System

## Purpose

**Wingchun** is the trading abstraction layer built on top of **yijinjing** (the event sourcing framework). While yijinjing provides low-level event journaling and messaging infrastructure, wingchun provides high-level trading primitives and gateway patterns for cryptocurrency/financial markets.

**Key Relationship**:
- **Yijinjing**: Event sourcing engine, journal storage, message bus, time-travel capabilities
- **Wingchun**: Trading business logic, gateway interfaces, order/position management, strategy framework

**Problem it solves**:
- Unified API across multiple cryptocurrency exchanges
- Separation of market data (MD) and trading (TD) concerns
- Type-safe trading messages with full event history
- Strategy framework with lifecycle management
- Position and account tracking with ledger service

**Core concept**: Gateway pattern separates market data ingestion from order execution, with strategies consuming events and submitting orders through a clean context API.

## Public API

### Gateway Interfaces

#### `MarketData` - Market Data Gateway Base Class

**Location**: [core/cpp/wingchun/include/kungfu/wingchun/broker/marketdata.h:22-49](../../core/cpp/wingchun/include/kungfu/wingchun/broker/marketdata.h#L22-L49)

**Purpose**: Base class for all market data gateways (Binance MD, OKX MD, etc.)

**Inherits From**: `practice::apprentice` (yi jinjing participant)

**Pure Virtual Methods** (must be implemented by exchange connectors):
```cpp
virtual void subscribe(const std::vector<msg::data::Instrument>& instruments) = 0;
virtual void subscribe_trade(const std::vector<msg::data::Instrument>& instruments) = 0;
virtual void subscribe_ticker(const std::vector<msg::data::Instrument>& instruments) = 0;
virtual void subscribe_index_price(const std::vector<msg::data::Instrument>& instruments) = 0;
virtual void subscribe_all() = 0;
virtual void unsubscribe(const std::vector<msg::data::Instrument>& instruments) = 0;
```

**Common Pattern**:
```cpp
class BinanceMarketData : public MarketData {
public:
    void subscribe(const std::vector<Instrument>& instruments) override {
        // Connect to Binance WebSocket
        // Subscribe to depth streams
        // Publish Depth events via write_to(0, msg::type::Depth, depth)
    }
};
```

**Implementation**: [core/cpp/wingchun/src/broker/marketdata.cpp:23-73](../../core/cpp/wingchun/src/broker/marketdata.cpp#L23-L73)

**Location Category**: `category::MD` (marketdata.cpp:26)

#### `Trader` - Trading Gateway Base Class

**Location**: [core/cpp/wingchun/include/kungfu/wingchun/broker/trader.h:22-75](../../core/cpp/wingchun/include/kungfu/wingchun/broker/trader.h#L22-L75)

**Purpose**: Base class for all trading gateways (account-specific order execution)

**Inherits From**: `practice::apprentice`

**Pure Virtual Methods**:
```cpp
virtual void insert_order(const event_ptr& event) = 0;          // Submit new order
virtual void cancel_order(const event_ptr& event) = 0;          // Cancel existing order
virtual void query_order(const event_ptr& event) = 0;           // Query order status
virtual bool req_position() = 0;                                 // Request position snapshot
virtual bool req_account() = 0;                                  // Request account balance
virtual void adjust_leverage(const event_ptr& event) = 0;        // Change leverage (futures)
virtual void merge_position(const event_ptr& event) = 0;         // Merge positions (futures)
```

**Common Pattern**:
```cpp
class BinanceTrader : public Trader {
public:
    void insert_order(const event_ptr& event) override {
        const OrderInput& order = event->data<OrderInput>();
        // Validate order
        // Send to Binance REST API
        // Publish Order event with status
        write_to(order.source, msg::type::Order, order_response);
    }
};
```

**Implementation**: [core/cpp/wingchun/src/broker/trader.cpp:25-74](../../core/cpp/wingchun/src/broker/trader.cpp#L25-L74)

**Location Category**: `category::TD` (trader.cpp:27)

### Strategy Framework

#### `Strategy` - User Strategy Interface

**Location**: [core/cpp/wingchun/include/kungfu/wingchun/strategy/strategy.h:27-103](../../core/cpp/wingchun/include/kungfu/wingchun/strategy/strategy.h#L27-L103)

**Purpose**: Virtual callback interface for user-defined trading strategies

**Lifecycle Hooks**:
```cpp
virtual void pre_start(Context_ptr context) {}   // Before strategy starts
virtual void post_start(Context_ptr context) {}  // After strategy starts
virtual void pre_stop(Context_ptr context) {}    // Before strategy stops
virtual void post_stop(Context_ptr context) {}   // After strategy stops
```

**Market Data Callbacks**:
```cpp
virtual void on_depth(Context_ptr context, const msg::data::Depth& depth) {}
virtual void on_ticker(Context_ptr context, const msg::data::Ticker& ticker) {}
virtual void on_transaction(Context_ptr context, const msg::data::Trade& trade) {}
virtual void on_bar(Context_ptr context, const msg::data::Bar& bar) {}
virtual void on_index_price(Context_ptr context, const msg::data::IndexPrice& index_price) {}
```

**Trading Callbacks**:
```cpp
virtual void on_order(Context_ptr context, const msg::data::Order& order) {}
virtual void on_trade(Context_ptr context, const msg::data::MyTrade& trade) {}
virtual void on_position(Context_ptr context, const msg::data::Position& position) {}
virtual void on_order_action_error(Context_ptr context, const msg::data::OrderActionError& error) {}
```

**Account Callbacks**:
```cpp
virtual void on_asset(Context_ptr context, const msg::data::Asset& asset) {}
virtual void on_asset_margin(Context_ptr context, const msg::data::AssetMargin& asset_margin) {}
```

**Usage Example**:
```cpp
class MyStrategy : public Strategy {
public:
    void pre_start(Context_ptr ctx) override {
        ctx->add_account("binance_td", "my_account");
        ctx->subscribe("binance_md", {"BTCUSDT"}, InstrumentType::Spot, Exchange::BINANCE);
    }

    void on_depth(Context_ptr ctx, const Depth& depth) override {
        if (depth.asks[0].price < threshold) {
            ctx->insert_order("BTCUSDT", InstrumentType::Spot, Exchange::BINANCE,
                             "my_account", depth.asks[0].price, 0.001, Side::Buy);
        }
    }

    void on_order(Context_ptr ctx, const Order& order) override {
        if (order.status == OrderStatus::Filled) {
            LOG_INFO("Order filled: {}", order.order_id);
        }
    }
};
```

#### `Context` - Strategy Execution Context

**Location**: [core/cpp/wingchun/include/kungfu/wingchun/strategy/context.h:25-184](../../core/cpp/wingchun/include/kungfu/wingchun/strategy/context.h#L25-L184)

**Purpose**: Provides API for strategies to interact with trading system

**Account Management**:
```cpp
void add_account(const std::string& source, const std::string& account);
void add_accounts(const std::vector<std::string>& sources, const std::string& account);
```

**Market Data Subscription**:
```cpp
void subscribe(const std::string& source,
               const std::vector<std::string>& instruments,
               msg::data::InstrumentType type = InstrumentType::Unknown,
               uint8_t exchange = Exchange::Unknown);
```

**Order Submission** ([context.cpp:360-401](../../core/cpp/wingchun/src/strategy/context.cpp#L360-L401)):
```cpp
uint64_t insert_order(const std::string& symbol,
                      msg::data::InstrumentType type,
                      uint8_t exchange,
                      const std::string& account,
                      double limit_price,
                      int64_t volume,
                      msg::data::Side side,
                      msg::data::OrderType order_type = OrderType::Limit,
                      msg::data::TimeCondition time_condition = TimeCondition::GTC,
                      ...);
// Returns: order_id (frame UID)
```

**Order Cancellation** ([context.cpp:436-454](../../core/cpp/wingchun/src/strategy/context.cpp#L436-L454)):
```cpp
uint64_t cancel_order(const std::string& account,
                      uint64_t order_id,
                      const std::string& symbol = "",
                      const std::string& ex_order_id = "",
                      msg::data::InstrumentType type = InstrumentType::Unknown);
// Returns: action_id
```

**Book Access**:
```cpp
BookContext_ptr get_book_context();  // Order/position tracking
```

**Timer Services** (inherited from apprentice):
```cpp
void add_timer(int64_t nanotime, std::function<void(event_ptr)> callback);
void add_time_interval(int64_t duration, std::function<void(event_ptr)> callback);
```

**Time Access**:
```cpp
int64_t now();  // Current event timestamp
```

**Implementation**: [core/cpp/wingchun/src/strategy/context.cpp](../../core/cpp/wingchun/src/strategy/context.cpp)

#### `Runner` - Strategy Orchestrator

**Location**: [core/cpp/wingchun/include/kungfu/wingchun/strategy/runner.h:21-43](../../core/cpp/wingchun/include/kungfu/wingchun/strategy/runner.h#L21-L43)

**Purpose**: Manages multiple strategy instances and routes events to them

**Key Fields**:
```cpp
std::map<uint32_t, Strategy_ptr> strategies_;  // strategy_id → strategy instance
```

**Event Routing** ([runner.cpp:66-193](../../core/cpp/wingchun/src/strategy/runner.cpp#L66-L193)):
- Receives all events from yijinjing reader
- Routes `Depth`, `Ticker`, `Order`, `MyTrade`, etc. to appropriate strategy callbacks
- Manages strategy lifecycle (start/stop)

**Usage**:
```cpp
Runner runner(home_location, low_latency);
runner.add_strategy(strategy_id, std::make_shared<MyStrategy>());
runner.run();  // Starts event loop
```

### Message Types

**Location**: [core/cpp/wingchun/include/kungfu/wingchun/msg.h](../../core/cpp/wingchun/include/kungfu/wingchun/msg.h)

#### Market Data Structures

**`Depth`** (msg.h:242-299) - 10-Level Orderbook:
```cpp
struct Depth {
    uint32_t source;
    int64_t rcv_time;
    char instrument_id[32];
    uint8_t instrument_type;
    uint8_t exchange_id;

    struct Level { double price; int64_t volume; };
    Level bids[10];    // Best bid at index 0
    Level asks[10];    // Best ask at index 0
};
```

**`Ticker`** (msg.h:176-210) - Best Bid/Ask Snapshot:
```cpp
struct Ticker {
    char instrument_id[32];
    uint8_t instrument_type;
    uint8_t exchange_id;

    double last_price;
    double bid_price;
    int64_t bid_volume;
    double ask_price;
    int64_t ask_volume;

    int64_t volume;        // 24h volume
    double turnover;       // 24h turnover
    double high_price;     // 24h high
    double low_price;      // 24h low
    double open_price;     // 24h open
    double close_price;    // Latest price
};
```

**`Trade`** (msg.h:331-369) - Public Trade Execution:
```cpp
struct Trade {
    char instrument_id[32];
    uint8_t instrument_type;
    uint8_t exchange_id;

    int64_t trade_time;
    double price;
    int64_t volume;
    uint8_t side;  // Buy or Sell
};
```

**`Bar`** (msg.h:446-474) - OHLCV Candlestick:
```cpp
struct Bar {
    char instrument_id[32];
    uint8_t instrument_type;
    uint8_t exchange_id;

    int64_t start_time;
    int64_t end_time;
    double open;
    double high;
    double low;
    double close;
    int64_t volume;
};
```

**`Instrument`** (msg.h:88-142) - Symbol Metadata:
```cpp
struct Instrument {
    char instrument_id[32];
    uint8_t instrument_type;
    uint8_t exchange_id;

    double contract_multiplier;
    double price_tick;             // Min price increment
    int64_t size_tick;             // Min volume increment

    double long_margin_ratio;      // Futures margin
    double short_margin_ratio;

    int64_t expire_date;           // Futures expiry
    int64_t delivery_date;
};
```

#### Trading Structures

**`OrderInput`** (msg.h:496-540) - Order Submission:
```cpp
struct OrderInput {
    uint32_t order_id;             // Unique ID (frame UID)
    uint32_t strategy_id;          // Strategy location UID

    char instrument_id[32];
    uint8_t instrument_type;
    uint8_t exchange_id;
    char account_id[64];

    double limit_price;
    int64_t volume;
    uint8_t side;                  // Buy/Sell/Lock/Unlock
    uint8_t position_side;         // Long/Short (futures)
    uint8_t order_type;            // Limit/Market/Mock
    uint8_t time_condition;        // GTC/IOC/FOK/GTX/POC

    bool reduce_only;              // Futures: only reduce position
};
```

**`Order`** (msg.h:666-741) - Order State Tracking:
```cpp
struct Order {
    uint32_t order_id;             // Internal ID
    char ex_order_id[64];          // Exchange order ID

    uint8_t status;                // OrderStatus enum

    int64_t volume;                // Original volume
    int64_t volume_traded;         // Filled volume
    int64_t volume_left;           // Remaining volume

    double limit_price;            // Order price
    double avg_price;              // Average fill price
    double fee;                    // Total trading fee

    int64_t insert_time;           // Submission time
    int64_t update_time;           // Last update time

    // ... (includes all OrderInput fields)
};
```

**`MyTrade`** (msg.h:827-899) - Execution Fill:
```cpp
struct MyTrade {
    uint64_t trade_id;             // Unique trade ID
    uint32_t order_id;             // Related order ID
    char ex_trade_id[64];          // Exchange trade ID

    int64_t trade_time;
    double price;
    int64_t volume;
    double fee;                    // Trade fee

    // ... (includes order context)
};
```

**`Position`** (msg.h:1000-1051) - Position Snapshot:
```cpp
struct Position {
    char instrument_id[32];
    uint8_t instrument_type;
    uint8_t exchange_id;
    char account_id[64];

    uint8_t direction;             // Long/Short
    int64_t volume;                // Position size
    int64_t frozen_volume;         // Pending close volume

    double margin;                 // Used margin
    double position_pnl;           // Position P&L
    double close_pnl;              // Realized P&L

    double avg_open_price;         // Average open price
    double last_price;             // Current market price

    int leverage;                  // Leverage multiplier (futures)
};
```

**`Asset`** (msg.h:947-983) - Account Balance:
```cpp
struct Asset {
    char account_id[64];
    uint8_t asset_type;            // Crypto/Fiat

    double initial_equity;
    double static_equity;
    double dynamic_equity;
    double realized_pnl;
    double unrealized_pnl;

    double avail;                  // Available balance
    double frozen;                 // Frozen balance
    double margin;                 // Used margin
    double fee;                    // Accumulated fees
};
```

#### Message Type Enum

**Location**: [msg.h:25-69](../../core/cpp/wingchun/include/kungfu/wingchun/msg.h#L25-L69)

```cpp
namespace msg::type {
    // Market data
    const int32_t Depth = 101;
    const int32_t Ticker = 102;
    const int32_t Trade = 103;
    const int32_t IndexPrice = 104;
    const int32_t Bar = 110;
    const int32_t Instrument = 120;

    // Trading
    const int32_t OrderInput = 201;
    const int32_t Order = 203;
    const int32_t MyTrade = 204;
    const int32_t Position = 205;
    const int32_t Asset = 206;

    // Actions
    const int32_t Subscribe = 302;
    const int32_t CancelOrder = 354;
    const int32_t AdjustLeverage = 352;
    const int32_t MergePosition = 353;
}
```

### Book Tracking

#### `Book` - Order/Position Tracking Interface

**Location**: [core/cpp/wingchun/include/kungfu/wingchun/book/book.h:22-47](../../core/cpp/wingchun/include/kungfu/wingchun/book/book.h#L22-L47)

**Purpose**: Abstract interface for account-specific order and position tracking

**Callbacks**:
```cpp
virtual void on_depth(const event_ptr& event, const Depth& depth) {}
virtual void on_trade(const event_ptr& event, const MyTrade& trade) {}
virtual void on_order(const event_ptr& event, const Order& order) {}
virtual void on_order_input(const event_ptr& event, const OrderInput& input) {}
virtual void on_position(const event_ptr& event, const Position& position) {}
virtual void on_asset(const event_ptr& event, const Asset& asset) {}
```

**Subclasses**: Implement account-specific bookkeeping logic (P&L calculation, risk checks, etc.)

#### `BookContext` - Multi-Book Manager

**Location**: [book.h:51-82](../../core/cpp/wingchun/include/kungfu/wingchun/book/book.h#L51-L82)

**Purpose**: Manages multiple book instances, provides instrument info lookup

**Key Methods**:
```cpp
const Instrument& get_inst_info(const std::string& symbol, uint8_t exchange);
void monitor_instruments(const std::vector<std::string>& symbols);
void monitor_positions(const std::string& account);
```

## Inputs / Outputs

### Inputs

**Strategy Inputs** (via Context):
- Market data subscriptions: `(source, symbols, instrument_type, exchange)`
- Order parameters: `(symbol, type, exchange, account, price, volume, side, order_type, ...)`
- Account registration: `(td_source, account_id)`

**Gateway Inputs**:
- MD gateway: `Subscribe` events with instrument list
- TD gateway: `OrderInput`, `CancelOrder`, `AdjustLeverage` events from strategies

### Outputs

**MD Gateway Outputs** (to journal, dest_id=0, broadcast):
- `Depth` events: 10-level orderbook updates
- `Ticker` events: Best bid/ask snapshots
- `Trade` events: Public trade executions
- `Bar` events: OHLCV candlesticks

**TD Gateway Outputs** (to journal, dest_id=strategy_location_uid, targeted):
- `Order` events: Order status updates (Submitted, Filled, Cancelled, etc.)
- `MyTrade` events: Execution fill notifications
- `Position` events: Position snapshots
- `Asset` events: Account balance updates
- `OrderActionError` events: Order rejection errors

**Strategy Outputs** (to journal):
- `OrderInput` events → TD gateways
- `CancelOrder` events → TD gateways
- Custom events for inter-strategy communication

## Dependencies

### External Libraries

- **yijinjing** - Event sourcing foundation (journal, reader, writer, practice framework)
- **RxCpp** - Reactive extensions for event filtering (`events_ | is(msg::type::Order)`)
- **spdlog** - Logging
- **nlohmann/json** - JSON serialization (config parsing, API communication)
- **pybind11** - Python bindings for strategies

### Internal Dependencies

- **kungfu/yijinjing/** - Core event sourcing infrastructure
- **kungfu/common.h** - Shared macros and types

### System APIs

- **REST clients** - HTTP libraries for exchange API calls (libcurl, etc.)
- **WebSocket clients** - WS libraries for real-time market data
- **SSL/TLS** - Secure communication with exchanges

## Architecture

### Gateway Pattern - MD vs TD Separation

```
┌─────────────────────────────────────────────────────────────────┐
│                       Strategy Layer                             │
│           (category::STRATEGY - business logic)                  │
│  - Receives MD events (Depth, Ticker, Trade)                    │
│  - Sends TD commands (OrderInput, CancelOrder)                  │
│  - Manages positions and risk via BookContext                   │
└─────────────────┬──────────────────────┬────────────────────────┘
                  │                      │
       ┌──────────▼──────────┐  ┌───────▼──────────┐
       │  MarketData (MD)    │  │   Trader (TD)    │
       │  category::MD       │  │   category::TD   │
       │  dest_id=0          │  │   dest_id=uid    │
       │                     │  │                  │
       │  - subscribe()      │  │  - insert_order()│
       │  - subscribe_trade()│  │  - cancel_order()│
       │  - subscribe_all()  │  │  - req_position()│
       │  - Publish events   │  │  - req_account() │
       │    (broadcast)      │  │  - Publish events│
       │                     │  │    (targeted)    │
       └──────────┬──────────┘  └───────┬──────────┘
                  │                     │
                  └──────────┬──────────┘
                             │
                    ┌────────▼─────────┐
                    │  Exchange APIs   │
                    │  (Binance, OKX)  │
                    │  REST + WebSocket│
                    └──────────────────┘
```

**Key Design Decisions**:
1. **MD/TD separation**: Market data is read-only, trading has authentication
2. **Broadcast vs Targeted**: MD events broadcast (dest_id=0), TD events targeted to specific strategies
3. **Stateless gateways**: All state stored in yijinjing journals, gateways are ephemeral
4. **Type safety**: Strongly-typed message structs prevent runtime errors
5. **Event replay**: Full system state can be reconstructed from journal

### Event Flow - Order Submission

**Complete Flow** (Strategy → TD Gateway → Exchange → Strategy):

```cpp
// 1. Strategy submits order (context.cpp:360-401)
uint64_t order_id = context->insert_order(
    "BTCUSDT", InstrumentType::Spot, Exchange::BINANCE,
    "my_account", 50000.0, 0.001, Side::Buy
);

// Behind the scenes:
//   a. Context creates OrderInput message:
OrderInput input;
input.order_id = gen_order_id();  // frame UID
input.strategy_id = get_location_uid();
input.limit_price = 50000.0;
input.volume = 0.001;
// ...

//   b. Lookup TD gateway location:
uint32_t td_location = lookup_account_location_id("my_account");

//   c. Write OrderInput event to TD gateway's journal:
writer->write(now, msg::type::OrderInput, input);  // yijinjing write

// 2. TD Gateway receives event (trader.cpp:37-42)
events_ | is(msg::type::OrderInput) | $([&](event_ptr e) {
    insert_order(e);  // Virtual method, implemented by BinanceTrader
});

// 3. Exchange-specific implementation (binance extension)
void BinanceTrader::insert_order(const event_ptr& event) {
    const OrderInput& input = event->data<OrderInput>();

    // Send REST API call to Binance
    auto response = binance_rest_client->post_order(...);

    // Publish Order event with Submitted status
    Order order = {...};  // Populate from input + response
    order.status = OrderStatus::Submitted;
    write_to(input.strategy_id, msg::type::Order, order);
}

// 4. Strategy receives order update (strategy.h:69)
void MyStrategy::on_order(Context_ptr ctx, const Order& order) {
    if (order.order_id == my_order_id) {
        if (order.status == OrderStatus::Filled) {
            LOG_INFO("Order filled at {}", order.avg_price);
        }
    }
}
```

**Time Complexity**: O(1) event write + network latency + O(1) event delivery

### Event Flow - Market Data

**Complete Flow** (MD Gateway → Strategy):

```cpp
// 1. Strategy subscribes (context.cpp:201-240)
context->subscribe("binance_md", {"BTCUSDT", "ETHUSDT"},
                   InstrumentType::Spot, Exchange::BINANCE);

// Behind the scenes:
//   a. Send Subscribe message to MD gateway:
Subscribe sub;
sub.instruments = {"BTCUSDT", "ETHUSDT"};
sub.instrument_type = InstrumentType::Spot;
write_to(md_location_uid, msg::type::Subscribe, sub);

//   b. Request read permission from master:
request_read_from(now, md_location_uid);

// 2. MD Gateway receives subscription (marketdata.cpp:42-69)
void BinanceMarketData::on_subscribe(const event_ptr& event) {
    const Subscribe& sub = event->data<Subscribe>();
    subscribe(sub.instruments);  // Virtual method
}

void BinanceMarketData::subscribe(const std::vector<Instrument>& instruments) {
    // Connect to Binance WebSocket
    for (const auto& inst : instruments) {
        ws_client->subscribe_depth(inst.instrument_id);
    }
}

// 3. WebSocket callback publishes events
void BinanceMarketData::on_ws_depth_update(const json& data) {
    Depth depth = parse_depth(data);

    // Broadcast to ALL strategies (dest_id=0)
    write_to(0, msg::type::Depth, depth);
}

// 4. Runner routes event to strategy (runner.cpp:66-76)
events_ | is(msg::type::Depth) | $([&](event_ptr e) {
    const Depth& depth = e->data<Depth>();

    // Check if strategy subscribed to this instrument
    if (is_subscribed(depth.instrument_id)) {
        strategy->on_depth(context, depth);
    }
});

// 5. Strategy callback
void MyStrategy::on_depth(Context_ptr ctx, const Depth& depth) {
    double spread = depth.asks[0].price - depth.bids[0].price;
    if (spread < threshold) {
        // Trade signal
    }
}
```

### Integration with Yijinjing

**Location System**:
```cpp
// MD gateway location
location md_loc(mode::LIVE, category::MD, "binance", "marketdata", locator);
// → uname: "md/binance/marketdata/live"
// → uid: hash("md/binance/marketdata/live")

// TD gateway location
location td_loc(mode::LIVE, category::TD, "binance", "trader", locator);
// → uname: "td/binance/trader/live"

// Strategy location
location strat_loc(mode::LIVE, category::STRATEGY, "demo", "my_strategy", locator);
// → uname: "strategy/demo/my_strategy/live"
```

**Apprentice Inheritance**:
```cpp
class MarketData : public practice::apprentice {
    // Inherits from apprentice → hero → yijinjing event sourcing
    // Automatic journal reading/writing
    // Observable event stream: events_ | is(msg_type) | $(callback)
    // Location registration with master
};
```

**Event Publishing Patterns**:
```cpp
// MD gateway: broadcast (all strategies receive)
write_to(0, msg::type::Depth, depth);

// TD gateway: targeted (specific strategy receives)
write_to(strategy_location_uid, msg::type::Order, order);

// Strategy: order submission (specific TD gateway receives)
write_to(td_location_uid, msg::type::OrderInput, input);
```

**Journal Structure**:
```
{KF_HOME}/runtime/journal/
├── live/
│   ├── md/
│   │   └── binance/
│   │       └── marketdata/
│   │           ├── 00000000.00000001.journal  # Broadcast MD events
│   │           └── 00000000.00000002.journal
│   ├── td/
│   │   └── binance/
│   │       └── trader/
│   │           ├── 12345abc.00000001.journal  # Strategy-specific TD events
│   │           └── 67890def.00000001.journal
│   └── strategy/
│       └── demo/
│           └── my_strategy/
│               └── 00000000.00000001.journal  # Strategy's own events
```

## Supporting Services

### Ledger Service

**Location**: [core/cpp/wingchun/include/kungfu/wingchun/service/ledger.h:22-109](../../core/cpp/wingchun/include/kungfu/wingchun/service/ledger.h#L22-L109)

**Purpose**: Central coordinator for trading system state

**Responsibilities**:
- Track broker connection states (MD/TD gateway status)
- Maintain instrument database (contract specifications, margins)
- Publish unified position/asset snapshots
- Handle request routing between strategies and gateways
- Provide system-wide view of all accounts and positions

**Usage**: Automatically started by kungfu master process

### Bar Generator Service

**Location**: [core/cpp/wingchun/include/kungfu/wingchun/service/bar.h:25-37](../../core/cpp/wingchun/include/kungfu/wingchun/service/bar.h#L25-L37)

**Purpose**: Generate OHLCV candlestick bars from trade events

**Process**:
1. Subscribe to `Trade` events from MD gateways
2. Accumulate trades into time buckets (1m, 5m, 1h, etc.)
3. Publish `Bar` events at interval boundaries
4. Strategies subscribe to `Bar` for technical analysis

**Configuration**: Bar intervals defined in kungfu config

### Commander

**Location**: [core/cpp/wingchun/include/kungfu/wingchun/commander.h:19-50](../../core/cpp/wingchun/include/kungfu/wingchun/commander.h#L19-L50)

**Purpose**: Base class for command handling services

**Responsibilities**:
- Order routing and validation
- Location registration/deregistration
- System command processing

## Common Enums and Constants

**Location**: [core/cpp/wingchun/include/kungfu/wingchun/common.h](../../core/cpp/wingchun/include/kungfu/wingchun/common.h)

### Trading Enums

**`InstrumentType`** (line 65-75):
```cpp
enum class InstrumentType : uint8_t {
    Unknown = 0,
    Spot = 1,        // Spot market
    FFuture = 2,     // Futures (forward)
    DFuture = 3,     // Delivery futures
    Swap = 4,        // Perpetual swap
    Index = 5,       // Index
    ETF = 6          // ETF
};
```

**`Side`** (line 91-99):
```cpp
enum class Side : uint8_t {
    Buy = 0,
    Sell = 1,
    Lock = 2,        // Lock position (futures)
    Unlock = 3       // Unlock position
};
```

**`OrderType`** (line 115-121):
```cpp
enum class OrderType : uint8_t {
    Limit = 0,       // Limit order
    Market = 1,      // Market order
    Mock = 2         // Simulated order (backtesting)
};
```

**`OrderStatus`** (line 139-150):
```cpp
enum class OrderStatus : uint8_t {
    Unknown = 0,
    Submitted = 1,            // Accepted by exchange
    Pending = 2,              // Awaiting confirmation
    Cancelled = 3,            // Cancelled
    Error = 4,                // Rejected
    Filled = 5,               // Fully filled
    PartialFilledActive = 6   // Partially filled, still active
};
```

**`TimeCondition`** (line 130-137):
```cpp
enum class TimeCondition : uint8_t {
    IOC = 0,         // Immediate or Cancel
    FOK = 1,         // Fill or Kill
    GTC = 2,         // Good Till Cancel
    GTX = 3,         // Good Till Crossing (post-only)
    POC = 4          // Partial or Cancel
};
```

**`Direction`** (line 152-156):
```cpp
enum class Direction : uint8_t {
    Long = 0,
    Short = 1
};
```

### Exchange Constants

**Location**: [common.h:23-30](../../core/cpp/wingchun/include/kungfu/wingchun/common.h#L23-L30)

```cpp
namespace Exchange {
    const uint8_t BINANCE = 1;
    const uint8_t XT = 2;
    const uint8_t KUCOIN = 3;
    const uint8_t GATE = 4;
    const uint8_t MEXC = 5;
    const uint8_t OKX = 6;
    const uint8_t BYBIT = 7;
}
```

### Helper Functions

**`is_final_status`** (line 293-306) - Check if order is terminal:
```cpp
inline bool is_final_status(OrderStatus status) {
    return status == OrderStatus::Filled ||
           status == OrderStatus::Cancelled ||
           status == OrderStatus::Error;
}
```

**`get_symbol_id`** (line 349-352) - Generate unique symbol hash:
```cpp
inline uint32_t get_symbol_id(const std::string& symbol, uint8_t exchange) {
    return hash_str_32(symbol + std::to_string(exchange));
}
```

**Floating Point Comparisons** (line 191-224) - Epsilon-based comparison:
```cpp
inline bool double_equals(double a, double b) {
    return std::fabs(a - b) < EPSILON;
}
```

## Usage Examples

### Pattern 1: Simple Market Making Strategy

**Python Strategy** (demo_spot.py):
```python
from kungfu.wingchun.constants import *

class MarketMaker(Strategy):
    def pre_start(self, context):
        # Register trading account
        context.add_account("binance_td", "my_account")

        # Subscribe to market data
        context.subscribe("binance_md", ["btc_usdt"],
                         InstrumentType.Spot, Exchange.BINANCE)

        self.position = 0
        self.target_spread = 0.001  # 0.1%

    def on_depth(self, context, depth):
        mid = (depth.bids[0].price + depth.asks[0].price) / 2

        # Place buy order below mid
        buy_price = mid * (1 - self.target_spread)
        context.insert_order("btc_usdt", InstrumentType.Spot,
                            Exchange.BINANCE, "my_account",
                            buy_price, 0.001, Side.Buy)

        # Place sell order above mid
        sell_price = mid * (1 + self.target_spread)
        context.insert_order("btc_usdt", InstrumentType.Spot,
                            Exchange.BINANCE, "my_account",
                            sell_price, 0.001, Side.Sell)

    def on_order(self, context, order):
        if order.status == OrderStatus.Filled:
            # Update position
            if order.side == Side.Buy:
                self.position += order.volume_traded
            else:
                self.position -= order.volume_traded

            context.log.info(f"Position: {self.position}")
```

### Pattern 2: C++ Strategy with BookContext

```cpp
#include <kungfu/wingchun/strategy/strategy.h>
#include <kungfu/wingchun/book/book.h>

class TrendFollower : public Strategy {
public:
    void pre_start(Context_ptr ctx) override {
        ctx->add_account("binance_td", "my_account");
        ctx->subscribe("binance_md", {"BTCUSDT"}, InstrumentType::Spot, Exchange::BINANCE);

        book_ctx_ = ctx->get_book_context();
        book_ctx_->monitor_instruments({"BTCUSDT"});
        book_ctx_->monitor_positions("my_account");
    }

    void on_bar(Context_ptr ctx, const Bar& bar) override {
        // Simple moving average
        prices_.push_back(bar.close);
        if (prices_.size() > 20) prices_.pop_front();

        double ma = std::accumulate(prices_.begin(), prices_.end(), 0.0) / prices_.size();

        // Trend signal
        if (bar.close > ma * 1.01 && position_ == 0) {
            // Buy signal
            order_id_ = ctx->insert_order("BTCUSDT", InstrumentType::Spot,
                                         Exchange::BINANCE, "my_account",
                                         bar.close, 0.01, Side::Buy);
        }
    }

    void on_position(Context_ptr ctx, const Position& pos) override {
        position_ = pos.volume;
    }

private:
    BookContext_ptr book_ctx_;
    std::deque<double> prices_;
    int64_t position_ = 0;
    uint64_t order_id_ = 0;
};
```

### Pattern 3: Timer-Based Strategy

```cpp
class PeriodicRebalancer : public Strategy {
public:
    void post_start(Context_ptr ctx) override {
        // Rebalance every 1 hour
        int64_t interval = 3600 * time_unit::NANOSECONDS_PER_SECOND;
        ctx->add_time_interval(interval, [this, ctx](event_ptr e) {
            rebalance(ctx);
        });
    }

    void rebalance(Context_ptr ctx) {
        // Query current positions
        auto book = ctx->get_book_context();
        // ... rebalancing logic

        ctx->insert_order(...);
    }
};
```

## Hotspots & Pitfalls

### Hotspot 1: Order ID Management

**Issue**: Order IDs must be unique across the entire system
**Solution**: Context automatically uses yijinjing frame UID (guaranteed unique)

```cpp
// WRONG: Manual order ID generation
uint64_t order_id = rand();  // Collision risk!

// CORRECT: Let context generate ID
uint64_t order_id = ctx->insert_order(...);  // Uses frame UID
```

### Hotspot 2: Market Data Latency

**Issue**: Depth updates may arrive out of order or with delay
**Solution**: Always use `rcv_time` (receive timestamp) vs `gen_time` (event timestamp)

```cpp
void on_depth(Context_ptr ctx, const Depth& depth) {
    int64_t latency = ctx->now() - depth.rcv_time;
    if (latency > 100 * time_unit::NANOSECONDS_PER_MILLISECOND) {
        LOG_WARN("Stale depth: {} ms latency", latency / 1000000);
        return;  // Skip stale data
    }
    // Process depth...
}
```

### Hotspot 3: Position Synchronization

**Issue**: Position updates are asynchronous, may lag behind orders
**Solution**: Track pending orders separately from confirmed positions

```cpp
class MyStrategy : public Strategy {
    int64_t confirmed_position_ = 0;  // From Position events
    int64_t pending_buy_volume_ = 0;  // From OrderInput

    void on_order_input(...) {
        if (input.side == Side::Buy) pending_buy_volume_ += input.volume;
    }

    void on_trade(...) {
        pending_buy_volume_ -= trade.volume;  // Reduce pending
    }

    void on_position(...) {
        confirmed_position_ = pos.volume;
    }

    int64_t effective_position() {
        return confirmed_position_ + pending_buy_volume_;
    }
};
```

### Pitfall 1: Forgetting to Add Account

**Symptom**: Orders fail with "account not found" error
**Cause**: Must call `ctx->add_account()` before inserting orders

```cpp
// WRONG: Insert order without adding account
void pre_start(Context_ptr ctx) {
    ctx->subscribe(...);  // OK
    ctx->insert_order(..., "my_account", ...);  // FAILS!
}

// CORRECT: Add account first
void pre_start(Context_ptr ctx) {
    ctx->add_account("binance_td", "my_account");  // Required!
    ctx->subscribe(...);
}
```

### Pitfall 2: Incorrect Instrument Type

**Symptom**: Orders rejected by exchange
**Cause**: Using wrong InstrumentType (e.g., Spot for futures symbol)

```cpp
// WRONG: Futures symbol with Spot type
ctx->insert_order("BTCUSDT_PERP",
                 InstrumentType::Spot,  // WRONG!
                 Exchange::BINANCE, ...);

// CORRECT: Match instrument type to symbol
ctx->insert_order("BTCUSDT_PERP",
                 InstrumentType::Swap,  // Correct for perpetual
                 Exchange::BINANCE, ...);
```

### Pitfall 3: Not Handling OrderActionError

**Symptom**: Silent order failures, no error logging
**Cause**: Not implementing `on_order_action_error` callback

```cpp
// Add error handler
void on_order_action_error(Context_ptr ctx, const OrderActionError& error) override {
    LOG_ERROR("Order action failed: {} - {}",
             error.error_id, error.error_msg);

    // Implement recovery logic (retry, cancel pending, etc.)
}
```

### Pitfall 4: Floating Point Price Precision

**Symptom**: Orders rejected due to invalid price (too many decimals)
**Cause**: Not rounding price to tick size

```cpp
// WRONG: Arbitrary precision
double price = 50000.123456789;  // Too precise!

// CORRECT: Round to tick size
const Instrument& inst = book_ctx->get_inst_info("BTCUSDT", Exchange::BINANCE);
double price = std::round(50000.123 / inst.price_tick) * inst.price_tick;
// Example: tick=0.01 → price=50000.12
```

### Pitfall 5: Blocking in Callbacks

**Symptom**: Strategy becomes unresponsive, misses market data
**Cause**: Performing slow operations in event callbacks

```cpp
// WRONG: Blocking I/O in callback
void on_depth(Context_ptr ctx, const Depth& depth) {
    auto result = http_client.get("http://slow-api.com/price");  // BLOCKS!
    // ... rest of strategy blocked
}

// CORRECT: Use async or move to separate thread
void on_depth(Context_ptr ctx, const Depth& depth) {
    // Quick decision logic only
    if (should_trade(depth)) {
        ctx->insert_order(...);  // Non-blocking
    }
}
```

## Code References

### Key Headers

**Gateway interfaces:**
- [core/cpp/wingchun/include/kungfu/wingchun/broker/marketdata.h](../../core/cpp/wingchun/include/kungfu/wingchun/broker/marketdata.h) - MD gateway base
- [core/cpp/wingchun/include/kungfu/wingchun/broker/trader.h](../../core/cpp/wingchun/include/kungfu/wingchun/broker/trader.h) - TD gateway base

**Strategy framework:**
- [core/cpp/wingchun/include/kungfu/wingchun/strategy/strategy.h](../../core/cpp/wingchun/include/kungfu/wingchun/strategy/strategy.h) - Strategy interface
- [core/cpp/wingchun/include/kungfu/wingchun/strategy/context.h](../../core/cpp/wingchun/include/kungfu/wingchun/strategy/context.h) - Execution context
- [core/cpp/wingchun/include/kungfu/wingchun/strategy/runner.h](../../core/cpp/wingchun/include/kungfu/wingchun/strategy/runner.h) - Strategy orchestrator

**Message types:**
- [core/cpp/wingchun/include/kungfu/wingchun/msg.h](../../core/cpp/wingchun/include/kungfu/wingchun/msg.h) - All trading message definitions
- [core/cpp/wingchun/include/kungfu/wingchun/common.h](../../core/cpp/wingchun/include/kungfu/wingchun/common.h) - Enums and constants

**Book tracking:**
- [core/cpp/wingchun/include/kungfu/wingchun/book/book.h](../../core/cpp/wingchun/include/kungfu/wingchun/book/book.h) - Order/position tracking

**Services:**
- [core/cpp/wingchun/include/kungfu/wingchun/service/ledger.h](../../core/cpp/wingchun/include/kungfu/wingchun/service/ledger.h) - Central coordinator
- [core/cpp/wingchun/include/kungfu/wingchun/service/bar.h](../../core/cpp/wingchun/include/kungfu/wingchun/service/bar.h) - Bar generator

### Key Implementations

**Brokers:**
- [core/cpp/wingchun/src/broker/marketdata.cpp](../../core/cpp/wingchun/src/broker/marketdata.cpp) - MD gateway implementation
- [core/cpp/wingchun/src/broker/trader.cpp](../../core/cpp/wingchun/src/broker/trader.cpp) - TD gateway implementation

**Strategy:**
- [core/cpp/wingchun/src/strategy/context.cpp](../../core/cpp/wingchun/src/strategy/context.cpp) - Context implementation (order submission logic)
- [core/cpp/wingchun/src/strategy/runner.cpp](../../core/cpp/wingchun/src/strategy/runner.cpp) - Event routing logic

**Python bindings:**
- [core/cpp/wingchun/pybind/pybind_wingchun.cpp](../../core/cpp/wingchun/pybind/pybind_wingchun.cpp) - PyBind11 wrappers

**Python examples:**
- [strategies/demo_spot.py](../../strategies/demo_spot.py) - Sample spot trading strategy

## Related Documentation

- [yijinjing.md](yijinjing.md) - Event sourcing foundation that wingchun is built upon
- [binance_extension.md](binance_extension.md) - Example gateway implementation for Binance (to be created)
- [../20_interactions/trading_flow.md](../20_interactions/trading_flow.md) - Complete order execution flow (to be created)
- [../20_interactions/event_flow.md](../20_interactions/event_flow.md) - System-wide event propagation (to be created)
- [../40_config/config_usage_map.md](../40_config/config_usage_map.md) - Configuration for wingchun components
- [../00_index/ARCHITECTURE.md](../00_index/ARCHITECTURE.md) - Overall system architecture

## Changelog

- **2025-11-17**: Initial module card creation with comprehensive trading API documentation

---

**Maintenance Note**: When modifying wingchun, update this document with:
1. New message types in `msg.h`
2. New gateway virtual methods
3. New strategy callbacks
4. Changes to order submission/cancellation flow
5. New supported exchanges in `common.h`
