---
title: Trading Flow - Order Execution End-to-End
updated_at: 2025-11-17
owner: core-dev
lang: en
tokens_estimate: 12000
layer: 20_interactions
tags: [trading, order-execution, interaction, event-flow, strategy, ledger, trader]
purpose: "Complete order lifecycle from strategy insertion to trade confirmation with error handling paths"
code_refs:
  - core/cpp/wingchun/src/strategy/context.cpp:360-401
  - core/cpp/wingchun/src/broker/trader.cpp:26-80
  - core/cpp/wingchun/src/service/ledger.cpp
  - core/extensions/binance/src/trader_binance.cpp:123-200
---

# Trading Flow - Order Execution End-to-End

## Overview

This document traces the complete order execution flow through the kungfu trading system, from a Python strategy's `insert_order()` call to the final trade confirmation event. It covers all system components involved, error handling paths, and the event-driven architecture that enables high-performance, auditable trading.

**Scope**: Full order lifecycle including insertion, routing, execution, acknowledgment, and position updates

**Architecture Pattern**: Event-sourcing with pub/sub messaging via yijinjing journals

## System Architecture Context

```
┌───────────────────────────────────────────────────────────────┐
│ Python Strategy Process (策略进程)                            │
│   strategy.on_depth() → context.insert_order()               │
└───────────────────┬───────────────────────────────────────────┘
                    │ (1) OrderInput event written to strategy journal
                    ↓
┌───────────────────────────────────────────────────────────────┐
│ Ledger Service (账本服务 - SYSTEM/service/ledger)             │
│   - Monitors all journals                                     │
│   - Routes OrderInput → TD Gateway                            │
│   - Aggregates Order/Trade events → Strategy                 │
│   - Publishes consolidated state via nanomsg                  │
└───────────────────┬───────────────────────────────────────────┘
                    │ (2) Event routing via journal subscriptions
                    ↓
┌───────────────────────────────────────────────────────────────┐
│ TD Gateway Process (TD/binance/gz_user1)                      │
│   broker::Trader → TraderBinance                              │
│   - Reads OrderInput from strategy journal                    │
│   - Executes order via exchange REST API                      │
│   - Writes Order/Trade events to TD journal                   │
└───────────────────┬───────────────────────────────────────────┘
                    │ (3) REST API call to exchange
                    ↓
┌───────────────────────────────────────────────────────────────┐
│ Exchange (Binance Testnet/Mainnet)                            │
│   - Processes order                                            │
│   - Returns order confirmation (orderId, status, fills)       │
│   - Sends WebSocket updates (userDataStream)                  │
└───────────────────┬───────────────────────────────────────────┘
                    │ (4) Order confirmation + WebSocket updates
                    ↓
┌───────────────────────────────────────────────────────────────┐
│ TD Gateway (continued)                                         │
│   - Receives REST response → emits Order event                │
│   - Receives WebSocket update → emits Order/Trade events      │
│   - Writes events to TD journal                               │
└───────────────────┬───────────────────────────────────────────┘
                    │ (5) Order/Trade events written to TD journal
                    ↓
┌───────────────────────────────────────────────────────────────┐
│ Ledger Service (continued)                                     │
│   - Reads Order/Trade from TD journal                         │
│   - Routes events back to Strategy journal                    │
└───────────────────┬───────────────────────────────────────────┘
                    │ (6) Events routed to strategy
                    ↓
┌───────────────────────────────────────────────────────────────┐
│ Python Strategy Process (continued)                           │
│   strategy.on_order() → log/update state                      │
│   strategy.on_trade() → log/update position                   │
└───────────────────────────────────────────────────────────────┘
```

## Complete Flow Sequence

### Phase 1: Order Insertion (Strategy → Ledger)

**Step 1.1: Strategy Invokes Context**

**Location**: Python strategy code (user-written)

```python
# Example: strategy.py
class MyStrategy(Strategy):
    def on_depth(self, context, depth, location, dest):
        # Decision logic
        if should_buy(depth):
            order_id = context.insert_order(
                symbol="ltc_usdt",
                exchange_id=EXCHANGE_BINANCE,
                instrument_type=InstrumentType.FFuture,
                account="gz_user1",
                limit_price=depth.AskPrice[0] - 0.01,
                volume=1.0,
                order_type=OrderType.Limit,
                side=Side.Buy
            )
            self.ctx.log_info(f"Inserted order {order_id:016x}")
```

**Step 1.2: Context Constructs OrderInput**

**Location**: `core/cpp/wingchun/src/strategy/context.cpp:360-401`

**Code Execution**:
```cpp
uint64_t Context::insert_order(
    const std::string &symbol, InstrumentType inst_type, const std::string &exchange,
    const std::string &account, double limit_price, double volume,
    OrderType type, Side side, TimeCondition time, Direction position_side, bool reduce_only)
{
    // (1) Lookup account location ID from hash map
    uint32_t account_location_id = lookup_account_location_id(account);  // e.g., TD/binance/gz_user1 UID

    // (2) Get writer for TD gateway journal
    auto writer = app_.get_writer(account_location_id);

    // (3) Open data frame for OrderInput message
    msg::data::OrderInput &input = writer->open_data<msg::data::OrderInput>(0, msg::type::OrderInput);

    // (4) Populate OrderInput fields
    input.strategy_id = current_strategy_idx;           // e.g., 0 (strategy index)
    input.order_id = writer->current_frame_uid();       // e.g., 0x0000000100000042 (unique order ID)
    strcpy(input.symbol, symbol.c_str());               // "ltc_usdt"
    strcpy(input.exchange_id, exchange.c_str());        // "binance"
    strcpy(input.account_id, account.c_str());          // "gz_user1"
    input.instrument_type = inst_type;                  // InstrumentType::FFuture
    input.price = limit_price;                          // 89.50
    input.volume = volume;                              // 1.0
    input.order_type = type;                            // OrderType::Limit
    input.side = side;                                  // Side::Buy
    input.position_side = position_side;                // Direction::Long
    input.reduce_only = reduce_only;                    // false

    // (5) Commit to journal (atomic write)
    writer->close_data();

    SPDLOG_TRACE("insert order with reduce_only {}@{}", symbol, reduce_only);
    return input.order_id;  // Return to strategy
}
```

**Journal Write**:
```
Journal: /app/runtime/strategy/my_strategy/LIVE/journal/STRATEGY.my_strategy.journal
Frame UID: 0x0000000100000042
Timestamp: 1731849600123456789 (nanoseconds)
Msg Type: OrderInput (0x0011)
Data: {
    order_id: 0x0000000100000042,
    strategy_id: 0,
    symbol: "ltc_usdt",
    exchange_id: "binance",
    account_id: "gz_user1",
    instrument_type: FFuture,
    price: 89.50,
    volume: 1.0,
    order_type: Limit,
    side: Buy,
    position_side: Long,
    reduce_only: false
}
```

**Step 1.3: Ledger Detects OrderInput Event**

**Location**: `core/cpp/wingchun/src/service/ledger.cpp`

**Event Detection**:
```cpp
// Ledger subscribes to all strategy journals during on_start()
events_ | is(msg::type::OrderInput) |
$([&](event_ptr event) {
    // Event detected from strategy journal
    const auto& input = event->data<OrderInput>();
    uint32_t td_location_id = lookup_td_location(input.account_id);  // Find TD gateway UID

    // Route event to TD gateway (write to TD's input journal)
    request_write_to(event->gen_time(), td_location_id);
});
```

**Routing Mechanism**:
- Ledger maintains bidirectional journal subscriptions
- Strategy → Ledger: Read strategy journal
- Ledger → TD: Write to TD gateway's input queue
- TD → Ledger: Read TD journal
- Ledger → Strategy: Write to strategy's input queue

### Phase 2: Order Execution (TD Gateway → Exchange)

**Step 2.1: TD Gateway Receives OrderInput**

**Location**: `core/cpp/wingchun/src/broker/trader.cpp:33-42`

**Event Subscription**:
```cpp
void Trader::on_start() {
    apprentice::on_start();

    // Subscribe to OrderInput events from strategy (via ledger routing)
    events_ | is(msg::type::OrderInput) |
    $([&](event_ptr event) {
        SPDLOG_DEBUG("insert_order in trader");
        insert_order(event);  // Virtual function call → TraderBinance::insert_order()
    });
}
```

**Step 2.2: TraderBinance Executes Order**

**Location**: `core/extensions/binance/src/trader_binance.cpp:123-200`

**Execution Flow**:
```cpp
bool TraderBinance::insert_order(const yijinjing::event_ptr& event) {
    const OrderInput& input = event->data<OrderInput>();

    // (1) Convert OrderInput to Order record
    msg::data::Order order{};
    order_from_input(input, order);
    order.insert_time = now();
    order.update_time = now();

    // (2) Store order in local cache (thread-safe)
    {
        std::lock_guard<std::mutex> lock(order_mtx_);
        order_data_.emplace(input.order_id,
            order_map_record(input.order_id, "", event->source(), now(), order));
    }

    // (3) Route to correct market (Spot vs Futures)
    if (input.instrument_type == InstrumentType::Spot) {
        // Guard clause: Check if Spot enabled (ADR-004)
        if (!config_.enable_spot || !rest_ptr_) {
            SPDLOG_ERROR("Spot market disabled, order rejected");
            // Emit error order event (not shown for brevity)
            return false;
        }

        // (4) Create REST API client for this order (thread-safe)
        auto order_ptr = std::make_shared<binapi::rest::api>(
            ioctx_, config_.spot_rest_host, std::to_string(config_.spot_rest_port),
            config_.access_key, config_.secret_key, 10000);

        // (5) Async REST API call with callback
        order_ptr->new_order(
            to_binance_symbol(input.symbol),          // "ltc_usdt" → "LTCUSDT"
            to_binance(input.side),                   // Side::Buy → binapi::e_side::buy
            to_binance(input.order_type),             // OrderType::Limit → binapi::e_type::limit
            binapi::e_time::GTC,                      // Time in force: Good Till Cancel
            binapi::e_trade_resp_type::FULL,          // Request full response (fills included)
            to_string(input.volume),                  // 1.0 → "1.0"
            to_string(input.limit_price),             // 89.50 → "89.50"

            // (6) Callback when REST response received
            [this, order_id=input.order_id, source=event->source()](
                const char* fl, int ec, std::string errmsg, auto resp) {

                if (ec) {
                    // Error path: API call failed
                    SPDLOG_ERROR("Order {} failed: ec({}), errmsg({})", order_id, ec, errmsg);

                    // Emit error order event
                    msg::data::Order& error_order = get_writer(source)->open_data<msg::data::Order>(0, msg::type::Order);
                    error_order.order_id = order_id;
                    error_order.status = OrderStatus::Error;
                    strcpy(error_order.error_msg, errmsg.c_str());
                    get_writer(source)->close_data();
                    return;
                }

                // Success path: Order accepted by exchange

                // (7) Update local cache with exchange order ID
                {
                    std::lock_guard<std::mutex> lock(order_mtx_);
                    order_data_[order_id].exchange_order_id = std::to_string(resp.orderId);
                }

                // (8) Emit Order event with exchange confirmation
                msg::data::Order& order = get_writer(source)->open_data<msg::data::Order>(0, msg::type::Order);
                order.order_id = order_id;
                order.exchange_order_id = resp.orderId;                     // Binance order ID
                strcpy(order.symbol, input.symbol);
                strcpy(order.exchange_id, input.exchange_id);
                order.instrument_type = input.instrument_type;
                order.price = std::stod(resp.price);
                order.volume = std::stod(resp.origQty);
                order.volume_traded = std::stod(resp.executedQty);          // Filled quantity
                order.status = from_binance(resp.status);                   // NEW/FILLED/PARTIALLY_FILLED
                order.insert_time = resp.transactTime * 1000000;            // ms → ns
                order.update_time = now();
                get_writer(source)->close_data();

                // (9) If order immediately filled, emit Trade event
                if (resp.status == binapi::e_status::filled) {
                    for (const auto& fill : resp.fills) {
                        msg::data::Trade& trade = get_writer(source)->open_data<msg::data::Trade>(0, msg::type::Trade);
                        trade.order_id = order_id;
                        trade.trade_id = now();  // Use timestamp as trade ID
                        strcpy(trade.symbol, input.symbol);
                        trade.price = std::stod(fill.price);
                        trade.volume = std::stod(fill.qty);
                        trade.trade_time = resp.transactTime * 1000000;
                        get_writer(source)->close_data();
                    }
                }
            }
        );
    } else if (input.instrument_type == InstrumentType::FFuture) {
        // Similar logic for Futures with frest_ptr_
        // ... (code omitted for brevity)
    }

    return true;
}
```

**Exchange API Request**:
```http
POST https://testnet.binancefuture.com/fapi/v1/order
Headers:
  X-MBX-APIKEY: YOUR_API_KEY
Body (URL-encoded):
  symbol=LTCUSDT
  side=BUY
  type=LIMIT
  timeInForce=GTC
  quantity=1.0
  price=89.50
  timestamp=1731849600123
  signature=<HMAC-SHA256 signature>
```

**Exchange API Response**:
```json
{
  "orderId": 123456789,
  "symbol": "LTCUSDT",
  "status": "NEW",
  "clientOrderId": "abc123",
  "price": "89.50",
  "origQty": "1.0",
  "executedQty": "0.0",
  "cumQuote": "0.0",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "BUY",
  "transactTime": 1731849600456,
  "fills": []
}
```

**Step 2.3: TD Gateway Emits Order Event**

**Journal Write**:
```
Journal: /app/runtime/td/binance/gz_user1/LIVE/journal/TD.binance.gz_user1.journal
Frame UID: 0x0000000200000123
Timestamp: 1731849600456789000 (nanoseconds)
Msg Type: Order (0x0201)
Data: {
    order_id: 0x0000000100000042,          // Original strategy order ID
    exchange_order_id: 123456789,          // Binance order ID
    symbol: "ltc_usdt",
    exchange_id: "binance",
    instrument_type: FFuture,
    price: 89.50,
    volume: 1.0,
    volume_traded: 0.0,
    status: Submitted,                     // OrderStatus::Submitted (NEW)
    insert_time: 1731849600456000000,
    update_time: 1731849600456789000
}
```

### Phase 3: Order Update Propagation (Ledger Routing)

**Step 3.1: Ledger Reads Order Event from TD Journal**

**Location**: `core/cpp/wingchun/src/service/ledger.cpp`

**Event Routing**:
```cpp
// Ledger subscribes to TD journal (registered during TD startup)
events_ | is(msg::type::Order) |
$([&](event_ptr event) {
    const auto& order = event->data<Order>();

    // Determine target strategy from order metadata
    // (In practice, ledger maintains order_id → strategy_uid mapping)
    uint32_t strategy_uid = lookup_strategy_for_order(order.order_id);

    // Write Order event to strategy journal
    auto writer = get_writer(strategy_uid);
    msg::data::Order& routed_order = writer->open_data<msg::data::Order>(0, msg::type::Order);
    memcpy(&routed_order, &order, sizeof(Order));  // Copy order data
    writer->close_data();

    // Publish consolidated state via nanomsg (for external monitoring)
    nlohmann::json msg = {
        {"type", "order"},
        {"order_id", order.order_id},
        {"status", static_cast<int>(order.status)},
        {"volume_traded", order.volume_traded}
    };
    publish(msg.dump());
});
```

**Step 3.2: Strategy Receives Order Event**

**Location**: Python strategy (user callback)

```python
class MyStrategy(Strategy):
    def on_order(self, context, order, location, dest):
        self.ctx.log_info(
            f"Order {order.order_id:016x} status: {order.status}, "
            f"filled: {order.volume_traded}/{order.volume}, "
            f"exchange_order_id: {order.exchange_order_id}"
        )

        # Example: Cancel if partially filled for too long
        if order.status == OrderStatus.PartialFilledActive:
            if context.now() - order.update_time > 5e9:  # 5 seconds
                context.cancel_order(
                    account="gz_user1",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    exchange_order_id=str(order.exchange_order_id)
                )
```

### Phase 4: Order Fill via WebSocket (Async Updates)

**Step 4.1: TD Gateway WebSocket Stream**

**Location**: `core/extensions/binance/src/trader_binance.cpp:220-280` (`_start_userdata()`)

**User Data Stream Setup**:
```cpp
void TraderBinance::_start_userdata(const InstrumentType type) {
    // (1) Request listen key from exchange REST API
    auto listen_key_resp = (type == InstrumentType::Spot)
        ? rest_ptr_->start_user_data_stream()
        : frest_ptr_->futures_start_user_data_stream();

    if (listen_key_resp.ec != 0) {
        SPDLOG_ERROR("Failed to get listen key: {}", listen_key_resp.errmsg);
        return;
    }

    std::string listen_key = listen_key_resp.v.listenKey;
    listenKeys.push_back(listen_key);

    // (2) Subscribe to WebSocket user data stream
    auto ws_ptr = (type == InstrumentType::Spot) ? ws_ptr_ : fws_ptr_;
    ws_ptr->user_data(
        listen_key.c_str(),

        // (3) Callback for execution reports
        [this, type](const char* fl, int ec, std::string errmsg, auto msg) {
            if (ec) {
                SPDLOG_ERROR("User data stream error: {}", errmsg);
                return false;
            }

            // (4) Handle different event types
            if (msg.e == "executionReport") {
                // Order update
                uint64_t order_id = find_order_id_by_exchange_id(msg.i);  // msg.i = exchange order ID

                if (order_id == 0) {
                    SPDLOG_WARN("Unknown exchange order ID: {}", msg.i);
                    return true;
                }

                // (5) Emit Order event for status update
                msg::data::Order& order = get_writer(0)->open_data<msg::data::Order>(0, msg::type::Order);
                order.order_id = order_id;
                order.exchange_order_id = msg.i;
                strcpy(order.symbol, from_binance_symbol(msg.s).c_str());  // "LTCUSDT" → "ltc_usdt"
                order.status = from_binance(msg.X);  // Order status (e.g., FILLED)
                order.volume_traded = std::stod(msg.z);  // Cumulative filled quantity
                order.update_time = now();
                get_writer(0)->close_data();

                // (6) If this is a trade execution, emit Trade event
                if (msg.x == "TRADE") {  // Execution type
                    msg::data::Trade& trade = get_writer(0)->open_data<msg::data::Trade>(0, msg::type::Trade);
                    trade.order_id = order_id;
                    trade.trade_id = msg.t;  // Trade ID from exchange
                    strcpy(trade.symbol, from_binance_symbol(msg.s).c_str());
                    trade.price = std::stod(msg.L);  // Last executed price
                    trade.volume = std::stod(msg.l);  // Last executed quantity
                    trade.side = from_binance(msg.S);
                    trade.trade_time = msg.T * 1000000;  // ms → ns
                    get_writer(0)->close_data();
                }
            } else if (msg.e == "ACCOUNT_UPDATE") {
                // Position update (not covered in detail here)
                // ... handle position changes ...
            }

            return true;
        }
    );
}
```

**WebSocket Message Example** (Order Filled):
```json
{
  "e": "executionReport",
  "E": 1731849605678,
  "s": "LTCUSDT",
  "c": "abc123",
  "S": "BUY",
  "o": "LIMIT",
  "f": "GTC",
  "q": "1.0",
  "p": "89.50",
  "X": "FILLED",
  "i": 123456789,
  "l": "1.0",
  "z": "1.0",
  "L": "89.49",
  "n": "0.0001",
  "N": "USDT",
  "T": 1731849605678,
  "t": 987654321,
  "x": "TRADE"
}
```

**Step 4.2: Trade Event Routed to Strategy**

**Ledger routing** (same as Step 3.1, but for `msg::type::Trade`):
```
TD Journal → Ledger → Strategy Journal
```

**Strategy Callback**:
```python
class MyStrategy(Strategy):
    def on_trade(self, context, trade, location, dest):
        self.ctx.log_info(
            f"Trade executed: {trade.symbol} {trade.side} "
            f"{trade.volume} @ {trade.price}, "
            f"trade_id={trade.trade_id}"
        )

        # Update internal position tracking
        if trade.side == Side.Buy:
            self.position += trade.volume
        else:
            self.position -= trade.volume
```

## Error Handling Paths

### Error 1: Invalid Account

**Trigger**: Strategy calls `insert_order()` with unknown account

**Location**: `core/cpp/wingchun/src/strategy/context.cpp:456-464`

```cpp
uint32_t Context::lookup_account_location_id(const std::string &account) {
    uint32_t account_id = yijinjing::util::hash_str_32(account);
    if (account_location_ids_.find(account_id) == account_location_ids_.end()) {
        throw wingchun_error("invalid account " + account);  // Exception thrown
    }
    return account_location_ids_[account_id];
}
```

**Result**: Python exception, no OrderInput event written

**Recovery**: Strategy must call `add_account()` during `on_start()`

### Error 2: Exchange API Rejection

**Trigger**: Binance rejects order (e.g., insufficient margin, invalid symbol)

**Location**: `trader_binance.cpp` REST callback (error path)

```cpp
[this, order_id](const char* fl, int ec, std::string errmsg, auto resp) {
    if (ec) {
        // Error code examples:
        // -1121: Invalid symbol
        // -2010: Insufficient balance
        // -2015: Invalid API key
        SPDLOG_ERROR("Order {} failed: ec({}), errmsg({})", order_id, ec, errmsg);

        // Emit error Order event
        msg::data::Order& error_order = get_writer(0)->open_data<msg::data::Order>(0, msg::type::Order);
        error_order.order_id = order_id;
        error_order.status = OrderStatus::Error;
        strcpy(error_order.error_msg, errmsg.c_str());
        get_writer(0)->close_data();
        return;
    }
    // ... success path ...
}
```

**Strategy receives**:
```python
def on_order(self, context, order, location, dest):
    if order.status == OrderStatus.Error:
        self.ctx.log_error(f"Order {order.order_id:016x} failed: {order.error_msg}")
        # Recovery logic (e.g., retry with adjusted params)
```

### Error 3: Network Timeout

**Trigger**: REST API call exceeds 10-second timeout

**Location**: `trader_binance.cpp` REST client initialization

```cpp
auto order_ptr = std::make_shared<binapi::rest::api>(
    ioctx_, config_.spot_rest_host, std::to_string(config_.spot_rest_port),
    config_.access_key, config_.secret_key, 10000);  // ← 10000ms timeout
```

**Behavior**:
- binapi library triggers timeout error
- Callback receives `ec != 0` with timeout error message
- Error Order event emitted (same as Error 2 path)

**Detection**: Check logs for "timeout" in error_msg

**Recovery**: TD gateway does NOT auto-retry (strategy must handle)

### Error 4: Disabled Market (ADR-004)

**Trigger**: Order for Spot market when `enable_spot = false`

**Location**: `trader_binance.cpp:135-145`

```cpp
if (input.instrument_type == InstrumentType::Spot) {
    if (!config_.enable_spot || !rest_ptr_) {
        SPDLOG_ERROR("Spot market disabled, order {} rejected", input.order_id);

        // Emit error Order event
        msg::data::Order& error_order = get_writer(event->source())->open_data<msg::data::Order>(0, msg::type::Order);
        error_order.order_id = input.order_id;
        error_order.status = OrderStatus::Error;
        strcpy(error_order.error_msg, "Spot market disabled by configuration");
        get_writer(event->source())->close_data();
        return false;
    }
    // ... normal execution ...
}
```

**Prevention**: Strategy should query available markets before trading

## Performance Characteristics

**Latency Breakdown** (typical values for Binance Futures Testnet):

| Stage | Component | Latency | Bottleneck |
|-------|-----------|---------|------------|
| Strategy insert | Context::insert_order() | ~10μs | Journal write (mmap) |
| Ledger routing | Event detection + routing | ~50μs | Event filter + copy |
| TD execution | TraderBinance::insert_order() | ~5μs | Mutex lock + cache update |
| REST API call | Network + exchange processing | 100-500ms | Network latency |
| WebSocket update | Exchange → TD callback | 50-200ms | Exchange internal latency |
| Order event routing | TD → Ledger → Strategy | ~100μs | 2× journal writes |
| **Total (round-trip)** | insert_order → on_order | **100-700ms** | Dominated by network |

**Throughput**:
- Max order rate per strategy: ~1000 orders/second (journal write limit)
- Binance rate limit: 50 orders/10s (Spot), 300 orders/10s (Futures)
- **Practical limit**: Exchange rate limits

**Resource Usage**:
- Memory per order: ~512 bytes (Order struct + cache entry)
- Journal space per order: ~600 bytes (frame header + data)
- Disk I/O: Asynchronous (mmap write-back)

## Data Structures

### OrderInput (Strategy → TD)

**Location**: `core/cpp/wingchun/include/kungfu/wingchun/msg.h`

```cpp
struct OrderInput {
    uint64_t order_id;          // Unique order ID (frame UID)
    uint32_t strategy_id;       // Strategy index
    char symbol[SYMBOL_LEN];    // "ltc_usdt"
    char exchange_id[EXCHANGE_ID_LEN];  // "binance"
    char account_id[ACCOUNT_ID_LEN];    // "gz_user1"
    InstrumentType instrument_type;     // Spot/FFuture/CBase
    double price;               // Limit price (0 for market orders)
    double stop_price;          // Stop price (for stop orders)
    double volume;              // Order quantity
    OrderType order_type;       // Limit/Market
    TimeCondition time_condition;  // GTC/IOC/FOK
    Side side;                  // Buy/Sell
    Direction position_side;    // Long/Short (futures only)
    bool reduce_only;           // Reduce-only flag (futures)
};
```

### Order (TD → Strategy)

**Location**: `core/cpp/wingchun/include/kungfu/wingchun/msg.h`

```cpp
struct Order {
    uint64_t order_id;          // Original order ID from strategy
    uint64_t exchange_order_id; // Exchange-assigned order ID
    uint32_t strategy_id;       // Strategy index
    char symbol[SYMBOL_LEN];
    char exchange_id[EXCHANGE_ID_LEN];
    char account_id[ACCOUNT_ID_LEN];
    InstrumentType instrument_type;
    double price;               // Actual order price
    double volume;              // Total quantity
    double volume_traded;       // Filled quantity
    double volume_left;         // Remaining quantity
    OrderStatus status;         // Submitted/PartialFilled/Filled/Cancelled/Error
    int64_t insert_time;        // Order creation time (ns)
    int64_t update_time;        // Last update time (ns)
    char error_msg[ERROR_MSG_LEN];  // Error message (if status = Error)
};
```

### Trade (TD → Strategy)

**Location**: `core/cpp/wingchun/include/kungfu/wingchun/msg.h`

```cpp
struct Trade {
    uint64_t order_id;          // Parent order ID
    uint64_t trade_id;          // Exchange trade ID
    char symbol[SYMBOL_LEN];
    char exchange_id[EXCHANGE_ID_LEN];
    double price;               // Execution price
    double volume;              // Execution quantity
    Side side;                  // Buy/Sell
    int64_t trade_time;         // Execution timestamp (ns)
    char commission_currency[CURRENCY_LEN];  // Fee currency (e.g., "USDT")
    double commission;          // Trading fee
};
```

## Testing

**Unit Test Example** (Order insertion):
```cpp
#include <gtest/gtest.h>
#include <kungfu/wingchun/strategy/context.h>

TEST(TradingFlow, InsertOrderBasic) {
    // Setup mock apprentice and events
    MockApprentice app;
    auto events = rx::observable<>::empty<event_ptr>().publish();
    Context ctx(app, events);

    // Add test account
    ctx.add_account("binance", "test_account");

    // Insert order
    uint64_t order_id = ctx.insert_order(
        "btc_usdt", InstrumentType::Spot, "binance", "test_account",
        50000.0, 0.01, OrderType::Limit, Side::Buy
    );

    // Verify order ID generated
    ASSERT_NE(order_id, 0);

    // Verify OrderInput written to journal
    auto last_frame = app.get_last_written_frame();
    ASSERT_EQ(last_frame->msg_type(), msg::type::OrderInput);

    const auto& input = last_frame->data<OrderInput>();
    ASSERT_EQ(input.order_id, order_id);
    ASSERT_STREQ(input.symbol, "btc_usdt");
    ASSERT_EQ(input.price, 50000.0);
}
```

**Integration Test** (End-to-end with mock exchange):
```python
# test_trading_flow.py
import pytest
from kungfu.wingchun.constants import *

class MockExchange:
    def __init__(self):
        self.orders = {}

    def new_order(self, symbol, side, order_type, quantity, price):
        order_id = len(self.orders) + 1
        self.orders[order_id] = {
            'symbol': symbol, 'side': side, 'status': 'NEW',
            'quantity': quantity, 'filled': 0.0, 'price': price
        }
        return order_id

    def fill_order(self, order_id, quantity, price):
        order = self.orders[order_id]
        order['filled'] += quantity
        if order['filled'] >= order['quantity']:
            order['status'] = 'FILLED'
        return {'trade_id': 123, 'price': price, 'quantity': quantity}

def test_complete_trading_flow(strategy_context, mock_exchange):
    # Strategy inserts order
    order_id = strategy_context.insert_order(
        symbol="btc_usdt",
        exchange_id="mock_exchange",
        instrument_type=InstrumentType.Spot,
        account="test_account",
        limit_price=50000.0,
        volume=0.01,
        order_type=OrderType.Limit,
        side=Side.Buy
    )

    # Wait for Order event (status=Submitted)
    order_event = strategy_context.wait_for_event(msg_type='Order', order_id=order_id, timeout=1.0)
    assert order_event.status == OrderStatus.Submitted

    # Simulate exchange fill
    mock_exchange.fill_order(order_event.exchange_order_id, 0.01, 50000.0)

    # Wait for Trade event
    trade_event = strategy_context.wait_for_event(msg_type='Trade', order_id=order_id, timeout=1.0)
    assert trade_event.volume == 0.01
    assert trade_event.price == 50000.0

    # Wait for final Order event (status=Filled)
    final_order = strategy_context.wait_for_event(msg_type='Order', order_id=order_id, timeout=1.0)
    assert final_order.status == OrderStatus.Filled
    assert final_order.volume_traded == 0.01
```

## Debugging

**Common Issues**:

1. **Order stuck in "Submitted" status**:
   - Check TD gateway logs: `tail -f /app/runtime/td/binance/gz_user1/log/live/gz_user1.log`
   - Look for WebSocket connection errors
   - Verify user data stream is active: `pm2 logs td_binance | grep "user data"`

2. **No on_order() callback**:
   - Verify ledger routing: `pm2 logs ledger | grep OrderInput`
   - Check strategy journal permissions
   - Confirm `add_account()` called in strategy `on_start()`

3. **Duplicate orders**:
   - Check for multiple strategy instances running
   - Verify `order_id` uniqueness in logs
   - Review frame UID generation

**Log Analysis**:
```bash
# Trace complete order flow for order_id 0x0000000100000042
grep "0000000100000042" /app/runtime/**/*.log

# Expected output sequence:
# strategy.log: insert order with reduce_only ltc_usdt@false
# ledger.log: routing OrderInput from strategy → TD
# td.log: insert_order in trader
# td.log: Order 0000000100000042 accepted by exchange, exchange_id=123456789
# ledger.log: routing Order from TD → strategy
# strategy.log: Order 0000000100000042 status: Submitted
```

## Related Documentation

- [yijinjing.md](../10_modules/yijinjing.md) - Event sourcing infrastructure
- [wingchun.md](../10_modules/wingchun.md) - Trading gateway framework
- [binance_extension.md](../10_modules/binance_extension.md) - Binance-specific implementation
- [event_flow.md](event_flow.md) - Generic event propagation patterns
- [ADR-004](../95_adr/004-binance-market-toggle.md) - Market toggle feature impact on order routing

## References

**Code Locations**:
- Strategy context: `core/cpp/wingchun/src/strategy/context.cpp`
- Trader base: `core/cpp/wingchun/src/broker/trader.cpp`
- Binance trader: `core/extensions/binance/src/trader_binance.cpp`
- Ledger service: `core/cpp/wingchun/src/service/ledger.cpp`
- Message definitions: `core/cpp/wingchun/include/kungfu/wingchun/msg.h`

**External Resources**:
- Binance API Docs: https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api
- WebSocket User Data: https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/User-Data-Streams

## Change History

- **2025-11-17**: Initial trading flow documentation created
