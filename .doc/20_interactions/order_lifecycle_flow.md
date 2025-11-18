---
title: Order Lifecycle Flow
updated_at: 2025-11-17
owner: core-dev
lang: en
tags: [interaction, flow, order, trading, execution]
dependencies:
  - strategy_framework
  - gateway_architecture
  - ledger_system
code_refs:
  - core/cpp/wingchun/include/kungfu/wingchun/msg.h:666-730
  - core/cpp/wingchun/src/strategy/context.cpp:150-250
purpose: "Documents the complete lifecycle of an order from insertion to final status"
tokens_estimate: 4800
---

# Order Lifecycle Flow

## Overview

This document traces the complete lifecycle of a trading order from the moment a strategy calls `insert_order()` through execution at the exchange and back to the strategy callback. Understanding this flow is essential for order management, error handling, and latency optimization.

## Order States

**Reference**: [Order Object Contract - State Machine](../30_contracts/order_object_contract.md#state-machine)

```
         ┌─────────┐
         │ Pending │ (Initial state, local only)
         └────┬────┘
              │
              ▼
         ┌───────────┐
         │ Submitted │ (Sent to exchange)
         └─────┬─────┘
               │
        ┌──────┼───────────┐
        ▼      ▼           ▼
   ┌────────┐ ┌───────────────────────┐ ┌──────────┐
   │ Error  │ │ PartialFilledActive   │ │  Filled  │
   └────────┘ └───────────┬───────────┘ └──────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │  Cancelled   │
                   └──────────────┘
```

## Complete Order Flow

### Phase 1: Order Insertion (Strategy → Ledger)

**Trigger**: Strategy calls `context.insert_order()`

**Python Code**:
```python
def on_depth(self, context, depth):
    if depth.bid_price[0] > self.buy_threshold:
        order_id = context.insert_order(
            symbol="btcusdt",
            inst_type=InstrumentType.Spot,
            exchange=Exchange.BINANCE,
            account="my_account",
            price=50000.0,
            volume=0.1,
            order_type=OrderType.Limit,
            side=Side.Buy
        )
        self.log.info(f"Inserted order {order_id}")
```

**Timeline**:
```
T=0μs:    Strategy calls insert_order()
T=1μs:    Python → C++ binding (pybind11)
T=2μs:    Context creates OrderInput struct
T=3μs:    Assign local order_id (strategy-scoped)
T=4μs:    Write OrderInput to journal
T=5μs:    Return order_id to strategy
```

**Data Structure** (OrderInput):
```cpp
struct OrderInput {
    uint64_t order_id;           // Assigned by strategy context
    uint32_t strategy_id;        // Current strategy ID
    char symbol[64];             // "btcusdt"
    InstrumentType instrument_type;  // Spot
    char exchange_id[64];        // "BINANCE"
    char account_id[64];         // "my_account"
    double price;                // 50000.0
    double volume;               // 0.1
    OrderType order_type;        // Limit
    Side side;                   // Buy
    // ... other fields
};
```

**Journal Write**:
```
Event Type: OrderInput
Destination: td_binance (account location)
Data: OrderInput struct (packed binary)
```

**State**: Order exists locally with `order_id`, but not yet at exchange

---

### Phase 2: Ledger Processing

**Timeline**:
```
T=5μs:    Ledger reads OrderInput from journal
T=10μs:   Check account balance
T=15μs:   Freeze required funds (0.1 BTC × 50000 = 5000 USDT)
T=20μs:   Create Order struct with status=Pending
T=25μs:   Write Order event to journal
T=30μs:   Forward OrderInput to trading gateway
```

**Ledger Actions**:

1. **Balance Check**:
```cpp
// Check if account has sufficient USDT
double required = price * volume;  // 5000 USDT
if (account.usdt_available < required) {
    // Reject order, write Order with status=Error
    return;
}
```

2. **Freeze Funds**:
```cpp
account.usdt_available -= required;  // 10000 → 5000
account.usdt_frozen += required;     // 0 → 5000
```

3. **Create Order Event**:
```cpp
Order order = {};
order.order_id = input.order_id;
order.strategy_id = input.strategy_id;
order.status = OrderStatus::Pending;
order.volume = input.volume;
order.volume_traded = 0.0;
order.volume_left = input.volume;
// ... copy from OrderInput

journal.write(msg::type::Order, order);
```

**First Callback**:
```python
def on_order(self, context, order):
    # order.status == OrderStatus.Pending
    # order.ex_order_id == ""  (not assigned yet)
    self.log.debug(f"Order {order.order_id} pending")
```

---

### Phase 3: Trading Gateway Submission

**Timeline**:
```
T=30μs:   TD gateway reads OrderInput from journal
T=50μs:   Build exchange API request (REST)
T=100μs:  Send HTTP POST to Binance API
```

**Gateway Actions**:

1. **Read OrderInput**:
```cpp
void Trader::on_order_input(event_ptr event) {
    auto input = event->data<OrderInput>();
    insert_order(event);  // Virtual method, implemented by BinanceTrader
}
```

2. **Build API Request** (Binance example):
```cpp
// POST /api/v3/order
{
    "symbol": "BTCUSDT",
    "side": "BUY",
    "type": "LIMIT",
    "timeInForce": "GTC",
    "quantity": "0.1",
    "price": "50000.0",
    "newClientOrderId": "strategy_123_order_456"  // Maps to local order_id
}
```

3. **Send to Exchange**:
```
HTTP POST https://testnet.binance.vision/api/v3/order
Headers: X-MBX-APIKEY, signature
```

**Latency**: Binance testnet ~20-50ms, mainnet ~5-15ms

---

### Phase 4: Exchange Acknowledgment

**Timeline**:
```
T=100μs + 20ms:   Binance processes order
T=100μs + 22ms:   Binance returns order ID
T=100μs + 23ms:   Gateway receives response
T=100μs + 24ms:   Gateway writes Order event
```

**Exchange Response**:
```json
{
    "orderId": 28457,                    // Exchange-assigned ID
    "clientOrderId": "strategy_123_order_456",
    "symbol": "BTCUSDT",
    "status": "NEW",                     // Binance status
    "price": "50000.0",
    "origQty": "0.1",
    "executedQty": "0.0",
    "transactTime": 1699564800000
}
```

**Gateway Mapping**:
```cpp
Order order = {};
order.order_id = input.order_id;  // From newClientOrderId
order.ex_order_id = "28457";      // From orderId
order.status = OrderStatus::Submitted;  // Map "NEW" → Submitted
order.insert_time = response.transactTime * 1e6;  // ms → ns
// ... update other fields

journal.write(msg::type::Order, order);
```

**Second Callback**:
```python
def on_order(self, context, order):
    # order.status == OrderStatus::Submitted
    # order.ex_order_id == "28457"
    self.log.info(f"Order {order.order_id} submitted to exchange as {order.ex_order_id}")
```

**State**: Order now exists at exchange, waiting for fills

---

### Phase 5: Partial Fill (WebSocket)

**Timeline**:
```
T=100μs + 50ms:   Market moves, order partially filled
T=100μs + 52ms:   Binance sends executionReport via WebSocket
T=100μs + 53ms:   MD gateway receives WebSocket message
T=100μs + 54ms:   Gateway writes Order + MyTrade events
```

**WebSocket Message** (Binance):
```json
{
    "e": "executionReport",
    "E": 1699564850000,
    "s": "BTCUSDT",
    "c": "strategy_123_order_456",
    "S": "BUY",
    "o": "LIMIT",
    "q": "0.1",
    "p": "50000.0",
    "X": "PARTIALLY_FILLED",        // Order status
    "i": 28457,
    "l": "0.04",                    // Last filled quantity
    "z": "0.04",                    // Cumulative filled quantity
    "L": "49995.0",                 // Last filled price
    "n": "0.5",                     // Commission
    "N": "USDT"                     // Commission asset
}
```

**Gateway Processing**:

1. **Update Order**:
```cpp
Order order = {};
order.order_id = get_local_order_id(msg.clientOrderId);
order.ex_order_id = "28457";
order.status = OrderStatus::PartialFilledActive;
order.volume = 0.1;
order.volume_traded = 0.04;         // Cumulative
order.volume_left = 0.06;           // Remaining
order.avg_price = 49995.0;          // Weighted average
order.update_time = msg.E * 1e6;

journal.write(msg::type::Order, order);
```

2. **Create Trade**:
```cpp
MyTrade trade = {};
trade.order_id = order.order_id;
trade.strategy_id = order.strategy_id;
trade.trade_id = generate_trade_id();
trade.symbol = "btcusdt";
trade.side = Side::Buy;
trade.price = 49995.0;               // This fill's price
trade.volume = 0.04;                 // This fill's quantity
trade.fee = 0.5;
trade.fee_currency = "USDT";
trade.trade_time = msg.E * 1e6;

journal.write(msg::type::MyTrade, trade);
```

**Third Callback** (Order):
```python
def on_order(self, context, order):
    # order.status == OrderStatus::PartialFilledActive
    # order.volume_traded == 0.04
    # order.volume_left == 0.06
    self.log.info(f"Order {order.order_id} partially filled: {order.volume_traded}/{order.volume}")
```

**Fourth Callback** (Trade):
```python
def on_transaction(self, context, trade):
    # trade.volume == 0.04
    # trade.price == 49995.0
    self.total_volume += trade.volume
    self.total_cost += trade.price * trade.volume
    self.log.info(f"Trade executed: {trade.volume} @ {trade.price}")
```

---

### Phase 6: Full Fill

**Timeline**:
```
T=100μs + 100ms:  Remaining 0.06 BTC fills
T=100μs + 102ms:  WebSocket executionReport received
T=100μs + 103ms:  Gateway writes final Order + MyTrade
```

**WebSocket Message**:
```json
{
    "X": "FILLED",                  // Order fully filled
    "z": "0.1",                     // Total filled = original quantity
    "l": "0.06",                    // Last fill quantity
    "L": "50005.0"                  // Last fill price
}
```

**Gateway Processing**:

1. **Final Order Update**:
```cpp
order.status = OrderStatus::Filled;
order.volume_traded = 0.1;
order.volume_left = 0.0;
order.avg_price = (49995.0 * 0.04 + 50005.0 * 0.06) / 0.1;  // 50001.0

journal.write(msg::type::Order, order);
```

2. **Second Trade Event**:
```cpp
trade.volume = 0.06;
trade.price = 50005.0;

journal.write(msg::type::MyTrade, trade);
```

**Fifth Callback** (Order):
```python
def on_order(self, context, order):
    # order.status == OrderStatus::Filled
    # order.volume_traded == 0.1
    # order.volume_left == 0.0
    self.log.info(f"Order {order.order_id} FILLED at avg {order.avg_price}")

    # Clean up local tracking
    del self.open_orders[order.order_id]
```

**Sixth Callback** (Trade):
```python
def on_transaction(self, context, trade):
    # trade.volume == 0.06
    # trade.price == 50005.0
    self.total_volume += trade.volume  # Now == 0.1
```

---

### Phase 7: Ledger Position Update

**Timeline**:
```
T=100μs + 103ms:  Ledger reads MyTrade event
T=100μs + 104ms:  Update position and asset
T=100μs + 105ms:  Write Position and Asset events
```

**Ledger Actions**:

1. **Unfreeze Funds**:
```cpp
account.usdt_frozen -= (order.price * order.volume);  // 5000 → 0
```

2. **Deduct Actual Cost**:
```cpp
double total_cost = order.avg_price * order.volume + order.fee;  // 50001.0 * 0.1 + 0.5 = 5000.6
account.usdt_available -= total_cost;  // (Already unfrozen, so net decrease)
```

3. **Update Position**:
```cpp
position.symbol = "btcusdt";
position.volume += order.volume;       // 0 → 0.1 BTC
position.avg_open_price = order.avg_price;  // 50001.0
position.realized_pnl = 0.0;           // No close yet
position.unrealized_pnl = 0.0;         // No price movement yet

journal.write(msg::type::Position, position);
```

**Seventh Callback** (Position):
```python
def on_position(self, context, position):
    # position.symbol == "btcusdt"
    # position.volume == 0.1
    # position.avg_open_price == 50001.0
    self.log.info(f"Position updated: {position.volume} BTC @ {position.avg_open_price}")
```

---

## Error Scenarios

### Scenario 1: Insufficient Balance

**Timeline**:
```
T=0μs:    Strategy calls insert_order(volume=100 BTC)
T=5μs:    Ledger reads OrderInput
T=10μs:   Balance check fails (only 5000 USDT available)
T=15μs:   Ledger writes Order with status=Error
```

**Order Event**:
```cpp
order.status = OrderStatus::Error;
order.error_code = "INSUFFICIENT_BALANCE";
order.volume_traded = 0.0;
order.volume_left = 0.0;
```

**Callback**:
```python
def on_order(self, context, order):
    if order.status == OrderStatus::Error:
        self.log.error(f"Order {order.order_id} rejected: {order.error_code}")
```

**No funds frozen**, order never sent to exchange.

---

### Scenario 2: Exchange Rejection

**Timeline**:
```
T=0-30μs:     Order passes ledger checks, sent to exchange
T=100μs+20ms: Exchange returns error (e.g., LOT_SIZE violation)
T=100μs+22ms: Gateway writes Order with status=Error
```

**Exchange Response**:
```json
{
    "code": -1013,
    "msg": "Filter failure: LOT_SIZE"
}
```

**Gateway Processing**:
```cpp
order.status = OrderStatus::Error;
order.error_code = "-1013: Filter failure: LOT_SIZE";
order.ex_order_id = "";  // Never assigned

journal.write(msg::type::Order, order);
```

**Ledger Action**: Unfreeze funds (order was never submitted)

**Callback**:
```python
def on_order(self, context, order):
    if order.status == OrderStatus::Error:
        # Parse error and retry with corrected parameters
        if "LOT_SIZE" in order.error_code:
            self.log.error("Invalid order size, check exchange filters")
```

---

### Scenario 3: Order Cancelled (User Request)

**Trigger**: Strategy calls `context.cancel_order()`

**Python Code**:
```python
def on_timer(self, context):
    # Cancel order if open for >60 seconds
    for order_id in self.open_orders:
        context.cancel_order("my_account", order_id, "", "", InstrumentType.Spot)
```

**Timeline**:
```
T=0μs:     Strategy calls cancel_order()
T=5μs:     Write OrderAction to journal
T=30μs:    Gateway reads OrderAction
T=50μs:    Gateway sends DELETE /api/v3/order to Binance
T=50μs+20ms: Exchange confirms cancellation
T=50μs+22ms: Gateway writes Order with status=Cancelled
```

**Exchange Response**:
```json
{
    "orderId": 28457,
    "status": "CANCELED",
    "executedQty": "0.04"  // Partial fill before cancel
}
```

**Order Event**:
```cpp
order.status = OrderStatus::Cancelled;
order.volume_traded = 0.04;  // What filled before cancel
order.volume_left = 0.06;    // What was cancelled
```

**Ledger Action**: Unfreeze remaining `0.06 * 50000 = 3000 USDT`

**Callback**:
```python
def on_order(self, context, order):
    if order.status == OrderStatus::Cancelled:
        self.log.info(f"Order {order.order_id} cancelled, filled {order.volume_traded}/{order.volume}")
```

---

## Latency Breakdown

### Typical Order Execution (Testnet)

| Phase | Time | Cumulative |
|-------|------|------------|
| Strategy → Journal | 5 μs | 5 μs |
| Ledger processing | 25 μs | 30 μs |
| Gateway processing | 20 μs | 50 μs |
| Network (testnet) | 20 ms | ~20 ms |
| Exchange processing | 2 ms | ~22 ms |
| Network return | 20 ms | ~42 ms |
| Gateway → Journal | 2 μs | ~42 ms |
| **Strategy callback** | **~42 ms** | **Total** |

### Production (Mainnet, Co-located)

| Phase | Time | Cumulative |
|-------|------|------------|
| Strategy → Journal | 5 μs | 5 μs |
| Ledger processing | 25 μs | 30 μs |
| Gateway processing | 20 μs | 50 μs |
| Network (co-located) | 0.5 ms | ~0.55 ms |
| Exchange processing | 1 ms | ~1.55 ms |
| Network return | 0.5 ms | ~2.05 ms |
| Gateway → Journal | 2 μs | ~2.05 ms |
| **Strategy callback** | **~2.1 ms** | **Total** |

**Optimization**: Co-location reduces network latency from 40ms → 1ms (20x improvement)

---

## Multi-Order Scenarios

### Scenario: Market Making (Simultaneous Buy/Sell)

```python
def on_depth(self, context, depth):
    # Place bid and ask simultaneously
    bid_id = context.insert_order(
        symbol="btcusdt",
        side=Side.Buy,
        price=depth.bid_price[0] + 0.01,  # Improve best bid
        volume=0.1,
        # ...
    )

    ask_id = context.insert_order(
        symbol="btcusdt",
        side=Side.Sell,
        price=depth.ask_price[0] - 0.01,  # Improve best ask
        volume=0.1,
        # ...
    )

    self.open_orders[bid_id] = "bid"
    self.open_orders[ask_id] = "ask"
```

**Flow**:
1. Both orders written to journal (sequential, ~10 μs total)
2. Ledger processes bid, freezes 5000 USDT
3. Ledger processes ask, freezes 0.1 BTC
4. Gateway sends both to exchange (HTTP pipelining possible)
5. Both orders acknowledged independently
6. Callbacks fire as each order fills

**Critical**: Track orders separately, one filling doesn't affect the other

---

## Related Documentation

### Contracts
- [Order Object Contract](../30_contracts/order_object_contract.md) - Complete order structure
- [Strategy Context API](../30_contracts/strategy_context_api.md) - Order management methods

### Modules
- [Strategy Framework](../10_modules/strategy_framework.md) - Strategy development
- [Gateway Architecture](../10_modules/gateway_architecture.md) - Exchange connectivity
- [Ledger System](../10_modules/ledger_system.md) - Position tracking
- [Yijinjing Journal](../10_modules/yijinjing_journal.md) - Event sourcing

### Interactions
- [Strategy Lifecycle Flow](strategy_lifecycle_flow.md) - Strategy execution model

### Operations
- [Debugging Guide](../90_operations/DEBUGGING.md) - Troubleshooting order issues

## Version History

- **2025-11-17**: Initial order lifecycle flow documentation
