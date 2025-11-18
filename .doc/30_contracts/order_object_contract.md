---
title: Order Object Contract
updated_at: 2025-11-17
owner: core-dev
lang: en
tags: [contract, order, trading, message, data-structure]
code_refs:
  - core/cpp/wingchun/include/kungfu/wingchun/msg.h:666-730
  - core/cpp/wingchun/include/kungfu/wingchun/common.h:91-150
purpose: "Defines the canonical order representation and state machine across the trading system"
---

# Order Object Contract

## Purpose

The `Order` struct is the canonical representation of a trading order throughout its entire lifecycle in the system. It flows from strategy creation through exchange execution and back via callbacks. This contract defines the exact structure, valid states, and invariants that all system components must respect.

## Structure Definition

**Source:** [core/cpp/wingchun/include/kungfu/wingchun/msg.h:666-730](../../core/cpp/wingchun/include/kungfu/wingchun/msg.h)

### Fields

| Field | Type | Size | Constraints | Description |
|-------|------|------|-------------|-------------|
| `strategy_id` | `uint32_t` | 4 bytes | >0 | Strategy instance identifier |
| `order_id` | `uint64_t` | 8 bytes | >0 | Client-assigned order ID (strategy-local) |
| `ex_order_id` | `char[]` | 64 bytes | null-terminated | Exchange-assigned order ID (populated after ack) |
| `symbol` | `char[]` | 64 bytes | lowercase | Trading symbol (e.g., "btcusdt") |
| `instrument_type` | `InstrumentType` | 1 byte | enum | Spot, Futures, etc. |
| `exchange_id` | `char[]` | 64 bytes | uppercase | Exchange identifier (e.g., "BINANCE") |
| `account_id` | `char[]` | 64 bytes | any | Account identifier |
| `source_id` | `char[]` | 64 bytes | any | Source/gateway identifier |
| `price` | `double` | 8 bytes | >=0 | Limit price (0 for market orders) |
| `volume` | `double` | 8 bytes | >0 | Original order quantity |
| `volume_traded` | `double` | 8 bytes | >=0 | Cumulative filled quantity |
| `volume_left` | `double` | 8 bytes | >=0 | Remaining unfilled quantity |
| `stop_price` | `double` | 8 bytes | >=0 | Stop/trigger price |
| `close_pnl` | `double` | 8 bytes | any | Realized PnL (Binance-specific) |
| `avg_price` | `double` | 8 bytes | >=0 | Average fill price |
| `fee` | `double` | 8 bytes | >=0 | Total transaction fees |
| `fee_currency` | `char[]` | 64 bytes | any | Fee currency symbol |
| `status` | `OrderStatus` | 1 byte | enum | Current order state |
| `time_condition` | `TimeCondition` | 1 byte | enum | Time validity (IOC, GTC, etc.) |
| `side` | `Side` | 1 byte | enum | Buy or Sell |
| `position_side` | `Direction` | 1 byte | enum | Long or Short |
| `order_type` | `OrderType` | 1 byte | enum | Limit, Market, etc. |
| `error_code` | `char[]` | 64 bytes | any | Exchange error code (if rejected) |
| `insert_time` | `int64_t` | 8 bytes | nanoseconds | Order creation timestamp |
| `update_time` | `int64_t` | 8 bytes | nanoseconds | Last update timestamp |

**Total Size:** ~618 bytes (packed struct)

### Enum Values

#### Side
**Source:** [core/cpp/wingchun/include/kungfu/wingchun/common.h:91-99](../../core/cpp/wingchun/include/kungfu/wingchun/common.h)

```cpp
enum class Side : int8_t {
    Buy,        // 0 - Open long or close short
    Sell,       // 1 - Open short or close long
    Lock,       // 2 - Lock position
    Unlock,     // 3 - Unlock position
    Exec,       // 4 - Execute
    Drop        // 5 - Drop/cancel
};
```

#### OrderStatus
**Source:** [core/cpp/wingchun/include/kungfu/wingchun/common.h:139-150](../../core/cpp/wingchun/include/kungfu/wingchun/common.h)

```cpp
enum class OrderStatus : int8_t {
    Unknown,                    // 0 - Uninitialized
    Submitted,                  // 1 - Acknowledged by exchange
    Pending,                    // 2 - Created but not yet sent
    Cancelled,                  // 3 - Cancelled (user or exchange)
    Error,                      // 4 - Rejected or error
    Filled,                     // 5 - Completely filled
    PartialFilledNotActive,     // 6 - Partially filled, no longer active
    PartialFilledActive,        // 7 - Partially filled, still active
    PreSend                     // 8 - Pre-send validation state
};
```

#### Direction
**Source:** [core/cpp/wingchun/include/kungfu/wingchun/common.h:152-156](../../core/cpp/wingchun/include/kungfu/wingchun/common.h)

```cpp
enum class Direction : int8_t {
    Long,       // 0 - Long position
    Short       // 1 - Short position
};
```

## Invariants

### 1. ID Assignment
- `order_id` MUST be assigned before `insert_order()` returns to caller
- `order_id` is unique within a strategy instance (not globally unique)
- `ex_order_id` is EMPTY ("") until exchange acknowledgment
- `ex_order_id` is unique globally once assigned by exchange

### 2. Symbol Format
- `symbol` MUST be normalized to lowercase (e.g., "btcusdt", not "BTCUSDT")
- `exchange_id` MUST be uppercase (e.g., "BINANCE", not "binance")
- Both are null-terminated C strings

### 3. Volume Relationships
- `volume` > 0 (original quantity)
- `volume_traded` >= 0 (cumulative fills)
- `volume_left` >= 0 (remaining)
- **Invariant:** `volume = volume_traded + volume_left` (approximately, due to floating-point)

### 4. Price Constraints
- `price` >= 0 (0 for market orders)
- `stop_price` >= 0 (0 if not a stop order)
- `avg_price` >= 0 (computed from fills)
- For limit orders: `price` > 0

### 5. Timestamp Monotonicity
- `update_time` >= `insert_time` (always)
- Timestamps are nanoseconds since epoch (Unix time * 1e9)
- `insert_time` set when order created in strategy
- `update_time` updated on every status change

### 6. Fee Tracking
- `fee` accumulates with each fill
- `fee_currency` specifies denomination (e.g., "USDT", "BTC")
- `close_pnl` only populated for futures positions (Binance-specific)

## State Machine

### Valid State Transitions

```
                    ┌──────────────────────────────────────┐
                    │                                      │
                    ▼                                      │
           ┌────────────────┐                              │
    Start  │  Unknown (0)   │                              │
           └────────┬───────┘                              │
                    │                                      │
                    │ insert_order()                       │
                    ▼                                      │
           ┌────────────────┐                              │
           │  Pending (2)   │                              │
           └────────┬───────┘                              │
                    │                                      │
                    │ sent to exchange                     │
                    ▼                                      │
           ┌────────────────┐                              │
           │ Submitted (1)  │──────────┐                   │
           └────────┬───────┘          │                   │
                    │                  │                   │
      ┌─────────────┼──────────────────┤                   │
      │             │                  │                   │
      │  partial    │  full fill       │  cancel/reject    │
      │  fill       │                  │                   │
      ▼             ▼                  ▼                   │
┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│ Partial-     │ │  Filled (5)  │ │ Cancelled(3) │        │
│ FilledActive │ │  [TERMINAL]  │ │  [TERMINAL]  │        │
│     (7)      │ └──────────────┘ └──────────────┘        │
└──────┬───────┘                                           │
       │                                                   │
       │  cancel or                                        │
       │  filled                                           │
       │                                                   │
       ▼                                                   │
┌──────────────┐                                           │
│ Partial-     │                                           │
│ FilledNot-   │                                           │
│ Active (6)   │                                           │
│  [TERMINAL]  │                                           │
└──────────────┘                                           │
                                                           │
              Any state can transition to Error (4) ───────┘
              [TERMINAL]
```

### Terminal States
- `Filled` (5) - Order completely filled
- `Cancelled` (3) - Order cancelled before complete fill
- `PartialFilledNotActive` (6) - Partially filled, cancelled or expired
- `Error` (4) - Rejected or encountered error

Once in a terminal state, the order will not change status.

### Error Transitions
Any state can transition to `Error` if:
- Exchange rejects the order
- Network/system failure occurs
- Validation fails
- `error_code` field will contain exchange-specific error code

## Usage Examples

### C++ (Strategy Context)

```cpp
// core/cpp/wingchun/src/strategy/context.cpp:473
auto order_id = context->insert_order(
    symbol,          // "btcusdt"
    exchange_id,     // "BINANCE"
    source_id,       // "binance"
    account_id,      // "my_account"
    price,           // 50000.0 (limit price)
    volume,          // 0.1 (quantity)
    Side::Buy,       // Buy order
    Direction::Long, // Long position
    OrderType::Limit,// Limit order
    TimeCondition::GTC, // Good-til-cancel
    false            // reduce_only
);
// order_id is assigned immediately
// Order object written to journal with status=Pending
```

### Python (Strategy API)

```python
# strategies/demo_spot.py:45
from kungfu.wingchun.constants import *

order_id = context.insert_order(
    symbol="btcusdt",
    exchange_id="binance",
    source_id="binance",
    account_id=self.account,
    price=depth.ask_price[0] * 0.99,  # Below ask
    volume=100,                         # 100 units
    side=Side.Buy,
    offset=Offset.Open,
    hedge_flag=HedgeFlag.Speculation,
    price_type=PriceType.Limit
)

# Later, in on_order() callback:
def on_order(self, context, order):
    self.log.info(f"Order {order.order_id} status: {order.status}")
    self.log.info(f"Exchange ID: {order.ex_order_id}")
    self.log.info(f"Filled: {order.volume_traded}/{order.volume}")

    if order.status == OrderStatus.Filled:
        self.log.info(f"Avg price: {order.avg_price}, Fee: {order.fee} {order.fee_currency}")
```

### Order Status Lifecycle Example

```python
# Callback sequence for a successful order:

# 1. After insert_order() - written to journal
on_order(order):
    order.status = OrderStatus.Pending
    order.order_id = 12345  # Assigned
    order.ex_order_id = ""  # Not yet assigned

# 2. After exchange acknowledgment
on_order(order):
    order.status = OrderStatus.Submitted
    order.ex_order_id = "EX-67890"  # Now assigned
    order.volume_traded = 0
    order.volume_left = 0.1

# 3. First partial fill
on_order(order):
    order.status = OrderStatus.PartialFilledActive
    order.volume_traded = 0.04
    order.volume_left = 0.06
    order.avg_price = 50001.2
    order.fee += 2.00

# 4. Second fill completes order
on_order(order):
    order.status = OrderStatus.Filled  # Terminal
    order.volume_traded = 0.1
    order.volume_left = 0
    order.avg_price = 50002.5  # Updated
    order.fee += 3.00  # Cumulative
```

## Related Contracts

- [OrderInput Contract](order_input_contract.md) - Order submission message
- [Trade Object Contract](trade_object_contract.md) - Individual fill/execution
- [Strategy Context API](strategy_context_api.md) - `insert_order()` and `on_order()` methods

## Related Documentation

### Modules
- [Strategy Framework](../10_modules/strategy_framework.md) - How strategies use orders
- [Binance Extension](../10_modules/binance_extension.md) - Exchange-specific order mapping

### Flows
- [Trading Flow](../20_interactions/trading_flow.md) - End-to-end order lifecycle
- [Order Lifecycle Flow](../20_interactions/order_lifecycle_flow.md) - State transitions in detail

### Configuration
- [Config Usage Map](../40_config/config_usage_map.md) - Account and exchange configuration

## Implementation Notes

### Memory Layout
The struct uses `__attribute__((packed))` on Linux to ensure no padding:
```cpp
#ifndef _WIN32
} __attribute__((packed));
#else
};  // Windows allows padding
#endif
```

This ensures consistent binary serialization across network and journal storage.

### String Handling
All `char[]` fields use fixed-length arrays for performance. Helper methods like `get_symbol()` convert to `std::string`:

```cpp
const std::string get_symbol() const
{ return std::string(symbol); }

void set_symbol(const std::string& symbol)
{ strncpy(this->symbol, symbol.c_str(), SYMBOL_LEN); }
```

**Warning:** `strncpy` does NOT null-terminate if source exceeds buffer. Always use defined constants (`SYMBOL_LEN`, etc.) and validate input lengths.

### JSON Serialization
The system provides `to_json()` and `from_json()` for all message types:

```cpp
inline void to_json(nlohmann::json &j, const Order &order)
{
    j["order_id"] = order.order_id;
    j["ex_order_id"] = order.get_ex_order_id();
    j["symbol"] = order.get_symbol();
    j["status"] = order.status;
    // ... all fields
}
```

See [core/cpp/wingchun/include/kungfu/wingchun/msg.h:745-780](../../core/cpp/wingchun/include/kungfu/wingchun/msg.h) for complete serialization.

## Version History

- **2025-11-17:** Initial contract documentation based on code analysis
- **2025-03-03:** Modified from original kungfu (Keren Dong) by kx@godzilla.dev
