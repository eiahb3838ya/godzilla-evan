---
title: Python Bindings (pybind11)
updated_at: 2025-11-17
owner: core-dev
lang: en
tags: [module, pybind11, python, cpp, bindings, interop]
dependencies:
  - yijinjing_journal
  - strategy_framework
  - gateway_architecture
code_refs:
  - core/cpp/wingchun/pybind/pybind_wingchun.cpp:1-950
  - core/cpp/yijinjing/pybind/pybind_yjj.cpp:1-300
  - core/extensions/binance/src/pybind_binance.cpp:1-150
  - core/deps/pybind11-2.10.4/include/pybind11/pybind11.h:1-2500
purpose: "Exposes C++ trading framework to Python using pybind11 for strategy development"
tokens_estimate: 5200
---

# Python Bindings (pybind11)

## Overview

The Python bindings module uses **pybind11** to expose the C++ trading framework to Python, enabling strategy development in Python while leveraging the performance and low-latency characteristics of the C++ core. This allows developers to write strategies in Python's friendly syntax while benefiting from microsecond-level event processing in C++.

## Architecture

### Binding Layers

```
┌─────────────────────────────────────────────────────┐
│        Python Strategy (strategies/demo_spot.py)     │
│  • User-written callbacks (on_depth, on_order, etc.) │
│  • Business logic in Python                          │
└─────────────────┬───────────────────────────────────┘
                  │ Python API
                  ▼
┌─────────────────────────────────────────────────────┐
│         pybind11 Bindings (Thin Wrapper)            │
│  • Type conversions (C++ ↔ Python)                   │
│  • Enum exports (Side, OrderStatus, etc.)           │
│  • Struct bindings (Order, Depth, Trade)            │
│  • Method bindings (insert_order, subscribe, etc.)  │
└─────────────────┬───────────────────────────────────┘
                  │ Native C++ Calls
                  ▼
┌─────────────────────────────────────────────────────┐
│        C++ Core Framework (wingchun/yijinjing)      │
│  • Strategy::Context (order management)             │
│  • Journal (event sourcing)                         │
│  • Gateway (exchange connections)                   │
└─────────────────────────────────────────────────────┘
```

### pybind11 Version

**Version**: 2.10.4 (vendored in [core/deps/pybind11-2.10.4/](../../core/deps/pybind11-2.10.4/))

**Rationale**: Vendored to ensure reproducible builds across environments. Version 2.10.4 is stable and supports Python 3.7-3.12.

## Key Binding Files

### 1. Wingchun Bindings

**File**: [core/cpp/wingchun/pybind/pybind_wingchun.cpp:1-950](../../core/cpp/wingchun/pybind/pybind_wingchun.cpp)

**Purpose**: Binds trading framework (orders, market data, strategies, ledger)

**Modules Exported**:
- `kungfu.wingchun` - Main module
- `kungfu.wingchun.constants` - Enums (Side, OrderStatus, Direction, etc.)

**Key Bindings**:

#### Enumerations (Constants)
```cpp
// Lines 264-277: Side enum
py::enum_<kungfu::wingchun::Side>(m_constants, "Side", py::arithmetic())
    .value("Buy", kungfu::wingchun::Side::Buy)
    .value("Sell", kungfu::wingchun::Side::Sell)
    .value("Lock", kungfu::wingchun::Side::Lock)
    .value("Unlock", kungfu::wingchun::Side::Unlock)
    .value("Exec", kungfu::wingchun::Side::Exec)
    .value("Drop", kungfu::wingchun::Side::Drop)
    .export_values()
    .def("__eq__", [](const kungfu::wingchun::Side &a, int b) {
        return static_cast<int>(a) == b;
    });

// Lines 303-319: OrderStatus enum
py::enum_<kungfu::wingchun::OrderStatus>(m_constants, "OrderStatus", py::arithmetic())
    .value("Unknown", kungfu::wingchun::OrderStatus::Unknown)
    .value("Submitted", kungfu::wingchun::OrderStatus::Submitted)
    .value("Pending", kungfu::wingchun::OrderStatus::Pending)
    .value("Cancelled", kungfu::wingchun::OrderStatus::Cancelled)
    .value("Error", kungfu::wingchun::OrderStatus::Error)
    .value("Filled", kungfu::wingchun::OrderStatus::Filled)
    .value("PartialFilledNotActive", kungfu::wingchun::OrderStatus::PartialFilledNotActive)
    .value("PartialFilledActive", kungfu::wingchun::OrderStatus::PartialFilledActive)
    .value("PreSend", kungfu::wingchun::OrderStatus::PreSend)
    .export_values();
```

**Usage in Python**:
```python
from kungfu.wingchun.constants import Side, OrderStatus, Direction

# Use enums directly
order_id = context.insert_order(
    symbol="btcusdt",
    side=Side.Buy,  # Enum value
    # ...
)

# Compare enum values
if order.status == OrderStatus.Filled:
    print("Order filled!")
```

#### Data Structures (Structs)
```cpp
// Lines 516-547: Order struct binding
py::class_<Order>(m, "Order")
    .def(py::init<>())
    .def_readwrite("strategy_id", &Order::strategy_id)
    .def_readwrite("order_id", &Order::order_id)
    .def_readwrite("insert_time", &Order::insert_time)
    .def_readwrite("update_time", &Order::update_time)
    .def_readwrite("price", &Order::price)
    .def_readwrite("volume", &Order::volume)
    .def_readwrite("volume_traded", &Order::volume_traded)
    .def_readwrite("volume_left", &Order::volume_left)
    .def_readwrite("status", &Order::status)
    .def_readwrite("side", &Order::side)
    .def_property("symbol", &Order::get_symbol, &Order::set_symbol)
    .def_property("ex_order_id", &Order::get_ex_order_id, &Order::set_ex_order_id)
    .def_property("exchange_id", &Order::get_exchange_id, &Order::set_exchange_id)
    .def_property_readonly("active", [](const Order& o) {
        return not is_final_status(o.status);
    })
    .def("__repr__", [](const Order &a) { return to_string(a); });
```

**Usage in Python**:
```python
def on_order(self, context, order):
    # Access fields directly
    print(f"Order {order.order_id}")
    print(f"Status: {order.status}")
    print(f"Symbol: {order.symbol}")  # Property (calls get_symbol())
    print(f"Filled: {order.volume_traded}/{order.volume}")
    print(f"Active: {order.active}")  # Computed property
```

#### Strategy Context
```cpp
// Lines 719-743: Context class binding
py::class_<strategy::Context, std::shared_ptr<strategy::Context>>(m, "Context")
    .def("now", &strategy::Context::now)
    .def("add_timer", &strategy::Context::add_timer)
    .def("add_time_interval", &strategy::Context::add_time_interval)
    .def("add_account", &strategy::Context::add_account)
    .def("list_accounts", &strategy::Context::list_accounts)
    .def("subscribe", &strategy::Context::subscribe)
    .def("unsubscribe", &strategy::Context::unsubscribe)
    .def("subscribe_trade", &strategy::Context::subscribe_trade)
    .def("subscribe_ticker", &strategy::Context::subscribe_ticker)
    .def("insert_order", &strategy::Context::insert_order)
    .def("cancel_order", &strategy::Context::cancel_order)
    .def("query_order", &strategy::Context::query_order,
         py::arg("account"), py::arg("order_id"), py::arg("ex_order_id"),
         py::arg("inst_type"), py::arg("symbol")="");  // Default argument
```

**Usage in Python**:
```python
def pre_start(self, context):
    # All methods directly available
    context.add_account("binance", "my_account")
    context.subscribe("binance", ["btcusdt"], InstrumentType.Spot, Exchange.BINANCE)

def on_depth(self, context, depth):
    order_id = context.insert_order(
        symbol=depth.symbol,
        # ... other args
    )
```

### 2. Yijinjing Bindings

**File**: [core/cpp/yijinjing/pybind/pybind_yjj.cpp:1-300](../../core/cpp/yijinjing/pybind/pybind_yjj.cpp)

**Purpose**: Binds journal system (event sourcing, message passing)

**Modules Exported**:
- `kungfu.yijinjing` - Journal API
- `kungfu.yijinjing.data` - Data structures (locator, journal, reader, writer)

**Key Bindings**:
- `data::locator` - File path management
- `journal::reader` - Journal reading
- `journal::writer` - Journal writing
- `event` - Event object
- `msg::type` - Message type enums

### 3. Extension Bindings (Binance Example)

**File**: [core/extensions/binance/src/pybind_binance.cpp:1-150](../../core/extensions/binance/src/pybind_binance.cpp)

**Purpose**: Binds Binance-specific gateway implementation (for testing/introspection)

**Module Exported**: `kungfu.extensions.binance`

**Typical Pattern**:
```cpp
py::class_<MarketDataBinance, MarketData, std::shared_ptr<MarketDataBinance>>(m, "MarketDataBinance")
    .def(py::init<>())
    .def("subscribe", &MarketDataBinance::subscribe)
    .def("on_start", &MarketDataBinance::on_start);
```

## Type Conversions

### C++ → Python (Automatic)

pybind11 provides automatic conversions for common types:

| C++ Type | Python Type | Example |
|----------|-------------|---------|
| `int`, `int32_t`, `int64_t` | `int` | `order.order_id` → `12345` |
| `uint32_t`, `uint64_t` | `int` | `order.strategy_id` → `42` |
| `double`, `float` | `float` | `order.price` → `50000.5` |
| `bool` | `bool` | `order.active` → `True` |
| `std::string` | `str` | `order.get_symbol()` → `"btcusdt"` |
| `char[64]` (via property) | `str` | `order.symbol` → `"btcusdt"` |
| `std::vector<T>` | `list` | `depth.bid_price` → `[50000.0, 49999.0, ...]` |
| `enum class` | `int` (with named constants) | `Side.Buy` → `0` |

### Python → C++ (Automatic)

When calling C++ methods from Python:

```python
# Python call
context.insert_order(
    symbol="btcusdt",        # str → const std::string&
    inst_type=InstrumentType.Spot,  # enum → InstrumentType
    exchange=Exchange.BINANCE,      # enum → const std::string&
    account="my_account",    # str → const std::string&
    price=50000.0,           # float → double
    volume=0.1,              # float → double
    order_type=OrderType.Limit,     # enum → OrderType
    side=Side.Buy            # enum → Side
)
```

pybind11 automatically:
1. Checks argument types
2. Converts Python types to C++ equivalents
3. Raises `TypeError` if conversion fails

### Custom Conversions (char[] fields)

Fixed-length C string fields require helper methods:

```cpp
// C++ struct
struct Order {
    char symbol[64];
    const std::string get_symbol() const { return std::string(symbol); }
    void set_symbol(const std::string& s) { strncpy(symbol, s.c_str(), 64); }
};

// Binding (line 537)
.def_property("symbol", &Order::get_symbol, &Order::set_symbol)
```

**Python usage**:
```python
# Read (calls get_symbol())
symbol = order.symbol  # "btcusdt"

# Write (calls set_symbol())
order.symbol = "ethusdt"
```

## Trampoline Classes (Virtual Method Overrides)

To allow Python classes to override C++ virtual methods, pybind11 uses "trampoline classes":

### MarketData Trampoline

```cpp
// Lines 35-53: PyMarketData trampoline
class PyMarketData : public MarketData {
public:
    using MarketData::MarketData;

    // Override pure virtual methods
    bool subscribe(const std::vector<Instrument> &instruments) override {
        PYBIND11_OVERLOAD_PURE(bool, MarketData, subscribe, instruments);
    }

    bool unsubscribe(const std::vector<Instrument> &instruments) override {
        PYBIND11_OVERLOAD_PURE(bool, MarketData, unsubscribe, instruments);
    }

    void on_start() override {
        PYBIND11_OVERLOAD(void, MarketData, on_start, );
    }
};
```

**Usage in Python**:
```python
from kungfu.wingchun import MarketData

class MyMarketData(MarketData):
    """Custom market data gateway"""

    def subscribe(self, instruments):
        """Override C++ virtual method in Python"""
        for inst in instruments:
            print(f"Subscribing to {inst.symbol}")
        return True

    def on_start(self):
        """Override lifecycle callback"""
        self.log.info("Market data gateway starting")
```

### Trader Trampoline

```cpp
// Lines 55-75: PyTrader trampoline
class PyTrader : public Trader {
public:
    using Trader::Trader;

    AccountType get_account_type() const override {
        PYBIND11_OVERLOAD_PURE(const AccountType, Trader, get_account_type,);
    }

    bool insert_order(const event_ptr &event) override {
        PYBIND11_OVERLOAD_PURE(bool, Trader, insert_order, event);
    }

    bool cancel_order(const event_ptr &event) override {
        PYBIND11_OVERLOAD_PURE(bool, Trader, cancel_order, event);
    }
};
```

This pattern allows custom trading gateways to be implemented in Python.

## Performance Considerations

### Zero-Copy for Structs

For performance-critical data structures, pybind11 can expose raw memory addresses:

```cpp
// Lines 511-512, 544-545: Raw memory access
.def_property_readonly("raw_address", [](const Order &a) {
    return reinterpret_cast<uintptr_t>(&a);
})
.def("from_raw_address", [](uintptr_t addr) {
    return * reinterpret_cast<Order*>(addr);
})
```

This allows zero-copy access to C++ structs from Python (advanced use case, rarely needed).

### Callback Overhead

**Python Callback Cost**:
- C++ → Python callback: ~5-10 microseconds (Python interpreter + GIL)
- Pure C++ function call: ~50 nanoseconds

**Implication**: Strategy callbacks (on_depth, on_order) run ~100x slower than C++ but are still fast enough for most trading strategies (sub-millisecond response).

**Optimization**: Keep Python callbacks lightweight. Move heavy computation to C++ if needed.

### GIL (Global Interpreter Lock)

pybind11 automatically manages the GIL:
- **C++ → Python**: Acquires GIL before callback
- **Python → C++**: Releases GIL during C++ execution

**Important**: Long-running Python callbacks block other Python code in the same process. Use multi-process strategies (multiple strategy instances) for parallelism.

## Module Loading

### Import Path

```python
# Standard imports for strategy development
from kungfu.wingchun.constants import *  # Enums
from kungfu.wingchun import Context      # Strategy context (rarely used directly)
from kungfu.yijinjing import data        # Journal data structures
```

**Note**: Strategy developers rarely import these directly. The strategy runner pre-imports and injects them into the strategy module namespace.

### Shared Library

**Build Output**: `libpykungfu.so` (Linux) or `pykungfu.pyd` (Windows)

**Location**: `build/kfc/python/kungfu/wingchun/libpykungfu.so`

**Loading**: Python's `import kungfu.wingchun` triggers dynamic loading of the shared library.

## Debugging Bindings

### Type Inspection

```python
# Check binding type
import kungfu.wingchun as wc
print(type(wc.Order))  # <class 'pybind11_builtins.pybind11_type'>

# Check method availability
print(dir(wc.Context))  # ['add_account', 'insert_order', ...]

# Check enum values
from kungfu.wingchun.constants import OrderStatus
print(OrderStatus.Filled)  # 5 (underlying int value)
```

### Error Messages

**Common Binding Errors**:

1. **Type Mismatch**:
```python
context.insert_order(symbol=123)  # TypeError: expected str, got int
```

2. **Missing Required Argument**:
```python
context.insert_order()  # TypeError: missing required argument 'symbol'
```

3. **Invalid Enum Value**:
```python
order.status = 999  # No error, but unexpected behavior (validate in C++)
```

### Binding Verification

```bash
# Check if module loads
python -c "import kungfu.wingchun; print('OK')"

# Check for missing symbols
python -c "from kungfu.wingchun.constants import Side, OrderStatus; print(Side.Buy, OrderStatus.Filled)"
```

## Extending Bindings

### Adding New Enums

**C++ (common.h)**:
```cpp
enum class NewEnum : int8_t {
    ValueA,
    ValueB
};
```

**Binding (pybind_wingchun.cpp)**:
```cpp
py::enum_<kungfu::wingchun::NewEnum>(m_constants, "NewEnum", py::arithmetic())
    .value("ValueA", kungfu::wingchun::NewEnum::ValueA)
    .value("ValueB", kungfu::wingchun::NewEnum::ValueB)
    .export_values();
```

**Python Usage**:
```python
from kungfu.wingchun.constants import NewEnum
print(NewEnum.ValueA)
```

### Adding New Structs

**C++ (msg.h)**:
```cpp
struct NewData {
    int64_t timestamp;
    double value;
    char symbol[64];

    const std::string get_symbol() const { return std::string(symbol); }
};
```

**Binding (pybind_wingchun.cpp)**:
```cpp
py::class_<NewData>(m, "NewData")
    .def(py::init<>())
    .def_readwrite("timestamp", &NewData::timestamp)
    .def_readwrite("value", &NewData::value)
    .def_property("symbol", &NewData::get_symbol, &NewData::set_symbol);
```

### Adding New Methods to Context

**C++ (context.h)**:
```cpp
class Context {
    // ...
    void new_method(const std::string& arg);
};
```

**Binding (pybind_wingchun.cpp)**:
```cpp
py::class_<strategy::Context, std::shared_ptr<strategy::Context>>(m, "Context")
    // ... existing bindings
    .def("new_method", &strategy::Context::new_method);
```

**Python Usage**:
```python
context.new_method("test")
```

## Related Documentation

### Contracts
- [Strategy Context API](../30_contracts/strategy_context_api.md) - Python API specification
- [Order Object Contract](../30_contracts/order_object_contract.md) - Order struct binding
- [Depth Object Contract](../30_contracts/depth_object_contract.md) - Depth struct binding

### Modules
- [Strategy Framework](strategy_framework.md) - How strategies use Python API
- [Yijinjing Journal](yijinjing_journal.md) - Underlying event system

### External Resources
- [pybind11 Documentation](https://pybind11.readthedocs.io/) - Official pybind11 docs
- [pybind11 Advanced Topics](https://pybind11.readthedocs.io/en/stable/advanced/) - Custom converters, GIL management

## Common Patterns

### Enum Usage

```python
from kungfu.wingchun.constants import Side, OrderStatus, Direction

# Creating orders
context.insert_order(
    side=Side.Buy,
    direction=Direction.Long,
    # ...
)

# Checking order status
if order.status == OrderStatus.Filled:
    print("Filled!")
elif order.status == OrderStatus.Cancelled:
    print("Cancelled")
```

### Struct Access

```python
def on_order(self, context, order):
    # Direct field access (readwrite)
    print(order.order_id)       # uint64_t
    print(order.volume)         # double
    print(order.status)         # OrderStatus enum

    # Property access (char[] fields)
    print(order.symbol)         # Calls get_symbol()
    print(order.exchange_id)    # Calls get_exchange_id()

    # Computed properties
    print(order.active)         # not is_final_status(order.status)
```

### Vector/Array Access

```python
def on_depth(self, context, depth):
    # C++ std::vector → Python list (automatic)
    bid_prices = depth.bid_price  # Returns list[float]

    # Iterate like Python list
    for i, price in enumerate(depth.bid_price):
        if price > 0:  # Non-zero level
            volume = depth.bid_volume[i]
            print(f"Bid {i}: {price} x {volume}")
```

## Version History

- **2025-11-17**: Initial Python bindings documentation
- **2025-03-03**: Modified pybind11 integration by kx@godzilla.dev
