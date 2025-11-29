---
title: System Architecture
updated_at: 2025-11-17
owner: core-dev
lang: en
tokens_estimate: 2700
layer: 00_index
tags: [architecture, yijinjing, wingchun, design, event-sourcing]
purpose: "Technical architecture based on actual code analysis"
---

# System Architecture

Technical architecture based on actual code analysis (2025-10-23).

## Overview

High-frequency trading framework with two subsystems:

- **yijinjing** (易筋經): Event sourcing infrastructure (7.3K lines C++)
- **wingchun** (詠春): Trading abstraction layer (5.8K lines C++)

**Technology Stack**:
- C++17 (core)
- Python 3.8+ (strategies, bindings via pybind11)
- RxCPP 4.1.0 (reactive programming)
- nanomsg 1.1.5 (IPC)
- SQLite (metadata storage)

**Performance Profile**:
- Time precision: nanosecond (int64_t)
- Event write: <1μs (memory-mapped)
- Throughput: 1M+ events/sec
- Zero-copy reads

## Architecture Layers

```
┌─────────────────────────────────────┐
│  Python Strategies (strategies/)    │
│  - User trading logic                │
│  - Strategy callbacks                │
└──────────────┬──────────────────────┘
               │ Python API (pybind11)
┌──────────────▼──────────────────────┐
│  wingchun (Trading Layer)           │
│  - Broker (order routing)           │
│  - Gateway (exchange connectors)    │
│  - Strategy context                 │
└──────────────┬──────────────────────┘
               │ Event API
┌──────────────▼──────────────────────┐
│  yijinjing (Event Infrastructure)   │
│  - Journal (persistent log)         │
│  - I/O device (read/write)          │
│  - Time system (nanosecond)         │
└──────────────┬──────────────────────┘
               │ Memory-mapped files
┌──────────────▼──────────────────────┐
│  OS / Hardware                      │
└─────────────────────────────────────┘
```

## yijinjing (易筋經) - Event Sourcing

**Location**: `core/cpp/yijinjing/`

### Three-Layer Storage Design

```
Journal (API layer)
   ↓
Page (1-128 MB memory-mapped file)
   ↓
Frame (48-byte header + data)
```

### Frame Structure

Source: `include/kungfu/yijinjing/journal/frame.h`

```cpp
struct frame_header {  // 48 bytes, packed
    uint32_t length;         // Total frame length
    uint32_t header_length;  // sizeof(frame_header) = 48
    int64_t gen_time;        // Generation time (nanoseconds)
    int64_t trigger_time;    // Trigger time (latency tracking)
    int32_t msg_type;        // Message type ID
    uint32_t source;         // Source location UID
    uint32_t dest;           // Destination location UID
} __attribute__((packed));
```

**Zero-Copy Design**:
- `frame` object = pointer to mmap region
- `data<T>()` template returns typed reference (no copy)
- Memory layout: `[header][user_data]`

### Page Management

Source: `include/kungfu/yijinjing/journal/page.h`

**Adaptive Sizing**:
```cpp
// From page.h
if (category == MD && dest_id == 0)
    page_size = 128 * MB;  // Market data: high throughput
else if ((category == TD || category == STRATEGY) && dest_id != 0)
    page_size = 4 * MB;    // Trading/Strategy: moderate
else
    page_size = 1 * MB;    // Default
```

**Page Header**:
```cpp
struct page_header {
    uint32_t version;              // Journal version = 4
    uint32_t page_size;            // Configured size
    uint64_t last_frame_position;  // Write pointer
} __attribute__((packed));
```

**Operations**:
- `is_full()`: check if space for next frame
- `begin_time()`, `end_time()`: time range of page
- Auto-rotation when full

### Journal Operations

Source: `include/kungfu/yijinjing/journal/journal.h`

**Reader** (multi-journal subscription):
```cpp
class reader {
    void join(location, dest_id, from_time);  // Subscribe from time T
    void disjoin(location_uid);               // Unsubscribe
    frame_ptr current_frame();                // Current event
    void next();                              // Advance
    void seek_to_time(nanotime);              // Jump to time
    void sort();                              // Time-based merge sort
};
```

**Writer** (single-journal writing):
```cpp
class writer {
    frame_ptr open_frame(trigger_time, msg_type, length);  // Allocate
    void close_frame(data_length);                         // Finalize
    template<typename T>
    frame_ptr write(trigger_time, msg_type, const T& data); // Type-safe write
};
```

### Location System

Source: `include/kungfu/yijinjing/common.h`

**Classification**:
```cpp
enum class mode : int8_t {
    LIVE,      // Real-time trading
    DATA,      // Data recording
    REPLAY,    // Event replay
    BACKTEST   // Backtesting
};

enum class category : int8_t {
    MD,        // Market Data
    TD,        // Trade Data  
    STRATEGY,  // Strategy
    SYSTEM     // System events
};

enum class layout : int8_t {
    JOURNAL,   // Memory-mapped journal
    SQLITE,    // SQLite database
    NANOMSG,   // IPC socket
    LOG        // Log file
};
```

**Location Identity**:
```cpp
class location {
    const mode mode;
    const category category;
    const string group;        // e.g., "binance"
    const string name;         // e.g., "BTC-USDT"
    const string uname;        // "md/binance/BTC-USDT/live"
    const uint32_t uid;        // hash32(uname)
};
```

Storage path: `runtime/journal/[date]/[source_uid]/[dest_uid]_[page_id].journal`

### Time System

Source: `include/kungfu/yijinjing/time.h`

**Precision**: nanosecond
```cpp
time_unit::NANOSECONDS_PER_SECOND = 1,000,000,000

int64_t time::now_in_nano();  // Unix timestamp * 1e9 + ns
int64_t time::strptime(timestr, format);
string time::strftime(nanotime, format);
```

All timestamps: `int64_t` (nanoseconds since Unix epoch)

### Event System

Source: `include/kungfu/yijinjing/common.h`

**Base Interface**:
```cpp
class event {
    virtual int64_t gen_time() const = 0;
    virtual int64_t trigger_time() const = 0;
    virtual int32_t msg_type() const = 0;
    virtual uint32_t source() const = 0;
    virtual uint32_t dest() const = 0;
    virtual uint32_t data_length() const = 0;
    
    template<typename T>
    const T& data() const;  // Zero-copy typed access
};
```

**Publisher/Observer** (IPC notifications):
```cpp
class publisher {
    virtual int notify() = 0;
    virtual int publish(const string& json_message) = 0;
};

class observer {
    virtual bool wait() = 0;
    virtual const string& get_notice() = 0;
};
```

### I/O Device

Source: `include/kungfu/yijinjing/io.h`

Central I/O hub for each component:
```cpp
class io_device {
    journal::reader_ptr open_reader_to_subscribe();
    journal::reader_ptr open_reader(location, dest_id);
    journal::writer_ptr open_writer(dest_id);
    nanomsg::socket_ptr connect_socket(location, protocol, timeout);
    nanomsg::socket_ptr bind_socket(protocol, timeout);
    publisher_ptr get_publisher();
    observer_ptr get_observer();
};
```

**Variants**:
- `io_device_master`: Master process (REP socket)
- `io_device_client`: Client process (REQ socket)

### Reactive Programming

Uses RxCPP for event filtering:
```cpp
// From common.h
rx::is(msg_type)   // Filter by message type
rx::from(source)   // Filter by source UID
rx::to(dest)       // Filter by destination UID
```

## wingchun (詠春) - Trading Layer

**Location**: `core/cpp/wingchun/`

Depends on yijinjing: `INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/cpp/yijinjing/include)`

### Trading Data Types

Source: `include/kungfu/wingchun/common.h`

**Constants**:
```cpp
const int SYMBOL_LEN = 32;
const int EXCHANGE_ID_LEN = 16;
const int ACCOUNT_ID_LEN = 32;
const int ORDER_ID_LEN = 32;
```

**Core Enums**:
```cpp
enum class InstrumentType : int8_t {
    Unknown, FFuture, DFuture, Future, 
    Etf, Spot, Index, Swap
};

enum class Side : int8_t {
    Buy, Sell, Lock, Unlock, Exec, Drop
};

enum class Offset : int8_t {
    Open, Close, CloseToday, CloseYesterday
};

enum class OrderType : int8_t {
    Limit,   // Limit order
    Market,  // Market order
    Mock,    // Self-trade
    UnKnown
};

enum class OrderStatus : int8_t {
    Unknown, Submitted, Pending, Cancelled, 
    Error, Filled, PartialFilledNotActive, 
    PartialFilledActive, PreSend
};
```

**Exchanges** (from source):
```cpp
#define EXCHANGE_BINANCE "binance"
#define EXCHANGE_OKX "okx"
#define EXCHANGE_GATE "gate"
#define EXCHANGE_MEXC "mexc"
#define EXCHANGE_BYBIT "bybit"
#define EXCHANGE_KUCOIN "kucoin"
#define EXCHANGE_XT "xt"
#define EXCHANGE_COINW "coinw"
```

### Message Types

Source: `include/kungfu/wingchun/msg.h` (1085 lines!)

**Market Data** (101-110):
```cpp
enum MsgType {
    Depth = 101,        // Order book depth
    Ticker = 102,       // Ticker data
    Trade = 103,        // Trade data
    IndexPrice = 104,   // Index price
    Bar = 110,          // K-line bar
```

**Trading** (201-213):
```cpp
    OrderInput = 201,        // Order submission
    OrderAction = 202,       // Order action (cancel/query)
    Order = 203,             // Order status update
    MyTrade = 204,           // Trade execution
    Position = 205,          // Position update
    Asset = 206,             // Asset balance
    AssetSnapshot = 207,     // Asset snapshot
    Instrument = 209,        // Instrument info
    AlgoOrderInput = 210,    // Algo order input
    AlgoOrderReport = 211,   // Algo order report
    AlgoOrderModify = 212,   // Algo order modify
    OrderActionError = 213,  // Order action error
```

**Subscription** (302-304):
```cpp
    Subscribe = 302,         // Subscribe market data
    SubscribeAll = 303,      // Subscribe all
    Unsubscribe = 304,       // Unsubscribe
```

**Key Data Structures** (packed structs):
```cpp
struct Instrument {
    char symbol[SYMBOL_LEN];
    char exchange_id[EXCHANGE_ID_LEN];
    InstrumentType instrument_type;
    char product_id[PRODUCT_ID_LEN];
    int contract_multiplier;
    double price_tick;
    char open_date[DATE_LEN];
    char expire_date[DATE_LEN];
    int delivery_year;
    int delivery_month;
    bool is_trading;
    double long_margin_ratio;
    double short_margin_ratio;
} __attribute__((packed));

struct Ticker {
    char source_id[SOURCE_ID_LEN];
    char symbol[SYMBOL_LEN];
    char exchange_id[EXCHANGE_ID_LEN];
    int64_t data_time;
    InstrumentType instrument_type;
    double bid_price;
    double bid_volume;
    double ask_price;
    double ask_volume;
} __attribute__((packed));

struct Order {
    char symbol[SYMBOL_LEN];
    char exchange_id[EXCHANGE_ID_LEN];
    char account_id[ACCOUNT_ID_LEN];
    char order_id[ORDER_ID_LEN];
    int64_t insert_time;
    int64_t update_time;
    OrderStatus status;
    Side side;
    Offset offset;
    double limit_price;
    double frozen_price;
    int64_t volume;
    int64_t volume_traded;
    // ... more fields
} __attribute__((packed));
```

### Broker Layer

Source: `include/kungfu/wingchun/broker/`

**MarketData** (marketdata.h):
```cpp
class MarketData : public practice::apprentice {
    virtual bool subscribe(const vector<Instrument>& instruments) = 0;
    virtual bool subscribe_trade(const vector<Instrument>&) = 0;
    virtual bool subscribe_ticker(const vector<Instrument>&) = 0;
    virtual bool subscribe_index_price(const vector<Instrument>&) = 0;
    virtual bool subscribe_all() = 0;
    virtual bool unsubscribe(const vector<Instrument>&) = 0;
};
```

**Trader** (trader.h):
- Order submission
- Order cancellation
- Position queries
- Asset queries

### Strategy Layer

Source: `include/kungfu/wingchun/strategy/strategy.h`

**Strategy Base Class**:
```cpp
class Strategy {
    // Lifecycle
    virtual void pre_start(Context_ptr context) {}
    virtual void post_start(Context_ptr context) {}
    virtual void pre_stop(Context_ptr context) {}
    virtual void post_stop(Context_ptr context) {}
    
    // Market data callbacks
    virtual void on_depth(Context_ptr, const Depth& depth) {}
    virtual void on_ticker(Context_ptr, const Ticker& ticker) {}
    virtual void on_index_price(Context_ptr, const IndexPrice&) {}
    virtual void on_bar(Context_ptr, const Bar& bar) {}
    
    // Trading callbacks
    virtual void on_order(Context_ptr, const Order& order) {}
    virtual void on_trade(Context_ptr, const Trade& trade) {}
    virtual void on_transaction(Context_ptr, const MyTrade&) {}
    virtual void on_order_action_error(Context_ptr, const OrderActionError&) {}
    virtual void on_position(Context_ptr, const Position&) {}
};
```

User strategies override these callbacks.

### Service Layer

Source: `include/kungfu/wingchun/service/`

**Ledger** (ledger.h):
- Position tracking
- PnL calculation
- Asset management

**Bar Service** (bar.h):
- K-line aggregation
- Time-based bar generation

**Algo Service** (algo.h):
- Algorithmic order management
- TWAP, VWAP, etc.

### Book

Source: `include/kungfu/wingchun/book/book.h`

Order book management for market making strategies.

## Data Flow

### Market Data Pipeline

```
Exchange (WebSocket)
  ↓
Gateway::on_message()
  ↓
Parse & normalize to wingchun::Ticker/Depth
  ↓
writer->write<Ticker>(msg_type::Ticker, ticker)
  ↓
yijinjing journal (persistent)
  ↓
Strategy reads via reader->next()
  ↓
Strategy::on_ticker(context, ticker)
```

### Order Execution Pipeline

```
Strategy::insert_order(context, order_input)
  ↓
writer->write<OrderInput>(msg_type::OrderInput, input)
  ↓
yijinjing journal
  ↓
Trader gateway reads order
  ↓
Exchange API call
  ↓
Exchange confirms
  ↓
Gateway::on_order_update()
  ↓
writer->write<Order>(msg_type::Order, order)
  ↓
yijinjing journal
  ↓
Strategy reads
  ↓
Strategy::on_order(context, order)
```

### Backtest Mode

Set `mode = BACKTEST`:
1. Historical journal replay (time-ordered)
2. Strategy processes as if live
3. Simulated order fills (based on market data)
4. Results written to new journal
5. Analysis via journal reader

## Python Layer

**Location**: `core/python/kungfu/`

### Package Structure

```
kungfu/
├── __main__.py         # kfc CLI entry (calls command.execute())
├── command/            # CLI implementation
│   ├── account/       # Account management
│   ├── algo/          # Algo order commands
│   └── journal/       # Journal inspection
├── yijinjing/         # Python wrapper (pybind11)
├── wingchun/          # Python wrapper (pybind11)
│   ├── algo/         # Algo order Python API
│   ├── backtest/     # Backtesting framework
│   ├── book/         # Order book
│   └── service/      # Services
└── data/
    └── sqlite/        # SQLite storage
```

### CLI Entry

Source: `core/python/kungfu/__main__.py`

```python
import kungfu.command as kfc

def main():
    kfc.execute()  # Dispatches to subcommands
```

Command: `kfc`

### Python Bindings

Built with pybind11:
- `core/cpp/yijinjing/pybind/` → `kungfu.yijinjing`
- `core/cpp/wingchun/pybind/` → `kungfu.wingchun`

Strategies written in Python call C++ core via bindings.

## Extensions

**Location**: `core/extensions/`

Exchange-specific gateway implementations.

Example: `core/extensions/binance/`
- Implements `MarketData` and `Trader` interfaces
- Handles Binance WebSocket protocol
- Converts to wingchun data types

## Threading Model

**yijinjing**:
- Single writer thread per journal (no locks)
- Multiple reader threads (lock-free)
- Nano

msg socket threads for IPC

**wingchun**:
- One thread per Gateway (MD + TD)
- One thread per Strategy
- Shared Broker thread

**Python**:
- GIL limits to single-threaded Python execution
- C++ worker threads bypass GIL

## Performance Notes

**Measured** (from code comments):
- Journal write: <1μs
- Software latency: <20μs (strategy decision to wire)
- Network latency: 1-10ms (variable)

**Optimizations**:
1. Memory-mapped files (zero-copy)
2. Lock-free reads (page is read-only after written)
3. Packed structs (cache-friendly)
4. Lazy page loading
5. Pre-allocated frame buffers

**Bottlenecks**:
- Network to exchange (largest component)
- Python GIL (if strategy in Python)
- Disk I/O (mitigated by mmap)

## Configuration

**Environment Variables**:
```bash
KF_HOME      # Base folder (default: ./runtime)
KF_LOG_LEVEL # Logging level (DEBUG/INFO/WARN/ERROR)
KF_NO_EXT    # Disable extensions if set
```

**Runtime Structure**:
```
runtime/
├── journal/           # Event journals
│   └── [date]/
│       └── [source_uid]/
├── log/              # Application logs
├── config/           # Configuration files
└── cache/            # Temporary data
```

## Build System

**Root**: `core/CMakeLists.txt`

```cmake
PROJECT(kungfu)
SET(CMAKE_CXX_STANDARD 17)

ADD_SUBDIRECTORY(deps)     # Third-party deps
ADD_SUBDIRECTORY(cpp)      # C++ core
ADD_SUBDIRECTORY(extensions) # Exchange gateways
```

**Dependencies** (from CMakeLists.txt):
- nanomsg 1.1.5
- spdlog 1.3.1
- json 3.5.0 (nlohmann)
- SQLiteCpp 2.3.0
- fmt 5.3.0
- rxcpp 4.1.0

**Python Setup**: `core/python/setup.py`
```python
setup(
    name="kungfu",
    entry_points={"console_scripts": ["kfc = kungfu.__main__:main"]},
    install_requires=[
        "click>=5.1",
        "sqlalchemy==1.3.8",
        "psutil==5.6.2",
        "numpy",
        "pandas",
        "tabulate==0.8.3",
        "PyInquirer==1.0.3",
        "prompt_toolkit==1.0.14",
        "rx==3.0.1"
    ],
)
```

## Code Statistics

Based on actual file counts (2025-10-23):

```
core/cpp/yijinjing:  7,357 lines
core/cpp/wingchun:   5,799 lines
core/python:         ~5,000 lines (estimated)
Total core:          ~18,000 lines
```

Key files:
- `wingchun/msg.h`: 1,085 lines (trading message definitions)
- `yijinjing/common.h`: 310 lines (core abstractions)

## Related Documentation

- [ORIGIN.md](ORIGIN.md) - Project fork history
- [INSTALL.md](INSTALL.md) - Environment setup
- [HACKING.md](HACKING.md) - Development workflow
- [adr/001-docker.md](adr/001-docker.md) - Docker decision
- [adr/002-wsl2.md](adr/002-wsl2.md) - WSL2 decision
- [adr/003-dns.md](adr/003-dns.md) - DNS strategy

---

Last Updated: 2025-10-23 (Based on code reading)
