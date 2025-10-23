# System Architecture

High-level architecture and design of the system.

## Overview

This project is based on the kungfu trading framework, consisting of two main subsystems:

1. **yijinjing** (易筋經): Event sourcing and journaling system
2. **wingchun** (詠春): Trading gateway abstraction layer

Both subsystems are implemented in C++17 for low latency, with Python bindings for strategy development.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Trading Strategies (Python)              │
│                      /app/strategies/*                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ Python API
                        ├─────────────────┐
                        │                 │
           ┌────────────▼─────────┐  ┌───▼────────────────────┐
           │    wingchun (詠春)   │  │  yijinjing (易筋經)    │
           │  Trading Gateway     │  │   Event Journal        │
           │                      │  │                        │
           │  ┌────────────────┐ │  │  ┌──────────────────┐  │
           │  │   Strategy     │ │  │  │  Journal Writer  │  │
           │  │   Engine       │ │  │  │                  │  │
           │  └────────────────┘ │  │  └──────────────────┘  │
           │                      │  │                        │
           │  ┌────────────────┐ │  │  ┌──────────────────┐  │
           │  │   Broker       │ │  │  │  Journal Reader  │  │
           │  │   (Orders)     │ │  │  │                  │  │
           │  └────────────────┘ │  │  └──────────────────┘  │
           │                      │  │                        │
           │  ┌────────────────┐ │  │  ┌──────────────────┐  │
           │  │   Gateway      │ │  │  │  Time Manager    │  │
           │  │   (Exchange)   │ │  │  │                  │  │
           │  └────────────────┘ │  │  └──────────────────┘  │
           └───────┬──────────────┘  └────────────────────────┘
                   │
                   │ Network
                   ▼
         ┌──────────────────────┐
         │   Exchange APIs      │
         │  (Binance, OKX, ...) │
         └──────────────────────┘
```

## Core Components

### yijinjing (易筋經)

Event sourcing infrastructure based on memory-mapped journal files.

**Code Location**: `core/cpp/yijinjing/`

#### Core Architecture: Three-Layer Design

```
Journal (continuous memory abstraction)
   ↓ manages
Page (memory-mapped file, 1-128 MB)
   ↓ contains
Frame (48-byte header + variable data)
```

#### Layer 1: Frame (Minimal Unit)

**Structure** (`journal/frame.h`):
```cpp
struct frame_header {  // 48 bytes, packed
    uint32_t length;           // Total frame length
    uint32_t header_length;    // Header length (48)
    int64_t gen_time;          // Generate time (nanoseconds)
    int64_t trigger_time;      // Trigger time (latency tracking)
    int32_t msg_type;          // Message type ID
    uint32_t source;           // Source location UID
    uint32_t dest;             // Destination location UID
} __attribute__((packed));
```

**Zero-Copy Design**:
- Frame object holds only a pointer to mmap region
- `data<T>()` template: type-safe data access without copy
- Memory layout: `[frame_header][user_data]`

**Implementation**:
- `frame` class implements `event` interface
- Direct pointer arithmetic to access data
- `move_to_next()`: advance to next frame in page

#### Layer 2: Page (Memory Page)

**Intelligent Sizing** (`journal/page.h`):
- **MD** (Market Data): 128 MB - high throughput
- **TD/STRATEGY**: 4 MB - moderate frequency  
- **Default**: 1 MB

**Memory-Mapped File**:
- One page = one mmap file on disk
- Structure: `[page_header][frame1][frame2]...[frameN]`
- Tracks `last_frame_position` for append
- `is_full()`: checks if space available

**Page Management**:
- Lazy loading: load on first access
- Auto-rotation: create new page when full
- Time-indexed: can seek to specific time

#### Layer 3: Journal (Log Abstraction)

**Purpose** (`journal/journal.h`):
- Manage page loading and switching
- Provide continuous memory view
- Support time-based navigation

**Key Operations**:
- `next()`: move to next frame (O(1))
- `seek_to_time(nanotime)`: jump to specific time (O(log n))
- Automatic page loading on boundary crossing

**Reader/Writer Pattern**:

**Reader**:
- Subscribe to multiple journals
- Time-based merge sort across journals
- Lock-free reading (multiple concurrent readers)
- `join()`: subscribe to a journal from time T
- `disjoin()`: unsubscribe

**Writer**:
- Single writer per journal (no locking needed)
- `open_frame()`: allocate space in current page
- `close_frame()`: finalize and publish notification
- Template `write<T>()`: type-safe write

#### Event System

**Base Abstraction** (`common.h`):
```cpp
class event {  // Abstract interface
    virtual int64_t gen_time() const = 0;
    virtual int64_t trigger_time() const = 0;
    virtual int32_t msg_type() const = 0;
    virtual uint32_t source() const = 0;
    virtual uint32_t dest() const = 0;
    
    template<typename T>
    const T& data() const;  // Zero-copy data access
};
```

All events (market data, orders, fills) implement this interface.

#### Location System

**Data Classification**:
```cpp
mode: LIVE | DATA | REPLAY | BACKTEST
category: MD | TD | STRATEGY | SYSTEM  
layout: JOURNAL | SQLITE | NANOMSG | LOG
```

**Location Identity**:
- Unique ID: `hash32(category/group/name/mode)`
- Example: `md/binance/BTC-USDT/live` → UID
- Determines storage path and page size

#### Time System

**Nanosecond Precision** (`time.h`):
```cpp
time_unit::NANOSECONDS_PER_SECOND = 1,000,000,000
time::now_in_nano() → int64_t  // Unix timestamp * 1e9
```

**Features**:
- `strptime()`: parse string to nano time
- `strftime()`: format nano time to string
- `next_minute_nano()`, `next_day_nano()`: time alignment

#### Publisher/Observer Pattern

**Publisher** (`common.h`):
- `notify()`: send notification
- `publish(json_message)`: publish update

**Observer**:
- `wait()`: block until notification
- `get_notice()`: retrieve message

Used for cross-process communication via nanomsg.

#### File Structure

```
runtime/journal/
└── [date]/
    └── [source_uid]/
        ├── [dest_uid]_0.journal    # Page 0
        ├── [dest_uid]_1.journal    # Page 1
        └── ...
```

Example:
```
runtime/journal/20251023/12345678/
├── 0_0.journal        # 128 MB (MD to all)
├── 87654321_0.journal # 4 MB (TD to specific strategy)
└── 87654321_1.journal # Next page when full
```

#### Performance Characteristics

- **Write latency**: <1μs (mmap + pointer move)
- **Read latency**: <100ns (pointer dereference)
- **Throughput**: 1M+ events/second
- **Storage**: Append-only, no compression (speed priority)
- **Concurrency**: Lock-free reading, single writer

### wingchun (詠春)

Trading gateway abstraction providing unified interface to multiple exchanges.

#### Purpose

- Normalize exchange APIs
- Order management and routing
- Position tracking
- Risk management hooks

#### Key Concepts

**Gateway**:
- Exchange-specific implementation
- Market data feed
- Order execution
- Account information

**Strategy**:
- User-defined trading logic
- Receives market data
- Sends orders
- Tracks positions

**Broker**:
- Routes orders to appropriate gateway
- Manages order lifecycle
- Aggregates positions across gateways
- Risk checks

#### Event Flow

```
Market Data:
Exchange → Gateway → yijinjing → Strategy

Order Flow:
Strategy → Broker → Gateway → Exchange
         ↓
    yijinjing (logged)
```

#### Supported Exchanges

Check `core/cpp/wingchun/gateway/` for available gateways:
- Binance
- OKX
- [Add more as implemented]

## Data Flow

### Market Data Flow

1. **Exchange** sends market data (WebSocket)
2. **Gateway** parses and normalizes data
3. **Gateway** writes to yijinjing journal
4. **Strategy** reads from journal
5. **Strategy** processes data and decides to trade

### Order Flow

1. **Strategy** creates order
2. **Broker** validates order (risk checks)
3. **Broker** writes order to journal
4. **Gateway** reads order from journal
5. **Gateway** sends order to exchange
6. **Exchange** confirms/rejects order
7. **Gateway** writes order update to journal
8. **Strategy** reads order update

### Backtest Flow

1. **Replay** historical market data from journal
2. **Strategy** processes data as if live
3. **Broker** simulates order fills
4. **Results** written to new journal
5. **Analysis** reads results from journal

## Threading Model

### yijinjing

- **Journal Writer**: Single thread per journal
- **Journal Reader**: Multiple threads, lock-free
- **Time Service**: Dedicated thread for time synchronization

### wingchun

- **Gateway Thread**: One thread per gateway
  - Market data reception
  - Order transmission
- **Strategy Thread**: One thread per strategy
  - Event processing
  - Order generation
- **Broker Thread**: Shared thread
  - Order routing
  - Position aggregation

## Memory Management

### Zero-Copy Design

- Journal stored as memory-mapped file
- Readers access journal directly (no copy)
- Minimizes memory allocation in hot path

### Object Pooling

- Pre-allocated event objects
- Reused across event lifecycle
- Reduces GC pressure in Python

## Time Management

### Time Sources

1. **System Time**: Local system clock
2. **Exchange Time**: Exchange timestamps
3. **Strategy Time**: Configurable for backtest

### Time Synchronization

- Periodic sync with NTP servers
- Monotonic time for interval measurement
- Adjustable precision (ms vs μs vs ns)

## Configuration

### Runtime Structure

```
runtime/
├── journal/              # Event journals
│   └── [date]/
│       └── [source]/
├── log/                 # Application logs
├── config/              # Configuration files
└── cache/               # Temporary data
```

### Environment Variables

- `KF_HOME`: Runtime directory (default: `/app/runtime`)
- `KF_LOG_LEVEL`: Logging level (DEBUG/INFO/WARN/ERROR)

## Error Handling

### Exception Strategy

- C++ core: No exceptions in hot path
- Return codes for error conditions
- Critical errors: Terminate with clear message

### Logging

- spdlog for C++ logging
- Python logging for strategies
- Separate log files per component
- Configurable log levels

## Performance Considerations

### Critical Path

Order execution latency breakdown (typical):
```
Strategy decision:     <10μs
Journal write:         <1μs
Gateway processing:    <5μs
Network to exchange:   1-10ms (variable)
Total (software):      <20μs
```

### Optimization Techniques

1. **Lock-Free Data Structures**: Journal reader
2. **Memory-Mapped I/O**: Journal storage
3. **Zero-Copy**: Event passing
4. **Object Pooling**: Reduce allocation
5. **Batch Processing**: Reduce syscalls

### Bottlenecks

- Network latency (exchange communication)
- Disk I/O (journal writes, mitigated by memory-mapping)
- Python GIL (strategies in Python, consider C++ for ultra-low latency)

## Scalability

### Horizontal Scaling

- Multiple strategy instances
- Each with own journal
- Centralized broker for aggregation

### Vertical Scaling

- CPU: Parallel strategy execution
- Memory: Large journal buffers
- Disk: SSD for journal storage

## Security

### API Keys

- Stored in configuration files
- Never logged or written to journal
- File permissions: 600 (owner read/write only)

### Network

- TLS for exchange communication
- API key rotation support
- IP whitelisting (exchange-side)

## Deployment

### Development

- Docker container (Ubuntu 20.04)
- Volume-mounted code
- Hot reload for Python strategies

### Production

- Same Docker image
- Persistent volume for journals
- Monitoring and alerting

See [adr/001-docker.md](adr/001-docker.md) for deployment strategy.

## Future Enhancements

Potential improvements:
- Distributed journal for multi-server setup
- GPU acceleration for strategy computation
- Machine learning integration
- Real-time risk analytics dashboard

## Code Structure Mapping

Documentation to actual code correspondence:

### yijinjing Core

```
core/cpp/yijinjing/
├── include/kungfu/yijinjing/
│   ├── common.h              # Event, location, data classification
│   ├── time.h                # Nanosecond time system
│   ├── io.h                  # I/O abstractions
│   ├── msg.h                 # Message type definitions
│   └── journal/
│       ├── common.h          # Journal forward declarations
│       ├── frame.h           # Frame structure (48-byte header)
│       ├── page.h            # Page management (1-128 MB)
│       └── journal.h         # Journal, reader, writer
├── src/
│   ├── time/                 # Time implementation
│   ├── journal/              # Journal implementation
│   └── io/                   # I/O implementation
└── pybind/                   # Python bindings (pybind11)
```

### wingchun Core

```
core/cpp/wingchun/
├── include/kungfu/wingchun/
│   ├── common.h              # Trading data types
│   ├── msg.h                 # Trading messages (51KB!)
│   ├── broker/
│   │   ├── marketdata.h      # Market data broker
│   │   └── trader.h          # Trading broker
│   ├── strategy/
│   │   ├── context.h         # Strategy context
│   │   ├── strategy.h        # Strategy base class
│   │   └── runner.h          # Strategy runner
│   ├── service/
│   │   ├── ledger.h          # Position/PnL tracking
│   │   ├── bar.h             # K-line aggregation
│   │   └── algo.h            # Algo orders
│   └── book/
│       └── book.h            # Order book
├── src/                      # Implementations
└── pybind/                   # Python bindings
```

### Python Layer

```
core/python/kungfu/
├── __main__.py               # kfc CLI entry point
├── yijinjing/                # Python wrapper for yijinjing
├── wingchun/                 # Python wrapper for wingchun
├── command/                  # CLI commands
│   ├── journal/              # Journal inspection
│   ├── account/              # Account management
│   └── algo/                 # Algo order management
└── data/
    └── sqlite/               # SQLite data storage
```

### Extensions

```
core/extensions/
└── binance/                  # Binance exchange gateway
    ├── CMakeLists.txt
    └── src/
```

## Statistics

Code size (as of 2025-10-23):
- yijinjing C++: ~7,357 lines
- wingchun C++: ~5,799 lines
- Python bindings: ~2,000 lines (estimated)
- Total core: ~15,000 lines

Key dependencies:
- C++17 standard
- spdlog 1.3.1 (logging)
- fmt 5.3.0 (formatting)
- rxcpp 4.1.0 (reactive extensions)
- nanomsg 1.1.5 (messaging)
- SQLiteCpp 2.3.0 (database)
- pybind11 (Python bindings)

## Related Documentation

- [INSTALL.md](INSTALL.md) - Setup and deployment
- [HACKING.md](HACKING.md) - Development workflow
- [adr/001-docker.md](adr/001-docker.md) - Docker architecture decision
- [ORIGIN.md](ORIGIN.md) - Project history and fork details

## References

- Event Sourcing: https://martinfowler.com/eaaDev/EventSourcing.html
- Memory-Mapped Files: https://en.wikipedia.org/wiki/Memory-mapped_file
- Lock-Free Programming: https://en.wikipedia.org/wiki/Non-blocking_algorithm

---

Last Updated: 2025-10-23

