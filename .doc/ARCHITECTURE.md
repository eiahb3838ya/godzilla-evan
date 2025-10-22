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

Journal-based event sourcing system for financial data.

#### Purpose

- Persistent storage of all events (quotes, trades, orders)
- Event replay for backtesting and debugging
- Time-series data management
- High-performance append-only storage

#### Key Concepts

**Journal**:
- Append-only memory-mapped file
- Stores events chronologically
- Supports multiple readers, single writer
- Lock-free reading

**Frame**:
- Time-indexed container for events
- Nanosecond precision timestamp
- Source identification
- Event type and data

**Reader/Writer**:
- Reader: Zero-copy read from journal
- Writer: Append events to journal
- Multiple concurrent readers
- Single writer per journal

#### File Structure

```
runtime/journal/
├── strategy_1/
│   ├── STRATEGY.journal     # Strategy events
│   └── STRATEGY.index       # Time index
├── gateway_binance/
│   ├── MD.journal          # Market data
│   ├── MD.index
│   ├── TD.journal          # Trade data
│   └── TD.index
```

#### Performance Characteristics

- **Latency**: <1μs for event write
- **Throughput**: 1M+ events/second
- **Storage**: Compressed, minimal overhead
- **Scalability**: Linear with event count

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

Last Updated: 2025-10-22

