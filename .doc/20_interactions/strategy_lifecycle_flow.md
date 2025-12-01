---
title: Strategy Lifecycle Flow
updated_at: 2025-11-17
owner: core-dev
lang: en
tags: [interaction, flow, strategy, lifecycle, callbacks]
dependencies:
  - strategy_framework
  - yijinjing_journal
  - python_bindings
code_refs:
  - core/cpp/wingchun/src/strategy/runner.cpp:55-215
  - core/cpp/wingchun/include/kungfu/wingchun/strategy/runner.h:1-50
purpose: "Documents the complete lifecycle of a strategy from startup to shutdown"
tokens_estimate: 4200
---

# Strategy Lifecycle Flow

## Overview

This document describes the complete lifecycle of a trading strategy, from process startup through event loop execution to graceful shutdown. Understanding this flow is critical for proper strategy initialization, resource management, and cleanup.

## Lifecycle Phases

```
1. Process Start
   └─> PM2 launches Python process

2. Runner Initialization
   └─> Create Runner, load strategy module

3. on_start() [C++ Internal]
   ├─> Create Context
   ├─> Register event subscriptions
   └─> Call strategy.pre_start()

4. pre_start() [User Callback]
   └─> User initialization code

5. Event Loop Setup
   └─> Subscribe to event streams (Depth, Order, etc.)

6. post_start() [User Callback]
   └─> Post-initialization code

7. Event Processing Loop [MAIN EXECUTION]
   ├─> on_depth()
   ├─> on_ticker()
   ├─> on_order()
   ├─> on_transaction()
   └─> ... (other callbacks)

8. Shutdown Signal (SIGINT)
   └─> Graceful shutdown initiated

9. pre_stop() [User Callback]
   └─> Cleanup before event loop stops

10. on_exit() [C++ Internal]
    └─> Stop event loop, close connections

11. post_stop() [User Callback]
    └─> Final cleanup

12. Process Exit
```

## Detailed Flow

### Phase 1-2: Process Startup

**Trigger**: PM2 starts strategy process

```bash
pm2 start strategy_demo.json
# Executes: python3 dev_run.py -l info strategy -s demo_spot -p strategies/demo_spot.py
```

**C++ Actions** ([runner.cpp:27-30](../../core/cpp/wingchun/src/strategy/runner.cpp)):
```cpp
Runner::Runner(locator_ptr locator, const string &group, const string &name, mode m, bool low_latency)
    : apprentice(location::make(m, category::STRATEGY, group, name, locator), low_latency)
{
    // Creates strategy process location
    // Registers with master service
}
```

**Python Actions**:
1. Import strategy module (`demo_spot.py`)
2. Instantiate strategy class
3. Create `Runner` instance
4. Call `runner.add_strategy(strategy, path)` ([runner.cpp:48-53](../../core/cpp/wingchun/src/strategy/runner.cpp))

**State After**: Strategy registered but not initialized

---

### Phase 3: on_start() - C++ Internal Setup

**Trigger**: `runner.run()` called from Python

**Source**: [runner.cpp:55-194](../../core/cpp/wingchun/src/strategy/runner.cpp)

**Step 1: Create Context** (line 57):
```cpp
context_ = make_context();
context_->react();  // Start reactive event processing
```

Creates the `Context` object that will be passed to all strategy callbacks.

**Step 2: Call User pre_start()** (lines 60-64):
```cpp
for (const auto &strategy : strategies_) {
    context_->set_current_strategy_index(strategy.first);
    strategy.second->pre_start(context_);
}
```

This is where user initialization code runs (see Phase 4).

**Step 3: Register Event Subscriptions** (lines 66-185):

The runner sets up reactive event pipelines for each message type:

#### Depth Events (lines 66-76):
```cpp
events_ | is(msg::type::Depth) |
$([&](event_ptr event) {
    for (const auto &strategy : strategies_) {
        context_->set_current_strategy_index(strategy.first);
        if (context_->is_subscribed("depth", strategy.first, event->data<Depth>())) {
            strategy.second->on_depth(context_, event->data<Depth>());
        }
    }
});
```

**Logic**:
1. Filter events by type (`Depth`)
2. For each strategy, check if subscribed to this symbol
3. If subscribed, call `strategy.on_depth(context, depth)`

#### Ticker Events (lines 78-88):
```cpp
events_ | is(msg::type::Ticker) |
$([&](event_ptr event) {
    for (const auto &strategy : strategies_) {
        // Similar to Depth
    }
});
```

#### Trade Events (lines 90-100):
```cpp
events_ | is(msg::type::Trade) |
$([&](event_ptr event) {
    for (const auto &strategy : strategies_) {
        // Similar to Depth
    }
});
```

#### Order Events (lines 124-141):
```cpp
events_ | is(msg::type::Order) | to(context_->app_.get_home_uid()) |
$([&](event_ptr event) {
    auto order = event->data<Order>();
    for (const auto &strategy : strategies_) {
        if (order.strategy_id == strategy.first) {  // Match by strategy ID
            context_->set_current_strategy_index(strategy.first);
            strategy.second->on_order(context_, order);
            break;  // Only route to owning strategy
        }
    }
});
```

**Key Difference**: Orders are routed only to the strategy that created them (via `strategy_id`).

#### MyTrade Events (lines 154-163):
```cpp
events_ | is(msg::type::MyTrade) | to(context_->app_.get_home_uid()) |
$([&](event_ptr event) {
    auto myTrade = event->data<MyTrade>();
    auto itr = strategies_.find(myTrade.strategy_id);
    if (itr != strategies_.end()) {
        context_->set_current_strategy_index(itr->first);
        itr->second->on_transaction(context_, myTrade);
    }
});
```

#### Position Events (lines 165-174):
```cpp
events_ | is(msg::type::Position) |
$([&](event_ptr event) {
    auto position = event->data<Position>();
    for (const auto &strategy : strategies_) {
        context_->set_current_strategy_index(strategy.first);
        strategy.second->on_position(context_, position);
    }
});
```

**Note**: All strategies receive position updates (not filtered by strategy_id).

**Step 4: Start Event Loop** (line 187):
```cpp
apprentice::on_start();  // Starts journal reader event loop
```

**State After**: Event subscriptions registered, waiting for events

---

### Phase 4: pre_start() - User Initialization

**Trigger**: Called from `Runner::on_start()` (line 63)

**Purpose**: User-defined initialization before event loop starts

**Typical Actions**:
```python
def pre_start(self, context):
    """Called ONCE before event loop starts"""

    # 1. Add trading accounts
    context.add_account("binance", "my_account")

    # 2. Subscribe to market data
    context.subscribe("binance", ["btcusdt", "ethusdt"],
                      InstrumentType.Spot, Exchange.BINANCE)

    # 3. Initialize strategy state
    context.set_object("position_target", 0.0)
    context.set_object("order_count", 0)

    # 4. Set timers (optional)
    context.add_timer(60 * 1e9, lambda ctx: self.on_minute(ctx))  # Every 60s

    # 5. Log startup
    context.log().info("Strategy initialized")
```

**Critical Rules**:
- ✅ **DO**: Initialize state, add accounts, subscribe to symbols
- ✅ **DO**: Set up timers, load configuration
- ❌ **DON'T**: Place orders (accounts may not be ready)
- ❌ **DON'T**: Access market data (not received yet)
- ❌ **DON'T**: Run long computations (blocks startup)

**Execution Order** (Multiple Strategies):
If multiple strategies are added to the runner, `pre_start()` is called sequentially:
```cpp
// Strategy A pre_start()
// Strategy B pre_start()
// Strategy C pre_start()
// Then event loop starts
```

---

### Phase 5: Event Loop Setup Complete

**Trigger**: `apprentice::on_start()` completes (line 187)

**State**:
- All event subscriptions active
- Journal reader polling for events
- Market data gateways streaming data
- Trading gateways ready to receive orders

**Timeline**:
```
T=0ms:    Runner::on_start() called
T=5ms:    Context created
T=10ms:   pre_start() executes
T=20ms:   Event subscriptions registered
T=25ms:   Event loop starts
T=30ms:   post_start() executes
T=35ms+:  Event callbacks start firing
```

---

### Phase 6: post_start() - Post-Initialization

**Trigger**: After event loop starts (line 192)

**Source**: [runner.cpp:189-193](../../core/cpp/wingchun/src/strategy/runner.cpp)
```cpp
for (const auto &strategy : strategies_) {
    context_->set_current_strategy_index(strategy.first);
    strategy.second->post_start(context_);
}
```

**Purpose**: Post-initialization actions that require event loop to be running

**Typical Actions**:
```python
def post_start(self, context):
    """Called ONCE after event loop starts"""

    # Query initial positions from exchange
    context.query_positions("my_account", "", InstrumentType.Spot)

    # Place initial orders (if strategy requires)
    # (Usually better to wait for market data)

    # Log that strategy is ready
    context.log().info("Strategy ready for trading")
```

**Difference from pre_start()**:
- `pre_start()`: Event loop NOT running yet → Cannot receive responses
- `post_start()`: Event loop IS running → Can receive query responses

---

### Phase 7: Event Processing Loop (Main Execution)

**Duration**: Runs continuously until shutdown signal

**Event Flow**:
```
Journal → Event Queue → Runner → Filter → Strategy Callback
```

**Example: Depth Event Processing**:

1. **Market Data Gateway** receives WebSocket depth update from Binance
2. **Gateway** writes `Depth` event to journal
3. **Journal** broadcasts event to all readers
4. **Runner** receives event, filters by type (`msg::type::Depth`)
5. **Runner** checks if strategy is subscribed to this symbol
6. **Runner** calls `strategy.on_depth(context, depth)`
7. **Strategy** executes user code (e.g., place order based on price)

**Callback Execution Model**:
```python
# Every depth update (10-20/sec per symbol)
def on_depth(self, context, depth):
    # Runs in ~5-10 microseconds (C++ → Python overhead)
    # MUST return quickly to avoid blocking other events
    if depth.bid_price[0] > self.buy_threshold:
        context.insert_order(...)

# Every order status change (per order)
def on_order(self, context, order):
    if order.status == OrderStatus.Filled:
        context.log().info(f"Filled at {order.avg_price}")

# Every trade execution (per fill)
def on_transaction(self, context, trade):
    self.total_volume += trade.volume
```

**Concurrency Model**:
- **Single-threaded**: All callbacks run sequentially in the same thread
- **Non-blocking**: Each callback should complete in <1ms
- **Event ordering**: Journal guarantees event order preservation

**Performance**:
- Depth callback overhead: ~5-10 μs (C++ → Python)
- Order callback overhead: ~5-10 μs
- Total latency: Event generation → Strategy callback ≈ 50-200 μs

---

### Phase 8: Shutdown Signal

**Trigger**: User sends SIGINT (Ctrl+C) or PM2 stop

**Signal Handling**:
```bash
# Graceful shutdown
pm2 stop strategy_demo

# Or manually
kill -2 <strategy_pid>  # SIGINT
```

**C++ Reception**:
The `apprentice` base class catches SIGINT and initiates graceful shutdown.

---

### Phase 9: pre_stop() - Pre-Shutdown Cleanup

**Trigger**: Shutdown signal received, before event loop stops

**Source**: [runner.cpp:196-202](../../core/cpp/wingchun/src/strategy/runner.cpp)
```cpp
void Runner::on_exit() {
    for (const auto &strategy : strategies_) {
        context_->set_current_strategy_index(strategy.first);
        strategy.second->pre_stop(context_);
    }
    // ... continue shutdown
}
```

**Purpose**: Clean up while event loop is still running

**Typical Actions**:
```python
def pre_stop(self, context):
    """Called when shutdown initiated, event loop still running"""

    # Cancel all open orders
    for order_id in self.open_orders:
        context.cancel_order("my_account", order_id, "", "", InstrumentType.Spot)

    # Unsubscribe from market data (optional)
    context.unsubscribe("binance", ["btcusdt"], InstrumentType.Spot, Exchange.BINANCE)

    # Log final state
    context.log().info(f"Total trades: {self.trade_count}")
    context.log().info(f"Final PnL: {self.total_pnl}")
```

**Execution Order** (Multiple Strategies):
```cpp
// Strategy A pre_stop()
// Strategy B pre_stop()
// Strategy C pre_stop()
// Then event loop stops
```

---

### Phase 10: on_exit() - Stop Event Loop

**Trigger**: After `pre_stop()` completes

**Source**: [runner.cpp:204](../../core/cpp/wingchun/src/strategy/runner.cpp)
```cpp
apprentice::on_exit();  // Stop journal reader, close connections
```

**Actions**:
- Stop reading from journal
- Close all journal connections
- Unregister from master service
- Release resources

**State After**: Event loop stopped, no more callbacks will fire

---

### Phase 11: post_stop() - Final Cleanup

**Trigger**: After event loop stops

**Source**: [runner.cpp:206-210](../../core/cpp/wingchun/src/strategy/runner.cpp)
```cpp
for (const auto &strategy : strategies_) {
    context_->set_current_strategy_index(strategy.first);
    strategy.second->post_stop(context_);
}
```

**Purpose**: Final cleanup after event loop is stopped

**Typical Actions**:
```python
def post_stop(self, context):
    """Called after event loop stops"""

    # Write final statistics to file
    with open("/tmp/strategy_stats.json", "w") as f:
        json.dump({
            "total_trades": self.trade_count,
            "total_pnl": self.total_pnl,
            "max_drawdown": self.max_drawdown
        }, f)

    # Close external connections (if any)
    # self.redis_client.close()

    # Log shutdown complete
    print("Strategy shutdown complete")
```

**Difference from pre_stop()**:
- `pre_stop()`: Can still interact with context (cancel orders, etc.)
- `post_stop()`: Event loop stopped, context methods may not work

---

### Phase 12: Process Exit

**Trigger**: `post_stop()` completes, `runner.run()` returns

**Actions**:
- Python process exits
- PM2 marks process as stopped
- All resources released

## Callback Summary Table

| Callback | When Called | Can Use Context? | Typical Use |
|----------|-------------|------------------|-------------|
| `pre_start()` | Before event loop | ✅ Yes (add_account, subscribe) | Setup, subscriptions |
| `post_start()` | After event loop starts | ✅ Yes (full access) | Query positions, initial state |
| `on_depth()` | Every depth update | ✅ Yes | Market making, price monitoring |
| `on_ticker()` | Every ticker update | ✅ Yes | Price alerts, statistics |
| `on_trade()` | Every public trade | ✅ Yes | Trade flow analysis |
| `on_order()` | Order status change | ✅ Yes | Order management, logging |
| `on_transaction()` | Your trade fills | ✅ Yes | Position tracking, PnL calc |
| `on_position()` | Position updates | ✅ Yes | Position sync, reconciliation |
| `pre_stop()` | Shutdown initiated | ✅ Yes (can cancel orders) | Cleanup, cancel orders |
| `post_stop()` | After event loop stops | ⚠️  Limited (no event responses) | Save state to disk |

## Multi-Strategy Behavior

**Scenario**: Multiple strategies in single runner process

```python
# Python startup
strategy_a = StrategyA()
strategy_b = StrategyB()

runner = Runner(...)
runner.add_strategy(strategy_a, "strategies/strategy_a.py")
runner.add_strategy(strategy_b, "strategies/strategy_b.py")
runner.run()
```

**Lifecycle Order**:
```
1. strategy_a.pre_start(context)
2. strategy_b.pre_start(context)
3. [Event loop starts]
4. strategy_a.post_start(context)
5. strategy_b.post_start(context)
6. [Event processing - both strategies receive events in parallel]
7. [Shutdown signal]
8. strategy_a.pre_stop(context)
9. strategy_b.pre_stop(context)
10. [Event loop stops]
11. strategy_a.post_stop(context)
12. strategy_b.post_stop(context)
```

**Event Routing**:
- **Market Data** (Depth, Ticker, Trade): All strategies receive if subscribed
- **Orders/Trades**: Only routed to strategy that created them (via `strategy_id`)

## Error Handling

### Exception in pre_start()

```python
def pre_start(self, context):
    raise RuntimeError("Config file not found")
```

**Result**: Strategy process exits, never reaches event loop

### Exception in Event Callback

```python
def on_depth(self, context, depth):
    raise ValueError("Invalid price")
```

**Result**:
- Exception logged to PM2 logs
- Current event processing aborts
- Next event continues normally
- Strategy keeps running (resilient to single event errors)

### Exception in pre_stop()

```python
def pre_stop(self, context):
    raise Exception("Cleanup failed")
```

**Result**:
- Exception logged
- Shutdown continues to `on_exit()` and `post_stop()`
- Process exits despite error

## Related Documentation

### Modules
- [Strategy Framework](../10_modules/strategy_framework.md) - Strategy development guide
- [Python Bindings](../10_modules/python_bindings.md) - C++/Python callback mechanism
- [Yijinjing Journal](../10_modules/yijinjing_journal.md) - Event sourcing system

### Contracts
- [Strategy Context API](../30_contracts/strategy_context_api.md) - Context methods reference

### Operations
- [PM2 Startup Guide](../90_operations/pm2_startup_guide.md) - Process management
- [Debugging Guide](../90_operations/DEBUGGING.md) - Troubleshooting strategies

## Version History

- **2025-11-17**: Initial strategy lifecycle flow documentation
