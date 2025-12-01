---
title: Yijinjing (易筋經) - Event Sourcing System
updated_at: 2025-11-17
owner: core-dev
lang: en
tokens_estimate: 6500
layer: 10_modules
tags: [yijinjing, event-sourcing, journal, mmap, high-frequency-trading, persistence]
purpose: "High-performance event sourcing infrastructure with nanosecond precision for trading systems"
code_refs:
  - core/cpp/yijinjing/include/kungfu/yijinjing/journal/journal.h
  - core/cpp/yijinjing/include/kungfu/yijinjing/journal/frame.h
  - core/cpp/yijinjing/src/journal/writer.cpp
  - core/cpp/yijinjing/src/journal/reader.cpp
---

# Yijinjing (易筋經) - Event Sourcing System

## Purpose

**Yijinjing** is a high-performance event sourcing infrastructure designed for low-latency trading systems. It provides persistent, lock-free journaling of all trading events with nanosecond precision timestamps.

**Problem it solves:**
- Persistent event logging for complete system audit trails
- Lock-free concurrent access for high-frequency trading
- Time-travel debugging and replay capabilities
- Zero-copy memory-mapped I/O for minimal latency
- Event streaming across distributed trading components

**Core concept:** All system state changes are captured as immutable events stored in memory-mapped journal files, enabling event replay, backtesting, and deterministic system behavior.

**System size:** ~6,014 lines of C++ code

## Public API

### Core Event Types

#### `event` - Abstract Event Base Class

**Location**: [core/cpp/yijinjing/include/kungfu/yijinjing/common.h:36-73](../../core/cpp/yijinjing/include/kungfu/yijinjing/common.h#L36-L73)

**Purpose**: Base interface for all events in the system

**Key Methods**:
- `int64_t gen_time()` - Event generation timestamp (nanoseconds since epoch)
- `int64_t trigger_time()` - Event trigger timestamp (business logic time)
- `int32_t msg_type()` - Message type identifier (see `kungfu::msg::type`)
- `uint32_t source()` - Source location UID (hash of location name)
- `uint32_t dest()` - Destination ID (0 for broadcasts)
- `const T& data<T>()` - Type-safe data access

**Usage**:
```cpp
const Order& order = event->data<Order>();
int64_t timestamp = event->gen_time();
```

#### `frame` - Concrete Event Implementation

**Location**: [core/cpp/yijinjing/include/kungfu/yijinjing/journal/frame.h:59-154](../../core/cpp/yijinjing/include/kungfu/yijinjing/journal/frame.h#L59-L154)

**Purpose**: Concrete event backed by memory-mapped storage

**Memory Layout**:
```cpp
struct frame_header {  // 48 bytes total, packed for alignment
    uint32_t length;           // Total frame length including header + data
    uint32_t header_length;    // Header size (always 48)
    int64_t gen_time;          // Generation timestamp (nanoseconds)
    int64_t trigger_time;      // Trigger timestamp (business time)
    int32_t msg_type;          // Message type identifier
    uint32_t source;           // Source location UID
    uint32_t dest;             // Destination ID
} __attribute__((packed));
```

**Data Access**:
- `address_t data_address()` - Raw data pointer (after header)
- `uint32_t data_length()` - Length of data payload
- `bool has_data()` - Check if frame contains data

### Journal System

#### `journal` - Continuous Event Stream

**Location**: [core/cpp/yijinjing/include/kungfu/yijinjing/journal/journal.h:43-85](../../core/cpp/yijinjing/include/kungfu/yijinjing/journal/journal.h#L43-L85)

**Purpose**: Provides continuous memory access abstraction over multiple pages

**Constructor**:
```cpp
journal(location_ptr location, uint32_t dest_id, bool is_writing, bool lazy);
```

**Key Methods**:
- `frame_ptr current_frame()` - Get current frame pointer
- `void next()` - Move to next frame in journal
- `void seek_to_time(int64_t nanotime)` - Seek to specific timestamp
- `int page_id()` - Get current page ID
- `location_ptr get_location()` - Get journal location

**Implementation**: [core/cpp/yijinjing/src/journal/journal.cpp:25-89](../../core/cpp/yijinjing/src/journal/journal.cpp#L25-L89)

#### `reader` - Multi-Journal Event Consumer

**Location**: [core/cpp/yijinjing/include/kungfu/yijinjing/journal/journal.h:87-122](../../core/cpp/yijinjing/include/kungfu/yijinjing/journal/journal.h#L87-L122)

**Purpose**: Consume events from multiple journals in timestamp order

**Constructor**:
```cpp
reader(bool lazy = true);
```

**Key Methods**:
- `void join(location_ptr location, uint32_t dest_id, int64_t from_time)` - Subscribe to journal
- `void disjoin(uint32_t location_uid)` - Unsubscribe from journal
- `bool data_available()` - Check if data is ready to read
- `frame_ptr current_frame()` - Get current frame (earliest across all journals)
- `void next()` - Advance to next frame (re-sorts all journals)
- `void seek_to_time(int64_t nanotime)` - Seek all journals to timestamp

**Auto-sorting Feature**: Reader maintains sorted order across multiple journals by comparing `gen_time` timestamps.

**Implementation**: [core/cpp/yijinjing/src/journal/reader.cpp:22-94](../../core/cpp/yijinjing/src/journal/reader.cpp#L22-L94)

**Example**:
```cpp
reader_ptr reader = std::make_shared<reader>(true);
reader->join(md_location, 0, start_time);
reader->join(td_location, 0, start_time);

while (reader->data_available()) {
    frame_ptr event = reader->current_frame();
    // Process event from earliest journal
    reader->next();  // Advances and re-sorts
}
```

#### `writer` - Event Producer

**Location**: [core/cpp/yijinjing/include/kungfu/yijinjing/journal/journal.h:124-207](../../core/cpp/yijinjing/include/kungfu/yijinjing/journal/journal.h#L124-L207)

**Purpose**: Thread-safe event writing to journal

**Constructor**:
```cpp
writer(location_ptr location, uint32_t dest_id, bool lazy, publisher_ptr publisher);
```

**Key Methods**:

**Low-level API** (open/close pattern):
- `frame_ptr open_frame(int64_t trigger_time, int32_t msg_type, uint32_t length)` - Begin writing
- `void close_frame(size_t data_length)` - Commit frame (sets `gen_time`)

**Type-safe API** (recommended):
- `T& open_data<T>(int64_t trigger_time, int32_t msg_type)` - Type-safe write access
- `void close_data()` - Commit typed data

**Convenience API** (most common):
- `void write<T>(int64_t trigger_time, int32_t msg_type, const T& data)` - Atomic write
- `void mark(int64_t trigger_time, int32_t msg_type)` - Write marker event (no data)

**Thread-Safety**: Uses mutex with spin lock (100 retries with 1μs sleep) for concurrent write protection.

**Implementation**: [core/cpp/yijinjing/src/journal/writer.cpp:38-149](../../core/cpp/yijinjing/src/journal/writer.cpp#L38-L149)

**Examples**:
```cpp
// Pattern 1: Simple write (most common)
Ticker ticker = {...};
writer->write(now, msg::type::Ticker, ticker);

// Pattern 2: Open/modify/close (for complex data)
auto& order = writer->open_data<Order>(now, msg::type::OrderInput);
order.volume = 100;
order.price = 25.5;
writer->close_data();

// Pattern 3: Marker event (no data)
writer->mark(now, msg::type::PageEnd);
```

### Storage Layer

#### `page` - Memory-Mapped Journal Page

**Location**: [core/cpp/yijinjing/include/kungfu/yijinjing/journal/page.h:43-106](../../core/cpp/yijinjing/include/kungfu/yijinjing/journal/page.h#L43-L106)

**Purpose**: Memory-mapped file representing a fixed-size journal page (1MB-128MB)

**Static Methods**:
- `page_ptr load(location_ptr, uint32_t dest_id, int page_id, bool is_writing, bool lazy)` - Load or create page
- `std::string get_page_path(location_ptr, uint32_t dest_id, int id)` - Generate page file path
- `int find_page_id(location_ptr, uint32_t dest_id, int64_t time)` - Find page containing timestamp

**Key Methods**:
- `uint32_t get_page_size()` - Get page size in bytes
- `int64_t begin_time()` - First frame timestamp in page
- `int64_t end_time()` - Last frame timestamp in page
- `bool is_full()` - Check if page is full (no space for next frame)
- `address_t address_border()` - End address of page

**Implementation**: [core/cpp/yijinjing/src/journal/page.cpp:21-125](../../core/cpp/yijinjing/src/journal/page.cpp#L21-L125)

**Adaptive Page Sizing**:
```cpp
// From page.h:108-119
MD category, dest_id=0:     128 MB  // Large market data streams
TD/STRATEGY, dest_id≠0:     4 MB    // Trading and strategy events
Default:                    1 MB    // General events
```

**File Naming Convention**: `{dest_id:08x}.{page_id}.journal`
- Example: `00000000.00000001.journal` (first page for dest_id=0)
- Example: `0000002a.00000005.journal` (page 5 for dest_id=42)

### I/O Device

#### `io_device` - Centralized I/O Hub

**Location**: [core/cpp/yijinjing/include/kungfu/yijinjing/io.h:20-66](../../core/cpp/yijinjing/include/kungfu/yijinjing/io.h#L20-L66)

**Purpose**: Centralized factory for creating readers and writers

**Constructor**:
```cpp
io_device(location_ptr home, bool low_latency = false, bool lazy = true);
```

**Key Methods**:
- `reader_ptr open_reader_to_subscribe()` - Create multi-journal subscriber
- `reader_ptr open_reader(location_ptr location, uint32_t dest_id)` - Open specific journal reader
- `writer_ptr open_writer(uint32_t dest_id)` - Create writer at home location
- `writer_ptr open_writer_at(location_ptr location, uint32_t dest_id)` - Create writer at specific location
- `publisher_ptr get_publisher()` - Get event publisher (for low-latency mode)
- `observer_ptr get_observer()` - Get event observer

**Implementation**: [core/cpp/yijinjing/src/io/io.cpp:26-200](../../core/cpp/yijinjing/src/io/io.cpp#L26-L200)

**Example**:
```cpp
auto io = std::make_shared<io_device>(home_location, true, false);
reader_ptr reader = io->open_reader_to_subscribe();
writer_ptr writer = io->open_writer(0);
```

### Location System

#### `location` - Component Identifier

**Location**: [core/cpp/yijinjing/include/kungfu/yijinjing/common.h:218-241](../../core/cpp/yijinjing/include/kungfu/yijinjing/common.h#L218-L241)

**Purpose**: Unique identifier and metadata for system components

**Constructor**:
```cpp
location(mode m, category cat, std::string group, std::string name, locator_ptr locator);
```

**Fields**:
- `mode` - LIVE, DATA, REPLAY, BACKTEST ([common.h:105-111](../../core/cpp/yijinjing/include/kungfu/yijinjing/common.h#L105-L111))
- `category` - MD, TD, STRATEGY, SYSTEM ([common.h:144-150](../../core/cpp/yijinjing/include/kungfu/yijinjing/common.h#L144-L150))
- `std::string group` - Component group (e.g., "binance")
- `std::string name` - Component name (e.g., "trader_binance")
- `std::string uname` - Unique name: `{category}/{group}/{name}/{mode}`
- `uint32_t uid` - 32-bit hash of `uname`

**Example**:
```
category=TD, group=binance, name=trader, mode=LIVE
→ uname = "td/binance/trader/live"
→ uid = hash("td/binance/trader/live") = 0x1a2b3c4d
```

### Time System

#### `time` - Nanosecond Precision Utilities

**Location**: [core/cpp/yijinjing/include/kungfu/yijinjing/time.h:40-80](../../core/cpp/yijinjing/include/kungfu/yijinjing/time.h#L40-L80)

**Static Methods**:
- `int64_t now_in_nano()` - Current time in nanoseconds since epoch
- `int64_t strptime(const std::string& timestr, const std::string& format)` - Parse time string to nanoseconds
- `const std::string strftime(int64_t nanotime, const std::string& format)` - Format nanoseconds to string
- `const std::string strfnow(const std::string& format)` - Format current time to string

**Constants** ([time_unit class, time.h:19-38](../../core/cpp/yijinjing/include/kungfu/yijinjing/time.h#L19-L38)):
```cpp
time_unit::NANOSECONDS_PER_SECOND    = 1000000000LL
time_unit::NANOSECONDS_PER_MILLISECOND = 1000000LL
time_unit::NANOSECONDS_PER_MICROSECOND = 1000LL
```

**Example**:
```cpp
int64_t now = time::now_in_nano();
std::string formatted = time::strfnow("%Y-%m-%d %H:%M:%S");
int64_t parsed = time::strptime("2025-01-15 10:30:00", "%Y-%m-%d %H:%M:%S");
```

## Inputs / Outputs

### Inputs

**Writer inputs:**
- `trigger_time` (int64_t) - Business logic timestamp for event
- `msg_type` (int32_t) - Message type identifier (see [kungfu/yijinjing/msg.h](../../core/cpp/yijinjing/include/kungfu/yijinjing/msg.h))
- `data` (template T) - Event data payload

**Reader inputs:**
- `location` (location_ptr) - Journal location to read from
- `dest_id` (uint32_t) - Destination ID (0 for broadcast journals)
- `from_time` (int64_t) - Start timestamp for reading

### Outputs

**Reader outputs:**
- `frame_ptr` - Immutable event frame with header + data
- Events delivered in timestamp order across all subscribed journals

**Storage outputs:**
- Journal files: `{KF_HOME}/runtime/journal/{location}/{dest_id:08x}.{page_id}.journal`
- Example: `/app/runtime/journal/live/td/binance/trader/00000000.00000001.journal`

## Dependencies

### External Libraries

- **RxCpp** - Reactive extensions for event streaming (used in practice framework)
- **spdlog** - Fast C++ logging library
- **nanomsg** - Scalable protocols library for inter-process communication
- **pybind11** - Python bindings
- **fmt** - String formatting library
- **Boost** (uuid, core) - UUID generation

### Internal Dependencies

- **kungfu/common.h** - Shared macros (`DECLARE_PTR`, `FORWARD_DECLARE_PTR`)

### System APIs

**Linux**:
- `mmap` - Memory mapping
- `mlock` - Lock pages in memory (prevents swapping)
- `madvise` - Memory access pattern hints
- `munmap` - Unmap memory

**Windows**:
- `CreateFileMapping` - Create file mapping object
- `MapViewOfFile` - Map view of file to memory

## Architecture

### Three-Layer Storage Design

```
┌─────────────────────────────────────────────────┐
│ Frame Layer (event records)                     │
│ - 48-byte header + variable data                │
│ - Atomic write unit                             │
│ - Immutable after commit                        │
└──────────────────┬──────────────────────────────┘
                   │ Multiple frames per page
┌──────────────────▼──────────────────────────────┐
│ Page Layer (memory-mapped files)                │
│ - 1MB-128MB adaptive sizing                     │
│ - mmap for zero-copy access                     │
│ - Sequential frame storage                      │
│ - Automatic rotation when full                  │
└──────────────────┬──────────────────────────────┘
                   │ Multiple pages per journal
┌──────────────────▼──────────────────────────────┐
│ Journal Layer (continuous stream)               │
│ - Automatic page rotation                       │
│ - Time-based seeking                            │
│ - Multi-reader support                          │
│ - Single writer per journal                     │
└─────────────────────────────────────────────────┘
```

### Event Sourcing Flow

**Write Path** ([writer.cpp:55-92](../../core/cpp/yijinjing/src/journal/writer.cpp#L55-L92)):

```cpp
1. writer->open_frame(trigger_time, msg_type, data_length)
   - Acquire writer mutex (spin lock with timeout)
   - Check page space, rotate to new page if needed
   - Set frame header fields (trigger_time, msg_type, source, dest)

2. Copy data to frame->data_address()
   - Zero-copy write directly to mmap buffer
   - Application controls data layout

3. writer->close_frame(data_length)
   - Set gen_time = now_in_nano()  // Commit timestamp
   - Update page last_frame_position
   - Release mutex
   - Notify publisher (if low_latency mode)
```

**Read Path** ([reader.cpp:55-91](../../core/cpp/yijinjing/src/journal/reader.cpp#L55-L91)):

```cpp
1. reader->join(location, dest_id, from_time)
   - Create journal for location/dest
   - Seek journal to from_time
   - Add journal to journals_ vector

2. reader->data_available()
   - Sort all journals by current frame.gen_time
   - Set current_ to journal with earliest frame
   - Return true if any journal has data

3. reader->current_frame()
   - Return frame pointer from current journal

4. reader->next()
   - Advance current journal to next frame
   - Re-sort journals for next read
```

### Memory-Mapped I/O

**Implementation**: [core/cpp/yijinjing/src/util/mmap.cpp:33-148](../../core/cpp/yijinjing/src/util/mmap.cpp#L33-L148)

**Linux Process** ([mmap.cpp:80-122](../../core/cpp/yijinjing/src/util/mmap.cpp#L80-L122)):
```cpp
1. open() with O_RDWR (writers) or O_RDONLY (readers)
2. lseek() + write() to stretch file to page_size (writers only)
3. mmap() with:
   - PROT_READ | PROT_WRITE (writers)
   - PROT_READ (readers)
   - MAP_SHARED (changes visible to all processes)
4. madvise(MADV_RANDOM) for random access pattern
5. mlock() to prevent swapping (if not lazy mode)
```

**Benefits**:
- **Zero-copy**: Direct memory access to file data without read/write syscalls
- **Kernel-managed**: OS handles paging and caching automatically
- **Lock-free reads**: Multiple processes can read simultaneously
- **Persistence**: Data survives process crashes

**Memory Management**:
- Lazy mode: Pages swapped to disk when not in use
- Non-lazy mode: `mlock()` keeps pages in RAM for minimal latency

### Page Management

**Page Rotation** ([writer.cpp:67-69, 133-149](../../core/cpp/yijinjing/src/journal/writer.cpp#L67-L69)):

```cpp
// Check if current frame would exceed page boundary
if (frame->address() + sizeof(frame_header) + data_length
    >= current_page->address_border()) {
    close_page(trigger_time);  // Write PageEnd marker
    load_next_page();          // Increment page_id, mmap new file
}
```

**Page Cleanup** ([journal.cpp:71-83](../../core/cpp/yijinjing/src/journal/journal.cpp#L71-L83)):

Optional cleanup via `CLEAR_JOURNAL` environment variable:
```cpp
const int MAX_PAGE_NUMBER = 50;  // MD journals use 8
if (page_id > MAX_PAGE_NUMBER) {
    unlink(path_to_old_page);  // Delete oldest page to save disk space
}
```

**Page File Format**:
```
Filename: {dest_id:08x}.{page_id}.journal
Location: {KF_HOME}/runtime/journal/{mode}/{category}/{group}/{name}/

Examples:
/app/runtime/journal/live/md/binance/marketdata/00000000.00000001.journal
/app/runtime/journal/live/td/binance/trader/00000000.00000003.journal
```

### Practice Framework (Hero Pattern)

**hero** - Base Actor Class

**Location**: [core/cpp/yijinjing/include/kungfu/practice/hero.h:22-118](../../core/cpp/yijinjing/include/kungfu/practice/hero.h#L22-L118)

**Purpose**: Base class for all system actors (MD, TD, Strategy)

**Provides**:
- Event loop with RxCpp observable stream (`events_`)
- Reader/writer management
- Location registry
- Channel management

**apprentice** - Client-Side Actor

**Location**: [core/cpp/yijinjing/include/kungfu/practice/apprentice.h:22-190](../../core/cpp/yijinjing/include/kungfu/practice/apprentice.h#L22-L190)

**Purpose**: Client-side actor that communicates with master

**Extends**: `hero`

**Provides**:
- Registration with master process
- Timer services (`add_timer`, `add_time_interval`)
- Read/write permission requests
- Convenience methods for event writing

**Common Pattern**:
```cpp
class MyStrategy : public apprentice {
public:
    MyStrategy(location_ptr location)
        : apprentice(location, false) {}

    void on_trading_day(const event_ptr& event, int64_t daytime) override {
        // Trading day initialization
    }

    void on_start() override {
        // Subscribe to market data
        subscribe(md_location, msg::type::Ticker);

        // Set up event handlers
        events_ | is(msg::type::Ticker) | $([&](event_ptr e) {
            const Ticker& tick = e->data<Ticker>();
            // Process ticker...
        });
    }
};
```

## Usage Examples

### Pattern 1: Simple Event Writing

**Most Common Pattern** ([apprentice.h:54-57](../../core/cpp/yijinjing/include/kungfu/practice/apprentice.h#L54-L57)):

```cpp
template<typename T>
void write_to(int64_t trigger_time, int32_t msg_type, T& data, uint32_t dest_id = 0) {
    writers_[dest_id]->write(trigger_time, msg_type, data);
}

// Usage in strategy:
Ticker ticker = {...};
write_to(now_, msg::type::Ticker, ticker);

Order order = {...};
write_to(now_, msg::type::OrderInput, order);
```

### Pattern 2: Open/Modify/Close (Complex Data)

**For Large or Complex Structures** ([writer.h:152-165](../../core/cpp/yijinjing/include/kungfu/yijinjing/journal/journal.h#L152-L165)):

```cpp
// Get writable reference to data in journal
auto& order = writer->open_data<Order>(trigger_time, msg::type::OrderInput);

// Modify in-place (zero-copy)
order.volume = 100;
order.price = 25.5;
strcpy(order.instrument_id, "BTCUSDT");

// Commit changes
writer->close_data();
```

### Pattern 3: Marker Events (No Data)

**For State Transitions** ([writer.cpp:94-98](../../core/cpp/yijinjing/src/journal/writer.cpp#L94-L98)):

```cpp
// Page rotation marker
writer->mark(trigger_time, msg::type::PageEnd);

// Trading day markers
writer->mark(now_, msg::type::TradingDayStart);
writer->mark(now_, msg::type::TradingDayEnd);
```

### Pattern 4: Multi-Journal Reading

**Subscribe to Multiple Sources** ([apprentice.cpp:123](../../core/cpp/yijinjing/src/practice/apprentice.cpp#L123)):

```cpp
// Subscribe to master commands
reader_->join(master_commands_location_, get_live_home_uid(), start_time);

// Subscribe to market data from Binance
reader_->join(binance_md_location, 0, start_time);

// Subscribe to trading events
reader_->join(binance_td_location, 0, start_time);

// Events automatically merged in timestamp order
```

### Pattern 5: Event Loop with RxCpp

**Functional Reactive Processing** ([apprentice.cpp:139-149](../../core/cpp/yijinjing/src/practice/apprentice.cpp#L139-L149)):

```cpp
// Update current time for all events
events_ | $([&](event_ptr event) {
    now_ = event->gen_time();
});

// Register locations
events_ | is(msg::type::Location) | $([&](event_ptr e) {
    register_location_from_event(e);
});

// Process tickers
events_ | is(msg::type::Ticker) | $([&](event_ptr e) {
    const Ticker& tick = e->data<Ticker>();
    on_ticker(tick);
});

// Process order responses
events_ | is(msg::type::Order) | $([&](event_ptr e) {
    const Order& order = e->data<Order>();
    on_order(order);
});
```

### Pattern 6: Time-Based Operations

**Seeking to Timestamp** ([reader.cpp:61-68](../../core/cpp/yijinjing/src/journal/reader.cpp#L61-L68)):

```cpp
// Seek all subscribed journals to specific time
int64_t replay_start = time::strptime("2025-01-15 09:30:00", "%Y-%m-%d %H:%M:%S");
reader->seek_to_time(replay_start);

// Read from that point forward
while (reader->data_available()) {
    frame_ptr event = reader->current_frame();
    // Process historical events...
    reader->next();
}
```

**Timer Services** ([apprentice.cpp:53-78](../../core/cpp/yijinjing/src/practice/apprentice.cpp#L53-L78)):

```cpp
// One-time timer
int64_t target_time = now_ + 5 * time_unit::NANOSECONDS_PER_SECOND;
add_timer(target_time, [&](event_ptr e) {
    LOG_INFO("Timer fired at {}", e->gen_time());
});

// Repeating interval
int64_t interval = 1 * time_unit::NANOSECONDS_PER_SECOND;  // 1 second
add_time_interval(interval, [&](event_ptr e) {
    LOG_INFO("Heartbeat at {}", time::strfnow("%H:%M:%S"));
});
```

## Hotspots & Pitfalls

### Hotspot 1: Writer Thread Safety

**Issue**: Only ONE writer per journal is safe
**Why**: Writer uses mutex but assumes single writer thread
**Solution**: Create separate writers with different `dest_id` for concurrent writes

```cpp
// WRONG: Multiple threads writing to same journal
writer->write(now, msg::type::Ticker, tick1);  // Thread 1
writer->write(now, msg::type::Ticker, tick2);  // Thread 2 - RACE CONDITION!

// CORRECT: Different dest_id for each thread
writer_thread1->write(now, msg::type::Ticker, tick1);  // dest_id=1
writer_thread2->write(now, msg::type::Ticker, tick2);  // dest_id=2
```

### Hotspot 2: Page Rotation Performance

**Issue**: Page rotation causes brief write stall
**When**: Frame size + current position > page boundary
**Impact**: 1-2ms latency spike during `mmap()` of new page
**Mitigation**: Use larger page sizes for high-throughput journals (set via page.h adaptive sizing)

### Hotspot 3: Memory Locking (mlock)

**Issue**: Non-lazy mode locks ALL pages in RAM
**Risk**: Can exhaust system memory with large journals
**When to use**:
- ✅ GOOD: Low-latency trading (mlock prevents page faults)
- ❌ BAD: Large historical data (wastes RAM)

```cpp
// Low-latency mode: mlock all pages (no swapping)
io_device io(home, true, false);  // low_latency=true, lazy=false

// Memory-efficient mode: allow swapping
io_device io(home, false, true);  // low_latency=false, lazy=true
```

### Pitfall 1: Forgetting to call `close_frame()` or `close_data()`

**Symptom**: Events not visible to readers, journal appears corrupt
**Cause**: `gen_time` only set in `close_frame()`, readers skip incomplete frames

```cpp
// WRONG: Missing close_data()
auto& order = writer->open_data<Order>(now, msg::type::OrderInput);
order.volume = 100;
// BUG: Forgot to call close_data()!

// CORRECT:
auto& order = writer->open_data<Order>(now, msg::type::OrderInput);
order.volume = 100;
writer->close_data();  // Essential!
```

### Pitfall 2: Using `trigger_time` vs `gen_time`

**Confusion**: Two timestamps per event
- `trigger_time`: Business logic time (when event SHOULD occur, can be future)
- `gen_time`: Actual generation time (set by `close_frame()`)

**Usage**:
```cpp
// Backtesting: trigger_time = historical time
writer->write(backtest_time, msg::type::Ticker, tick);  // trigger_time in past

// Live trading: trigger_time = now
writer->write(time::now_in_nano(), msg::type::Ticker, tick);

// Scheduled events: trigger_time in future
int64_t target = now + 5 * time_unit::NANOSECONDS_PER_SECOND;
writer->write(target, msg::type::ScheduledTask, task);
```

### Pitfall 3: Page Cleanup Enabled in Production

**Risk**: Deleting pages while readers are still accessing them
**Symptom**: Segmentation fault or corrupt reads
**Check**: Ensure `CLEAR_JOURNAL` environment variable is NOT set in production

```bash
# Development: OK to enable cleanup
export CLEAR_JOURNAL=1

# Production: NEVER enable
unset CLEAR_JOURNAL
```

### Pitfall 4: Incorrect `dest_id` for Broadcasts

**Rule**: `dest_id=0` means broadcast (all readers)
**Common mistake**: Using `dest_id=0` for point-to-point messages

```cpp
// WRONG: Using dest_id=0 for targeted message
writer->write(now, msg::type::OrderInput, order);  // Broadcast to ALL!

// CORRECT: Use specific dest_id for targeted delivery
uint32_t strategy_uid = get_location_uid("strategy/my_strat");
writer_to_strategy->write(now, msg::type::OrderInput, order);  // Only to strategy
```

### Pitfall 5: Not Handling `data_available()` Correctly

**Issue**: Reader returns false when all journals exhausted
**Mistake**: Assuming false means "no data forever"
**Reality**: New data may arrive later (live mode)

```cpp
// WRONG: Exits on first false
while (reader->data_available()) {
    process(reader->current_frame());
    reader->next();
}
// BUG: Never checks for new data!

// CORRECT: Live mode loop
while (running_) {
    if (reader->data_available()) {
        process(reader->current_frame());
        reader->next();
    } else {
        usleep(1000);  // Sleep 1ms and retry
    }
}
```

## Code References

### Key Headers

**Core interfaces:**
- [core/cpp/yijinjing/include/kungfu/yijinjing/common.h](../../core/cpp/yijinjing/include/kungfu/yijinjing/common.h) - Base types (event, publisher, observer, location)
- [core/cpp/yijinjing/include/kungfu/yijinjing/io.h](../../core/cpp/yijinjing/include/kungfu/yijinjing/io.h) - I/O device
- [core/cpp/yijinjing/include/kungfu/yijinjing/time.h](../../core/cpp/yijinjing/include/kungfu/yijinjing/time.h) - Time utilities
- [core/cpp/yijinjing/include/kungfu/yijinjing/msg.h](../../core/cpp/yijinjing/include/kungfu/yijinjing/msg.h) - System message types

**Journal subsystem:**
- [core/cpp/yijinjing/include/kungfu/yijinjing/journal/common.h](../../core/cpp/yijinjing/include/kungfu/yijinjing/journal/common.h) - Journal types
- [core/cpp/yijinjing/include/kungfu/yijinjing/journal/frame.h](../../core/cpp/yijinjing/include/kungfu/yijinjing/journal/frame.h) - Frame and frame_header
- [core/cpp/yijinjing/include/kungfu/yijinjing/journal/page.h](../../core/cpp/yijinjing/include/kungfu/yijinjing/journal/page.h) - Page and page_header
- [core/cpp/yijinjing/include/kungfu/yijinjing/journal/journal.h](../../core/cpp/yijinjing/include/kungfu/yijinjing/journal/journal.h) - journal, reader, writer

**Practice framework:**
- [core/cpp/yijinjing/include/kungfu/practice/hero.h](../../core/cpp/yijinjing/include/kungfu/practice/hero.h) - Base hero class
- [core/cpp/yijinjing/include/kungfu/practice/apprentice.h](../../core/cpp/yijinjing/include/kungfu/practice/apprentice.h) - Apprentice actor
- [core/cpp/yijinjing/include/kungfu/practice/master.h](../../core/cpp/yijinjing/include/kungfu/practice/master.h) - Master coordinator

**Utilities:**
- [core/cpp/yijinjing/include/kungfu/yijinjing/util/os.h](../../core/cpp/yijinjing/include/kungfu/yijinjing/util/os.h) - OS operations (mmap)
- [core/cpp/yijinjing/include/kungfu/yijinjing/util/util.h](../../core/cpp/yijinjing/include/kungfu/yijinjing/util/util.h) - Hash utilities
- [core/cpp/yijinjing/include/kungfu/yijinjing/nanomsg/socket.h](../../core/cpp/yijinjing/include/kungfu/yijinjing/nanomsg/socket.h) - Nanomsg IPC
- [core/cpp/yijinjing/include/kungfu/yijinjing/log/setup.h](../../core/cpp/yijinjing/include/kungfu/yijinjing/log/setup.h) - Logging

### Key Implementations

**Journal:**
- [core/cpp/yijinjing/src/journal/journal.cpp](../../core/cpp/yijinjing/src/journal/journal.cpp) - Journal core (89 lines)
- [core/cpp/yijinjing/src/journal/reader.cpp](../../core/cpp/yijinjing/src/journal/reader.cpp) - Multi-journal reader (94 lines)
- [core/cpp/yijinjing/src/journal/writer.cpp](../../core/cpp/yijinjing/src/journal/writer.cpp) - Thread-safe writer (153 lines)
- [core/cpp/yijinjing/src/journal/page.cpp](../../core/cpp/yijinjing/src/journal/page.cpp) - Page management (125+ lines)

**I/O:**
- [core/cpp/yijinjing/src/io/io.cpp](../../core/cpp/yijinjing/src/io/io.cpp) - I/O device implementation (200+ lines)

**Practice:**
- [core/cpp/yijinjing/src/practice/hero.cpp](../../core/cpp/yijinjing/src/practice/hero.cpp) - Hero implementation
- [core/cpp/yijinjing/src/practice/apprentice.cpp](../../core/cpp/yijinjing/src/practice/apprentice.cpp) - Apprentice implementation
- [core/cpp/yijinjing/src/practice/master.cpp](../../core/cpp/yijinjing/src/practice/master.cpp) - Master implementation

**Utilities:**
- [core/cpp/yijinjing/src/util/mmap.cpp](../../core/cpp/yijinjing/src/util/mmap.cpp) - Memory-mapped I/O (153 lines)
- [core/cpp/yijinjing/src/time/time.cpp](../../core/cpp/yijinjing/src/time/time.cpp) - Time implementation
- [core/cpp/yijinjing/src/nanomsg/socket.cpp](../../core/cpp/yijinjing/src/nanomsg/socket.cpp) - Nanomsg wrapper

**Python bindings:**
- [core/cpp/yijinjing/pybind/pybind_yjj.cpp](../../core/cpp/yijinjing/pybind/pybind_yjj.cpp) - pybind11 bindings

## Related Documentation

- [wingchun.md](wingchun.md) - Trading gateway that uses yijinjing for event persistence
- [binance_extension.md](binance_extension.md) - Binance connector built on yijinjing
- [../20_interactions/event_flow.md](../20_interactions/event_flow.md) - System-wide event propagation (to be created)
- [../00_index/ARCHITECTURE.md](../00_index/ARCHITECTURE.md) - Overall system architecture
- [../85_memory/DEBUGGING.md](../85_memory/DEBUGGING.md) - Journal-related debugging cases

## Changelog

- **2025-11-17**: Initial module card creation with comprehensive API documentation

---

**Maintenance Note**: When modifying yijinjing, update this document with:
1. New message types in `msg.h`
2. Changes to frame/page/journal structures
3. New usage patterns discovered in wingchun or strategies
4. Performance characteristics from production use
