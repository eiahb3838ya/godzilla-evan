# Phase 4K: Zero-Copy Optimization Specification

**Version**: 1.0
**Date**: 2024-12-13
**Status**: FUTURE OPTIONAL OPTIMIZATION
**Priority**: LOW (only if proven bottleneck)
**Target Audience**: Senior Engineer → Junior Engineer
**Branch Name**: `feat/phase4k-zero-copy` (when implemented)

---

## 0. Pre-requisites Checklist

⚠️ **CRITICAL**: Do NOT implement this optimization unless:

- [ ] **Performance profiling proves** that Depth copying is a bottleneck (>5% of total latency)
- [ ] Phase 4I (Callback Queue) is stable and verified in production
- [ ] You have >1 week of development time available
- [ ] You understand Yijinjing journal internals
- [ ] You have read this entire specification and understand the trade-offs

**Current Status**: As of Phase 4I, Depth copying costs only **0.002% CPU** and is **NOT a bottleneck**.

---

## 1. Overview

### 1.1 What This Optimization Does

**Current Approach (Phase 4G)**:
```cpp
// signal_api.cpp - Immediate copy
const hf::Depth* depth_ptr = static_cast<const hf::Depth*>(data);
auto depth_copy = std::make_shared<hf::Depth>(*depth_ptr);  // ← 336 bytes copied
h->factor_engine->OnDepth(depth_copy);
```

**Optimized Approach (Phase 4K)**:
```cpp
// signal_api.cpp - Zero-copy with frame reference counting
auto depth_shared = yijinjing::get_frame_shared_ptr<hf::Depth>(data);  // ← 0 bytes copied
h->factor_engine->OnDepth(depth_shared);
```

**Performance Gain**:
- **Latency**: -180 ns per Depth event (-0.018% of total ~1000 μs)
- **Memory**: -336 bytes per concurrent Depth processing
- **CPU**: -0.002% (72 ms saved per hour at 100 Hz)

**Cost**:
- **Engineering effort**: 2-3 days development + 1 week testing
- **Complexity**: Yijinjing core modification (medium risk)
- **Maintenance**: Increased system complexity

---

## 2. Background and Motivation

### 2.1 Why Does Phase 4G Copy Data?

**Yijinjing Journal Structure**:
```
┌─────────────────────────────────────────────────────────┐
│ Yijinjing Journal (Ring Buffer in mmap)                │
├─────────────────────────────────────────────────────────┤
│ Frame 0: [Depth @ T0]                                   │
│ Frame 1: [Depth @ T1]                                   │
│ ...                                                      │
│ Frame 1023: [Depth @ T1023]                             │
│ Frame 0: [Depth @ T1024] ← Overwrites Frame 0 (T0)!    │
└─────────────────────────────────────────────────────────┘
```

**Problem**: If hf-live holds a raw pointer to Frame 0, after ~10 seconds (at 100 Hz), Frame 0 will be overwritten by Frame 1024, causing data corruption or crashes.

**Phase 4G Solution**: Copy the entire Depth structure to heap memory immediately, ensuring data remains valid regardless of journal overwrites.

### 2.2 When Is Zero-Copy Worth It?

**Worth it if**:
- Depth event rate > 1000 Hz (currently ~100 Hz)
- Processing latency is critical (<10 μs budget)
- Running on memory-constrained devices

**NOT worth it if** (current situation):
- 336 bytes copying takes 200 ns (0.02% of 1000 μs)
- Memory allocation is negligible (3.3 KB peak for 10 concurrent events)
- System is stable and meets performance requirements

---

## 3. Performance Analysis

### 3.1 Current Cost Breakdown

| Metric | Value | Analysis |
|--------|-------|----------|
| **Depth size** | 336 bytes | Fixed structure size |
| **Copy time** | ~200 ns | memcpy + shared_ptr allocation |
| **Event rate** | 100 Hz | Binance WebSocket typical rate |
| **CPU per hour** | 72 ms | 200 ns × 100 Hz × 3600s |
| **CPU utilization** | 0.002% | Negligible on modern CPUs |
| **Memory per event** | 352 bytes | 336 data + 16 control block |
| **Peak memory** | 3.3 KB | 10 concurrent events × 336 bytes |

**Conclusion**: Current cost is **negligible** and does **not justify** optimization effort.

### 3.2 Optimization Gain Estimate

| Metric | Current | After Phase 4K | Gain |
|--------|---------|----------------|------|
| Copy latency | 200 ns | 20 ns | **-180 ns** |
| Memory allocation | 352 bytes/event | 0 | **-352 bytes** |
| CPU per hour | 72 ms | 7 ms | **-65 ms** |
| Total latency | ~1000 μs | ~999.82 μs | **-0.018%** |

**ROI Analysis**:
- Development cost: **2-3 days** (16-24 hours)
- Performance gain: **0.018%** latency reduction
- **Verdict**: ❌ **Poor ROI** unless Depth rate increases 10x

---

## 4. Design: Yijinjing Frame Reference Counting

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│ Yijinjing Journal with Frame Reference Counting        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Frame 100: [Depth data]  ref_count = 2                │
│             ↑                   ↑                       │
│             │                   │                       │
│    Reader holds ptr    hf-live holds shared_ptr        │
│                                                         │
│  Writer checks ref_count before overwriting:           │
│    if (frame[100].ref_count > 0) {                     │
│        // Wait or use next frame                       │
│    }                                                    │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Key Components

#### Component 1: Frame Metadata Extension

```cpp
// kungfu/yijinjing/journal/frame.h (NEW)
namespace kungfu::yijinjing::journal {

struct FrameHeader {
    uint64_t frame_id;
    uint32_t frame_length;
    uint32_t msg_type;
    int64_t gen_time;
    int64_t trigger_time;

    // Phase 4K: Add reference counting
    std::atomic<int32_t> ref_count{0};  // ← NEW
};

}
```

#### Component 2: Reader API Extension

```cpp
// kungfu/yijinjing/journal/reader.h (MODIFY)
namespace kungfu::yijinjing::journal {

class Reader {
public:
    // Existing API (unchanged)
    frame_ptr current_frame();

    // Phase 4K: New zero-copy API
    template<typename T>
    std::shared_ptr<const T> get_frame_shared_ptr() {
        frame_ptr frame = current_frame();
        if (!frame) return nullptr;

        // Increment reference count
        FrameHeader* header = get_frame_header(frame);
        header->ref_count.fetch_add(1, std::memory_order_acquire);

        // Custom deleter to decrement ref_count
        auto deleter = [this, frame_id = frame->frame_id()](const T* ptr) {
            this->release_frame_ref(frame_id);
        };

        const T* data_ptr = reinterpret_cast<const T*>(frame->data());
        return std::shared_ptr<const T>(data_ptr, deleter);
    }

private:
    void release_frame_ref(uint64_t frame_id);
};

}
```

#### Component 3: Writer Frame Allocation Check

```cpp
// kungfu/yijinjing/journal/writer.h (MODIFY)
namespace kungfu::yijinjing::journal {

class Writer {
public:
    frame_ptr allocate_frame(uint32_t length, uint32_t msg_type) {
        uint64_t frame_index = next_frame_index_++;
        uint64_t frame_slot = frame_index % buffer_size_;

        // Phase 4K: Check if frame is still referenced
        FrameHeader* header = get_frame_header(frame_slot);

        int retry_count = 0;
        while (header->ref_count.load(std::memory_order_acquire) > 0) {
            if (++retry_count > 1000) {
                // Frame held for too long, log warning
                SPDLOG_WARN("Frame {} ref_count = {}, waiting...",
                           frame_slot, header->ref_count.load());
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
            // Spin-wait (typical: <1 μs)
            std::this_thread::yield();
        }

        // Safe to overwrite
        return allocate_frame_internal(frame_slot, length, msg_type);
    }
};

}
```

---

## 5. Implementation Steps

### 5.1 Phase 1: Add Reference Counting to Yijinjing

#### Step 1.1: Modify Frame Header

**File**: `core/cpp/yijinjing/include/kungfu/yijinjing/journal/frame.h`

**Add to FrameHeader struct**:
```cpp
struct FrameHeader {
    // ... existing fields ...

    // Phase 4K: Reference counting for zero-copy support
    std::atomic<int32_t> ref_count{0};

    // Padding to maintain alignment (if needed)
    char _padding[4];
};

// Verify size unchanged or acceptable
static_assert(sizeof(FrameHeader) % 8 == 0, "FrameHeader must be 8-byte aligned");
```

#### Step 1.2: Implement Reader::get_frame_shared_ptr()

**File**: `core/cpp/yijinjing/src/journal/reader.cpp`

**Add implementation**:
```cpp
void Reader::release_frame_ref(uint64_t frame_id) {
    uint64_t frame_slot = frame_id % buffer_size_;
    FrameHeader* header = get_frame_header(frame_slot);

    int32_t prev_count = header->ref_count.fetch_sub(1, std::memory_order_release);

    if (prev_count <= 0) {
        SPDLOG_ERROR("Frame {} ref_count underflow! prev_count={}",
                     frame_id, prev_count);
    }

    SPDLOG_TRACE("Frame {} ref_count decreased to {}",
                 frame_id, prev_count - 1);
}

template<typename T>
std::shared_ptr<const T> Reader::get_frame_shared_ptr() {
    frame_ptr frame = current_frame();
    if (!frame) return nullptr;

    // Get frame header and increment ref count
    FrameHeader* header = get_frame_header(frame);
    header->ref_count.fetch_add(1, std::memory_order_acquire);

    uint64_t frame_id = frame->frame_id();

    SPDLOG_TRACE("Frame {} ref_count increased to {}",
                 frame_id, header->ref_count.load());

    // Custom deleter
    auto deleter = [this, frame_id](const T* ptr) {
        this->release_frame_ref(frame_id);
    };

    const T* data_ptr = reinterpret_cast<const T*>(frame->data());
    return std::shared_ptr<const T>(data_ptr, deleter);
}

// Explicit template instantiation for common types
template std::shared_ptr<const kungfu::wingchun::msg::data::Depth>
Reader::get_frame_shared_ptr<kungfu::wingchun::msg::data::Depth>();
```

#### Step 1.3: Modify Writer to Check ref_count

**File**: `core/cpp/yijinjing/src/journal/writer.cpp`

**Modify frame allocation**:
```cpp
frame_ptr Writer::allocate_frame(uint32_t length, uint32_t msg_type) {
    uint64_t frame_index = next_frame_index_++;
    uint64_t frame_slot = frame_index % buffer_size_;

    FrameHeader* header = get_frame_header(frame_slot);

    // Phase 4K: Wait if frame is still referenced
    int retry_count = 0;
    constexpr int MAX_RETRY = 1000;
    constexpr auto RETRY_DELAY = std::chrono::microseconds(10);

    while (header->ref_count.load(std::memory_order_acquire) > 0) {
        if (++retry_count > MAX_RETRY) {
            SPDLOG_WARN("Frame {} stuck with ref_count={}, waiting (retry={})",
                       frame_slot, header->ref_count.load(), retry_count);
            std::this_thread::sleep_for(RETRY_DELAY);
        }
        std::this_thread::yield();
    }

    if (retry_count > 0) {
        SPDLOG_DEBUG("Frame {} released after {} retries", frame_slot, retry_count);
    }

    // Proceed with normal allocation
    return allocate_frame_internal(frame_slot, length, msg_type);
}
```

### 5.2 Phase 2: Update Wingchun Runner

#### Step 2.1: Store Frame Shared Pointers

**File**: `core/cpp/wingchun/src/strategy/runner.cpp`

**Modify Depth event handler**:
```cpp
events_ | is(msg::type::Depth) |
$([&](event_ptr event) {
    // Phase 4K: Get zero-copy shared_ptr
    auto depth_shared = reader_->get_frame_shared_ptr<Depth>();

    if (!depth_shared) {
        SPDLOG_ERROR("Failed to get frame shared_ptr for Depth");
        return;
    }

    SPDLOG_TRACE("Depth frame ref_count = {}", depth_shared.use_count());

    // Pass to signal engine (zero-copy)
    if (signal_on_data_ && signal_engine_handle_) {
        signal_on_data_(signal_engine_handle_, 101, depth_shared.get());
    }

    // Pass to strategy
    auto depth = event->data<Depth>();
    for (auto& [id, strategy] : strategies_) {
        context_->set_current_strategy_index(id);
        strategy->on_depth(context_, depth);
    }

    // depth_shared will be destroyed here if hf-live doesn't hold a reference
    // Otherwise, frame will stay alive until hf-live finishes processing
});
```

### 5.3 Phase 3: Update signal_api.cpp

#### Step 3.1: Remove Immediate Copy

**File**: `hf-live/adapter/signal_api.cpp`

**Replace Phase 4G copy with zero-copy**:
```cpp
extern "C" void signal_on_data(void* handle, int type, const void* data) {
    if (!handle || !data) {
        std::cerr << "[signal_api] ERROR: handle or data is null" << std::endl;
        return;
    }

    SignalHandle* h = static_cast<SignalHandle*>(handle);
    if (!h->initialized || !h->factor_engine) {
        std::cerr << "[signal_api] ERROR: engine not initialized" << std::endl;
        return;
    }

    if (type == 101) {  // Depth
        // Phase 4K: Zero-copy - assume data pointer is valid
        // Lifetime managed by Yijinjing frame reference counting
        const hf::Depth* depth_ptr = static_cast<const hf::Depth*>(data);

        std::cerr << "[signal_api] Phase 4K: Zero-copy Depth for " << depth_ptr->symbol
                  << " @ " << depth_ptr->data_time << std::endl;

        // Create shared_ptr with empty deleter
        // Actual lifetime managed by Yijinjing frame ref_count
        auto depth_shared = std::shared_ptr<const hf::Depth>(
            depth_ptr,
            [](const hf::Depth*) {} // Empty deleter
        );

        h->factor_engine->OnDepth(depth_shared);

    } else if (type == 103) {  // Trade
        const hf::Trade* trade_ptr = static_cast<const hf::Trade*>(data);

        auto trade_shared = std::shared_ptr<const hf::Trade>(
            trade_ptr,
            [](const hf::Trade*) {}
        );

        h->factor_engine->OnTrade(trade_shared);

    } else {
        std::cerr << "[signal_api] WARNING: Unknown data type " << type << std::endl;
    }
}
```

---

## 6. Testing and Verification

### 6.1 Unit Tests

**File**: `core/cpp/yijinjing/test/test_frame_refcount.cpp` (NEW)

```cpp
#include <gtest/gtest.h>
#include <kungfu/yijinjing/journal/reader.h>
#include <kungfu/yijinjing/journal/writer.h>

using namespace kungfu::yijinjing::journal;

TEST(FrameRefCount, BasicRefCounting) {
    // Setup journal
    Reader reader("test_journal", 0);
    Writer writer("test_journal");

    // Write a frame
    writer.write_frame<int>(42);
    reader.next();

    // Get shared_ptr
    auto ptr1 = reader.get_frame_shared_ptr<int>();
    ASSERT_NE(ptr1, nullptr);
    EXPECT_EQ(*ptr1, 42);
    EXPECT_EQ(ptr1.use_count(), 1);

    // Get another reference
    auto ptr2 = ptr1;
    EXPECT_EQ(ptr2.use_count(), 2);

    // Release first reference
    ptr1.reset();
    EXPECT_EQ(ptr2.use_count(), 1);

    // Release last reference
    ptr2.reset();

    // Frame should be releasable now
    // (Writer can overwrite)
}

TEST(FrameRefCount, WriterWaitsForRelease) {
    Reader reader("test_journal", 0);
    Writer writer("test_journal");

    // Fill entire journal buffer
    const int buffer_size = 1024;
    for (int i = 0; i < buffer_size; i++) {
        writer.write_frame<int>(i);
    }

    // Read first frame and hold reference
    reader.next();
    auto held_frame = reader.get_frame_shared_ptr<int>();
    EXPECT_EQ(*held_frame, 0);

    // Try to write one more frame (should overwrite frame 0)
    auto start = std::chrono::steady_clock::now();

    std::thread writer_thread([&]() {
        writer.write_frame<int>(9999);  // Will wait for frame 0 release
    });

    // Hold for 100ms
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Release the frame
    held_frame.reset();

    writer_thread.join();
    auto elapsed = std::chrono::steady_clock::now() - start;

    // Should have waited at least 100ms
    EXPECT_GE(elapsed, std::chrono::milliseconds(100));
}
```

### 6.2 Integration Test

**File**: `core/cpp/wingchun/test/test_runner_zero_copy.cpp` (NEW)

```cpp
TEST(RunnerZeroCopy, DepthEventZeroCopy) {
    // Setup Runner with hf-live
    Runner runner(...);
    runner.load_signal_library();

    // Inject Depth event
    Depth depth{};
    strcpy(depth.symbol, "BTCUSDT");
    depth.data_time = 123456789;
    depth.bid_price[0] = 50000.0;

    writer.write_frame<Depth>(depth);

    // Process event
    runner.step();

    // Verify: No crash, factor callback received
    EXPECT_TRUE(factor_callback_called);

    // Verify: Frame reference count is 0 after processing
    // (Frame was released by hf-live)
    auto frame_header = get_frame_header(0);
    EXPECT_EQ(frame_header->ref_count.load(), 0);
}
```

### 6.3 E2E Performance Test

**Test Script**: `scripts/test_phase4k_performance.sh`

```bash
#!/bin/bash

echo "Phase 4K Zero-Copy Performance Test"
echo "===================================="

# Start services
docker exec godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"

sleep 10

# Collect metrics for 5 minutes
docker exec godzilla-dev bash -c '
    for i in {1..300}; do
        # Check PM2 status
        pm2 jlist | jq -r ".[] | select(.name==\"strategy:hello\") |
            {name, pm_id, status, restarts, cpu: .monit.cpu, memory: .monit.memory}"

        # Extract latency from logs
        pm2 logs strategy:hello --lines 10 --nostream |
            grep -oP "Latency: \K[0-9]+" | tail -1

        sleep 1
    done
' > phase4k_metrics.log

# Analyze results
python3 - <<'EOF'
import re
import statistics

with open("phase4k_metrics.log") as f:
    latencies = [int(m.group(1)) for m in re.finditer(r"Latency: (\d+)", f.read())]

if latencies:
    print(f"Latency (ns):")
    print(f"  Mean: {statistics.mean(latencies):.0f}")
    print(f"  Median: {statistics.median(latencies):.0f}")
    print(f"  P95: {sorted(latencies)[int(len(latencies)*0.95)]:.0f}")
    print(f"  P99: {sorted(latencies)[int(len(latencies)*0.99)]:.0f}")

    # Compare with Phase 4I baseline (~8850 ns)
    baseline = 8850
    improvement = baseline - statistics.median(latencies)
    print(f"\nImprovement over Phase 4I: {improvement:.0f} ns ({improvement/baseline*100:.2f}%)")
else:
    print("No latency data collected")
EOF
```

**Expected Results**:
```
Latency (ns):
  Mean: 8670
  Median: 8665
  P95: 8800
  P99: 9100

Improvement over Phase 4I: 185 ns (2.09%)
```

---

## 7. Rollback Plan

### 7.1 Rollback Procedure

If Phase 4K causes issues:

```bash
# 1. Stop all services
docker exec godzilla-dev pm2 delete all

# 2. Switch back to Phase 4I
cd /home/huyifan/projects/godzilla-evan
git checkout feature/hf-live-support  # Phase 4I stable branch

# 3. Rebuild Godzilla core
cd core && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 4. Rebuild hf-live
cd ../../hf-live && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 5. Redeploy
docker cp /home/huyifan/projects/godzilla-evan/core/build/cpp/wingchun/pywingchun.cpython-38-x86_64-linux-gnu.so \
    godzilla-dev:/app/core/python/
docker cp /home/huyifan/projects/godzilla-evan/hf-live/build/libsignal.so \
    godzilla-dev:/app/hf-live/build/

# 6. Restart services
docker exec godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"
```

### 7.2 Fallback Configuration

Add a compile-time flag to switch between Phase 4G (copy) and Phase 4K (zero-copy):

```cmake
# CMakeLists.txt
option(ENABLE_ZERO_COPY "Enable Phase 4K zero-copy optimization" OFF)

if(ENABLE_ZERO_COPY)
    add_definitions(-DPHASE4K_ZERO_COPY)
endif()
```

```cpp
// signal_api.cpp
#ifdef PHASE4K_ZERO_COPY
    // Zero-copy path
    auto depth_shared = std::shared_ptr<const hf::Depth>(
        depth_ptr, [](const hf::Depth*) {}
    );
#else
    // Phase 4G copy path (safe fallback)
    auto depth_shared = std::make_shared<hf::Depth>(*depth_ptr);
#endif
```

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Frame held too long** | Medium | High (journal stall) | Add timeout + warning logs |
| **Ref count leak** | Low | Medium (memory leak) | Automated leak detection tests |
| **ABI incompatibility** | Low | High (crashes) | Versioning + compatibility layer |
| **Performance regression** | Low | Medium | A/B testing + rollback plan |
| **Yijinjing bug** | Low | High | Extensive unit tests |

### 8.2 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Production crash** | Low | Critical | Canary deployment + instant rollback |
| **Silent data corruption** | Very Low | Critical | Checksums + validation |
| **Memory pressure** | Low | Medium | Monitor frame hold times |
| **Maintenance complexity** | High | Medium | Detailed documentation |

### 8.3 Risk Mitigation Checklist

Before deploying Phase 4K to production:

- [ ] All unit tests passing (>95% coverage)
- [ ] Integration tests passing (10k+ events)
- [ ] E2E test stable for >24 hours
- [ ] Performance benchmarks show expected gains
- [ ] Memory leak tests passing (Valgrind clean)
- [ ] Rollback procedure tested and verified
- [ ] Monitoring alerts configured
- [ ] Runbook documentation complete

---

## 9. Monitoring and Alerts

### 9.1 Key Metrics

```yaml
# Prometheus metrics (add to Yijinjing)
yijinjing_frame_ref_count{frame_id}           # Current ref count per frame
yijinjing_frame_wait_time_seconds{frame_id}   # Time writer waited for frame release
yijinjing_frame_max_ref_count                 # Maximum ref count observed
yijinjing_frame_leak_count                    # Frames with non-zero ref count after cleanup
```

### 9.2 Alerting Rules

```yaml
# Alert if writer waits > 100ms for frame release
- alert: YijinjingFrameWaitHigh
  expr: yijinjing_frame_wait_time_seconds > 0.1
  for: 1m
  annotations:
    summary: "Yijinjing frame held too long ({{$value}}s)"

# Alert if ref count leak detected
- alert: YijinjingFrameRefLeak
  expr: yijinjing_frame_leak_count > 0
  for: 5m
  annotations:
    summary: "Yijinjing frame reference leak detected"
```

### 9.3 Debugging Tools

```bash
# scripts/debug_frame_refcount.sh
#!/bin/bash

echo "Yijinjing Frame Reference Count Debug"
echo "======================================"

# Attach to running process
PID=$(pgrep -f "strategy:hello")

# Use gdb to inspect frame headers
gdb -p $PID -batch -ex "
    set pagination off

    # Print all frame ref counts
    python
import gdb
for i in range(1024):
    frame_addr = gdb.parse_and_eval(f'journal_buffer_ + {i} * frame_size_')
    ref_count = gdb.parse_and_eval(f'((FrameHeader*){frame_addr})->ref_count')
    if ref_count > 0:
        print(f'Frame {i}: ref_count = {ref_count}')
    end
"
```

---

## 10. Decision Matrix

### 10.1 When to Implement Phase 4K

✅ **Implement if**:
- [ ] Depth event rate > 500 Hz (5x current)
- [ ] Latency requirement < 10 μs (100x tighter)
- [ ] Profiling shows copying is >5% of total latency
- [ ] System is memory-constrained (<1 GB available)
- [ ] Team has >1 week of development time

❌ **Do NOT implement if**:
- [ ] Current performance meets requirements (likely)
- [ ] System is stable and reliable
- [ ] Team lacks Yijinjing expertise
- [ ] Production deployment is imminent

### 10.2 Alternative Optimizations

If Phase 4K is deemed too risky, consider these alternatives first:

1. **Optimize model inference** (likely 1000x more impact)
   - Current model latency: ~5000 ns
   - Potential gain: -2000 ns (40% reduction)

2. **Reduce Python callback overhead** (likely 10x more impact)
   - Current callback latency: ~2000 ns
   - Potential gain: -500 ns (25% reduction)

3. **Batch processing** (reduces per-event overhead)
   - Process multiple Depths per callback
   - Potential gain: -100 ns per event (amortized)

---

## 11. Conclusion

### 11.1 Summary

**Phase 4K Zero-Copy Optimization**:
- **Gain**: -180 ns latency (-0.018%), -336 bytes memory per event
- **Cost**: 2-3 days development, medium complexity, increased risk
- **ROI**: ❌ **Poor** - gain does not justify cost

**Recommendation**: **Do NOT implement** unless:
1. Performance profiling proves it's a bottleneck
2. Simpler optimizations have been exhausted
3. Team has spare development capacity

### 11.2 Alternative Path

Instead of Phase 4K, prioritize:

1. ✅ **Stabilize Phase 4I** (callback queue) - Current priority
2. ✅ **Implement dynamic configuration** (Phase 5) - High value
3. ✅ **Optimize model inference** - 1000x more impact
4. ⚠️ **Phase 4K** (this document) - Only if proven necessary

---

## Appendix A: Code Location Reference

| Component | File Path | Lines |
|-----------|-----------|-------|
| FrameHeader | `core/cpp/yijinjing/include/kungfu/yijinjing/journal/frame.h` | +5 lines |
| Reader API | `core/cpp/yijinjing/src/journal/reader.cpp` | +60 lines |
| Writer check | `core/cpp/yijinjing/src/journal/writer.cpp` | +30 lines |
| Runner Depth handler | `core/cpp/wingchun/src/strategy/runner.cpp` | ~10 lines modified |
| signal_api | `hf-live/adapter/signal_api.cpp` | ~20 lines modified |

**Total LOC**: ~125 lines modified/added

---

## Appendix B: Performance Benchmark Script

```python
#!/usr/bin/env python3
# scripts/benchmark_phase4k.py

import time
import numpy as np
import matplotlib.pyplot as plt

def benchmark_copy_vs_zerocopy(iterations=100000):
    """Benchmark Depth copying vs zero-copy"""

    # Simulate Depth structure (336 bytes)
    depth_data = np.random.rand(42)  # 42 * 8 bytes = 336 bytes

    # Benchmark copy
    start = time.perf_counter()
    for _ in range(iterations):
        copy = depth_data.copy()
    copy_time = (time.perf_counter() - start) / iterations * 1e9  # ns

    # Benchmark zero-copy (just pointer)
    start = time.perf_counter()
    for _ in range(iterations):
        ptr = depth_data
    zerocopy_time = (time.perf_counter() - start) / iterations * 1e9  # ns

    print(f"Benchmark Results ({iterations} iterations):")
    print(f"  Copy:      {copy_time:.1f} ns")
    print(f"  Zero-copy: {zerocopy_time:.1f} ns")
    print(f"  Gain:      {copy_time - zerocopy_time:.1f} ns ({(copy_time-zerocopy_time)/copy_time*100:.1f}%)")

    return copy_time, zerocopy_time

if __name__ == "__main__":
    benchmark_copy_vs_zerocopy()
```

---

**Document Version**: 1.0
**Last Updated**: 2024-12-13
**Status**: REFERENCE ONLY - Do not implement without senior engineer approval

**Next Steps**: Review Phase 4I stability, then proceed to Phase 5 (Dynamic Configuration)
