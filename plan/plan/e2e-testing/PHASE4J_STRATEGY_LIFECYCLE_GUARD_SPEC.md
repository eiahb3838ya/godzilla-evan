# Phase 4J: Strategy Lifecycle Guard Fix Specification

**Version**: 1.0
**Date**: 2024-12-13
**Status**: PENDING APPROVAL
**Target Audience**: Junior Engineer
**Branch Name**: `fix/phase4j-strategy-lifecycle-guard`

---

## 0. Pre-requisites Checklist

Before starting, confirm:

- [ ] You have access to the godzilla-evan repository
- [ ] Docker container `godzilla-dev` is running
- [ ] You understand C++ `shared_ptr`, `weak_ptr`, and RAII
- [ ] You have read the root cause analysis in `debug_hf-live.02-spmc-buffer-data-race.md`
- [ ] Phase 4H SPMC Buffer fix is already applied

**CRITICAL WARNING**:
```
!! DO NOT MODIFY ANY FILES IN ref/hf-stock-live-demo-main/ !!
!! The ref project is READ-ONLY reference code !!
```

---

## 1. Overview

### 1.1 What We're Fixing

**Problem**: `ModelResultScanThread` 在背景線程調用 `send_callback_`，當 Strategy 對象正在被銷毀時，會導致 "pure virtual method called" 崩潰。

**Solution**: 使用 `shared_ptr` + `weak_ptr` 模式，確保回調執行期間 Strategy 對象不會被銷毀。如果 Strategy 已被銷毀，回調會被安全跳過。

**Symptom**: Crashes with "pure virtual method called" when Strategy is destroyed during callback.

### 1.2 Scope of Changes

| File | Action |
|------|--------|
| `core/cpp/wingchun/include/kungfu/wingchun/strategy/strategy.h` | MODIFY - add shared_ptr support |
| `core/cpp/wingchun/src/strategy/runner.h` | MODIFY - use weak_ptr for callback |
| `core/cpp/wingchun/src/strategy/runner.cpp` | MODIFY - lifecycle guard in callback |
| `hf-live/*` | NO CHANGES |
| `ref/*` | DO NOT TOUCH |

### 1.3 Key Difference from Phase 4I

| Aspect | Phase 4I (Queue) | Phase 4J (Lifecycle Guard) |
|--------|-----------------|---------------------------|
| Changes hf-live | ✅ Yes | ❌ No |
| Changes Godzilla | ✅ Yes (minimal) | ✅ Yes (more) |
| Callback thread | Main thread | Background thread (guarded) |
| Latency | Slightly higher | No change |
| Complexity | Medium | Medium-High |

---

## 2. Architecture Diagram

### 2.1 Before (Problem)

```
Background Thread                    Main Thread
────────────────                    ───────────
send_callback_()
      │
      ▼
Runner::on_factor_callback()
      │
      ▼
strategy->on_factor()  ◄────────────  ~Strategy() running!
      │                                    │
      ❌ Race condition!                   │
```

### 2.2 After (Solution)

```
Background Thread                    Main Thread
────────────────                    ───────────
send_callback_()
      │
      ▼
Runner::on_factor_callback()
      │
      ▼
weak_ptr.lock()  ──────────────►  Returns nullptr if destroyed
      │
      ├── nullptr? ──► Skip callback (safe)
      │
      └── shared_ptr valid? ──► Execute on_factor()
                                      │
                                ✅ Strategy kept alive during call!
```

---

## 3. Step-by-Step Modification Guide

### 3.1 Create Git Branch

```bash
cd /home/huyifan/projects/godzilla-evan
git checkout -b fix/phase4j-strategy-lifecycle-guard
```

### 3.2 Modification #1: Add LifecycleGuard to Strategy

**File**: `core/cpp/wingchun/include/kungfu/wingchun/strategy/strategy.h`

**Find the Strategy class definition** and add the following:

```cpp
// Find the Strategy class, add these members:

class Strategy {
public:
    // ... existing public methods ...

    // Lifecycle guard: returns a shared_ptr that keeps Strategy alive
    // Returns nullptr if Strategy is being destroyed
    std::shared_ptr<Strategy> get_lifecycle_guard() {
        return lifecycle_guard_.lock();
    }

    // Enable shared_from_this pattern
    void enable_lifecycle_guard(std::shared_ptr<Strategy> self) {
        lifecycle_guard_ = self;
        std::cerr << "[Strategy] Lifecycle guard enabled" << std::endl;
    }

private:
    // ... existing private members ...

    // Weak pointer to self, used for lifecycle guard
    std::weak_ptr<Strategy> lifecycle_guard_;
};
```

### 3.3 Modification #2: Store Strategies as shared_ptr

**File**: `core/cpp/wingchun/src/strategy/runner.h`

**Find the strategies_ member** and change it:

Find:
```cpp
std::unordered_map<int, std::unique_ptr<Strategy>> strategies_;
```

Replace with:
```cpp
std::unordered_map<int, std::shared_ptr<Strategy>> strategies_;

// Weak pointers for safe callback access
std::unordered_map<int, std::weak_ptr<Strategy>> strategy_weak_ptrs_;
```

### 3.4 Modification #3: Enable Lifecycle Guard on Strategy Creation

**File**: `core/cpp/wingchun/src/strategy/runner.cpp`

**Find where strategies are created/added** and enable the lifecycle guard:

After a strategy is created and added to strategies_, add:
```cpp
// After: strategies_[id] = std::move(strategy);
// Change to:
strategies_[id] = std::make_shared<Strategy>(/* constructor args */);
strategies_[id]->enable_lifecycle_guard(strategies_[id]);
strategy_weak_ptrs_[id] = strategies_[id];

std::cerr << "[Runner] Strategy " << id << " lifecycle guard enabled" << std::endl;
```

**Note**: The exact location depends on how strategies are created. Search for `strategies_[` or `strategies_.emplace`.

### 3.5 Modification #4: Use Lifecycle Guard in Callback

**File**: `core/cpp/wingchun/src/strategy/runner.cpp`

**Find the on_factor_callback function** and modify it:

Find:
```cpp
void Runner::on_factor_callback(const char* symbol, long long timestamp, const double* values, int count)
{
    std::cerr << "[FACTOR] 🎊 Received factor for " << symbol
              << " @ " << timestamp << " (count=" << count << ")" << std::endl;

    SPDLOG_DEBUG("Received factor for {} @ {}: count={}", symbol, timestamp, count);

    // 調用所有策略的 on_factor 回調
    std::vector<double> factor_values(values, values + count);
    for (auto& [id, strategy] : strategies_)
    {
        std::cerr << "[FACTOR] Calling strategy on_factor for strategy_id=" << id << std::endl;
        context_->set_current_strategy_index(id);
        strategy->on_factor(context_, std::string(symbol), timestamp, factor_values);
    }

    std::cerr << "[FACTOR] ✅ on_factor completed" << std::endl;
}
```

Replace with:
```cpp
void Runner::on_factor_callback(const char* symbol, long long timestamp, const double* values, int count)
{
    std::cerr << "[FACTOR] 🎊 Received factor for " << symbol
              << " @ " << timestamp << " (count=" << count << ")" << std::endl;

    SPDLOG_DEBUG("Received factor for {} @ {}: count={}", symbol, timestamp, count);

    // 調用所有策略的 on_factor 回調 (使用 lifecycle guard)
    std::vector<double> factor_values(values, values + count);

    for (auto& [id, weak_ptr] : strategy_weak_ptrs_)
    {
        // 嘗試獲取 shared_ptr (lifecycle guard)
        auto strategy = weak_ptr.lock();

        if (!strategy) {
            // Strategy 已被銷毀，安全跳過
            std::cerr << "[FACTOR] ⚠️ Strategy " << id << " already destroyed, skipping" << std::endl;
            continue;
        }

        std::cerr << "[FACTOR] Calling strategy on_factor for strategy_id=" << id
                  << " (lifecycle guard active, ref_count=" << strategy.use_count() << ")" << std::endl;

        context_->set_current_strategy_index(id);
        strategy->on_factor(context_, std::string(symbol), timestamp, factor_values);

        // strategy shared_ptr 在這裡析構，如果這是最後一個引用，Strategy 會被銷毀
        // 但 on_factor() 已經安全執行完畢
    }

    std::cerr << "[FACTOR] ✅ on_factor completed" << std::endl;
}
```

### 3.6 Modification #5: Update Other Strategy Access Points

**Search for all places that access `strategies_`** and update them to handle shared_ptr:

```bash
grep -n "strategies_\[" core/cpp/wingchun/src/strategy/runner.cpp
grep -n "strategies_\." core/cpp/wingchun/src/strategy/runner.cpp
```

For each access point, ensure:
1. If iterating: `for (auto& [id, strategy] : strategies_)` works with shared_ptr
2. If accessing by id: `strategies_[id]->method()` works with shared_ptr
3. If checking existence: `strategies_.count(id)` unchanged

Most code should work without changes since `shared_ptr` has similar API to raw pointer.

---

## 4. Verification Steps

### 4.1 Syntax Check

```bash
cd /home/huyifan/projects/godzilla-evan/core

# Quick syntax check of modified header
g++ -std=c++17 -fsyntax-only -I./cpp/wingchun/include cpp/wingchun/include/kungfu/wingchun/strategy/strategy.h
```

### 4.2 Full Build (Godzilla core only)

```bash
cd /home/huyifan/projects/godzilla-evan/core
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Expected Result**: Build completes with no errors

### 4.3 Copy to Container

```bash
# Copy updated pywingchun module
docker cp /home/huyifan/projects/godzilla-evan/core/build/cpp/wingchun/pywingchun.cpython-38-x86_64-linux-gnu.so \
    godzilla-dev:/app/core/python/pywingchun.cpython-38-x86_64-linux-gnu.so
```

---

## 5. Testing Protocol

### 5.1 Pre-Test Cleanup

```bash
docker exec godzilla-dev pm2 delete all 2>/dev/null || true
docker exec godzilla-dev rm -rf /shared/kungfu/runtime/*
```

### 5.2 Start Services

```bash
docker exec godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"
```

### 5.3 Monitor for 5 Minutes

```bash
# In terminal 1: Watch PM2 status
watch -n 5 'docker exec godzilla-dev pm2 list'

# In terminal 2: Watch logs for errors
docker exec godzilla-dev pm2 logs strategy:hello --lines 100
```

### 5.4 Success Criteria

| Metric | Pass Condition |
|--------|----------------|
| PM2 Restart Count | 0 (no restarts) |
| Runtime | > 5 minutes without crash |
| Error Logs | No "pure virtual method called" |
| Error Logs | No "bus error" |
| Log Output | Should see "lifecycle guard active" messages |
| Log Output | May see "Strategy X already destroyed, skipping" (if shutdown) |

### 5.5 Failure Actions

If tests fail:

1. **Collect crash logs**:
   ```bash
   docker exec godzilla-dev pm2 logs strategy:hello --lines 500 > crash_logs_phase4j.txt
   ```

2. **Check for compilation errors** in Godzilla build

3. **Verify shared_ptr conversion** is complete (no mix of unique_ptr and shared_ptr)

---

## 6. Rollback Procedure

```bash
# 1. Stop all services
docker exec godzilla-dev pm2 delete all

# 2. Switch back to main branch
cd /home/huyifan/projects/godzilla-evan
git checkout main

# 3. Rebuild Godzilla core
cd core && mkdir -p build && cd build && cmake .. && make -j$(nproc)

# 4. Redeploy
docker cp build/cpp/wingchun/pywingchun.cpython-38-x86_64-linux-gnu.so \
    godzilla-dev:/app/core/python/
```

---

## 7. Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Spec Author | Claude (Linus) | 2024-12-13 | |
| Reviewer | | | |
| Executor | | | |
| Tester | | | |

---

## Appendix A: Why This Fix Works

### A.1 shared_ptr + weak_ptr Pattern

```cpp
// 主線程持有 shared_ptr
std::shared_ptr<Strategy> strategy = ...;  // ref_count = 1

// 回調持有 weak_ptr
std::weak_ptr<Strategy> weak = strategy;   // ref_count = 1 (weak 不增加)

// 背景線程嘗試訪問
auto locked = weak.lock();                 // ref_count = 2 if alive, nullptr if dead

if (locked) {
    // Strategy 保證在此期間存活
    locked->on_factor(...);
}
// ref_count = 1 (locked 析構)

// 當主線程銷毀 strategy
strategy.reset();                          // ref_count = 0, Strategy 析構

// 之後背景線程嘗試訪問
auto locked2 = weak.lock();                // 返回 nullptr
if (!locked2) {
    // 安全跳過，不會崩潰
}
```

### A.2 Why This Is Better Than Mutex

| Aspect | Mutex | shared_ptr/weak_ptr |
|--------|-------|---------------------|
| 阻塞 | ✅ 會阻塞 | ❌ 不阻塞 |
| 死鎖風險 | ✅ 有風險 | ❌ 無風險 |
| 已銷毀對象 | 需要額外檢查 | 自動處理 |
| 性能影響 | 中 | 低 |
| 代碼複雜度 | 低 | 中 |

### A.3 Reference Count Lifecycle

```
時間軸:
────────────────────────────────────────────►

主線程:     [create]  [use]  [use]  [destroy]
              │         │      │        │
ref_count:    1         1      1        0

背景線程:         [lock]  [use]  [unlock]
                    │       │        │
ref_count:          2       2        1

安全情況: 背景線程在主線程 destroy 之前 unlock
危險情況: 背景線程在主線程 destroy 時正在 use
          → shared_ptr 保證 use 完成後才真正 destroy!
```

---

## Appendix B: Comparison with Phase 4I

### B.1 When to Prefer Phase 4I (Queue)

- 需要確保回調在主線程執行
- 回調可能修改主線程狀態
- 希望最小化 Godzilla 核心修改

### B.2 When to Prefer Phase 4J (Lifecycle Guard)

- 希望保持低延遲
- 回調是唯讀的 (如 on_factor)
- 已經使用 shared_ptr 管理 Strategy

### B.3 Recommendation

**建議先測試 Phase 4I (Queue)**，因為:
1. 改動範圍更明確
2. 完全避免跨線程調用虛函數
3. 更符合單線程策略模型

如果 Phase 4I 因延遲問題不可接受，再考慮 Phase 4J。
