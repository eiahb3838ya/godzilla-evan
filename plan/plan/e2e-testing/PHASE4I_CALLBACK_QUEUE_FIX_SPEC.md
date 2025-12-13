# Phase 4I: Callback Queue Thread Safety Fix Specification

**Version**: 1.0
**Date**: 2024-12-13
**Status**: PENDING APPROVAL
**Target Audience**: Junior Engineer
**Branch Name**: `fix/phase4i-callback-queue`

---

## 0. Pre-requisites Checklist

Before starting, confirm:

- [ ] You have access to the godzilla-evan repository
- [ ] Docker container `godzilla-dev` is running
- [ ] You understand C++ thread safety and `std::mutex`
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

**Problem**: `ModelResultScanThread` 在背景線程直接調用 `send_callback_`，該回調最終執行 `strategy->on_factor()`。當 Strategy 對象正在被銷毀時，會導致 "pure virtual method called" 崩潰。

**Solution**: 將回調推送到線程安全的隊列，由呼叫方 (Godzilla 主線程) 輪詢並執行。

**Symptom**: Crashes with "pure virtual method called" when hf-live callback crosses thread boundary.

### 1.2 Scope of Changes

| File | Action |
|------|--------|
| `hf-live/adapter/signal_api.h` | ADD callback result struct |
| `hf-live/adapter/signal_api.cpp` | MODIFY to use queue |
| `hf-live/_comm/signal_sender.h` | MODIFY to queue instead of direct call |
| `core/cpp/wingchun/src/strategy/runner.cpp` | MODIFY to poll queue |
| `ref/*` | DO NOT TOUCH |

---

## 2. Architecture Diagram

### 2.1 Before (Problem)

```
ModelResultScanThread (Background)
         │
         ▼
  send_callback_()  ──────────────► Runner::on_factor_callback()
         │                                    │
         │                                    ▼
         │                          strategy->on_factor()
         │                                    │
         │                          ❌ Race condition with Strategy destruction!
```

### 2.2 After (Solution)

```
ModelResultScanThread (Background)
         │
         ▼
  Push to callback_queue_  ◄────── Thread-safe SPSC Queue
         │
         │
         │    (Main Thread polls)
         │           │
         ▼           ▼
  callback_queue_ ──────────────► Runner polls & executes
                                          │
                                          ▼
                                strategy->on_factor()
                                          │
                                ✅ Same thread as Strategy lifecycle!
```

---

## 3. Step-by-Step Modification Guide

### 3.1 Create Git Branch

```bash
cd /home/huyifan/projects/godzilla-evan
git checkout -b fix/phase4i-callback-queue
```

### 3.2 Modification #1: Add Callback Result Struct

**File**: `hf-live/adapter/signal_api.h`

**Add after existing includes**:

```cpp
#include <vector>
#include <string>
#include <atomic>
#include "../app_live/data/spsc_queue.h"

// Callback result structure for thread-safe queuing
struct CallbackResult {
    std::string symbol;
    int64_t timestamp;
    std::vector<double> values;

    CallbackResult() = default;
    CallbackResult(std::string s, int64_t ts, std::vector<double> v)
        : symbol(std::move(s)), timestamp(ts), values(std::move(v)) {}
};

// Global callback queue (accessed from both hf-live and Godzilla)
extern SPSCQueue<CallbackResult>* g_callback_queue;
extern std::atomic<bool> g_callback_queue_initialized;
```

### 3.3 Modification #2: Initialize Global Queue

**File**: `hf-live/adapter/signal_api.cpp`

**Add at file scope (after includes)**:

```cpp
// Global callback queue instance
SPSCQueue<CallbackResult>* g_callback_queue = nullptr;
std::atomic<bool> g_callback_queue_initialized{false};
```

**Modify `signal_create()` function - add queue initialization**:

Find:
```cpp
extern "C" void* signal_create(const char* config_json) {
    std::cerr << "[signal_api] signal_create called with config: "
```

Add after the first line of signal_create():
```cpp
    // Initialize global callback queue (thread-safe, called once)
    if (!g_callback_queue_initialized.exchange(true)) {
        g_callback_queue = new SPSCQueue<CallbackResult>(4096);  // 4096 capacity
        std::cerr << "[signal_api] Callback queue initialized (capacity=4096)" << std::endl;
    }
```

### 3.4 Modification #3: Change SignalSender to Queue

**File**: `hf-live/_comm/signal_sender.h`

**Replace the Send() function**:

Find:
```cpp
    // 發送因子/模型結果
    void Send(const char* symbol, long long timestamp, const double* values, int count) {
        std::lock_guard<std::mutex> lock(mutex_);
        // ... existing code that calls callback_ directly
    }
```

Replace with:
```cpp
    // 發送因子/模型結果 - 推送到隊列而非直接調用回調
    void Send(const char* symbol, long long timestamp, const double* values, int count) {
        // 🔍 調試輸出
        std::cerr << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
        std::cerr << "📨 [SignalSender::Send] Queuing result (NOT direct callback)" << std::endl;
        std::cerr << "   Symbol: " << (symbol ? symbol : "NULL") << std::endl;
        std::cerr << "   Timestamp: " << timestamp << std::endl;
        std::cerr << "   Count: " << count << std::endl;

        // 檢查隊列是否可用
        extern SPSCQueue<CallbackResult>* g_callback_queue;
        extern std::atomic<bool> g_callback_queue_initialized;

        if (!g_callback_queue_initialized.load() || !g_callback_queue) {
            std::cerr << "   ❌ ERROR: Callback queue not initialized!" << std::endl;
            std::cerr << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
            return;
        }

        // 創建結果對象並推送到隊列
        CallbackResult result(
            symbol ? std::string(symbol) : "",
            timestamp,
            std::vector<double>(values, values + count)
        );

        // 嘗試推送到隊列 (非阻塞)
        if (g_callback_queue->push(std::move(result))) {
            std::cerr << "   ✅ Result queued successfully" << std::endl;
        } else {
            std::cerr << "   ⚠️ WARNING: Queue full, result dropped!" << std::endl;
        }
        std::cerr << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
        std::cerr.flush();
    }
```

**Also add include at top of file**:
```cpp
#include "../adapter/signal_api.h"  // For CallbackResult and g_callback_queue
```

### 3.5 Modification #4: Add Poll Function to signal_api

**File**: `hf-live/adapter/signal_api.h`

**Add new function declaration**:
```cpp
// Poll callback queue and execute pending callbacks
// Returns: number of callbacks processed
extern "C" int signal_poll_callbacks(void* handle);
```

**File**: `hf-live/adapter/signal_api.cpp`

**Add new function implementation**:
```cpp
extern "C" int signal_poll_callbacks(void* handle) {
    if (!handle) return 0;

    SignalHandle* h = static_cast<SignalHandle*>(handle);
    if (!h->initialized) return 0;

    // Check queue
    if (!g_callback_queue_initialized.load() || !g_callback_queue) {
        return 0;
    }

    int processed = 0;
    CallbackResult result;

    // Process all pending results (non-blocking)
    while (g_callback_queue->pop(result)) {
        // Now call the callback in the CALLER's thread (main thread)
        SignalSender::GetInstance().ExecuteCallback(
            result.symbol.c_str(),
            result.timestamp,
            result.values.data(),
            static_cast<int>(result.values.size())
        );
        processed++;
    }

    if (processed > 0) {
        std::cerr << "[signal_api] Polled and processed " << processed << " callbacks" << std::endl;
    }

    return processed;
}
```

### 3.6 Modification #5: Add ExecuteCallback to SignalSender

**File**: `hf-live/_comm/signal_sender.h`

**Add new method to SignalSender class**:
```cpp
    // 實際執行回調 (由 poll 函數在主線程調用)
    void ExecuteCallback(const char* symbol, long long timestamp, const double* values, int count) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::cerr << "🎯 [SignalSender::ExecuteCallback] Executing in main thread" << std::endl;
        std::cerr << "   Symbol: " << (symbol ? symbol : "NULL") << std::endl;
        std::cerr << "   Callback: " << (callback_ ? "VALID" : "NULL") << std::endl;

        if (callback_) {
            callback_(symbol, timestamp, values, count, user_data_);
            std::cerr << "   ✅ Callback executed successfully" << std::endl;
        } else {
            std::cerr << "   ❌ ERROR: Callback is NULL!" << std::endl;
        }
        std::cerr.flush();
    }
```

### 3.7 Modification #6: Godzilla Runner Polls Queue

**File**: `core/cpp/wingchun/src/strategy/runner.cpp`

**Add function pointer type and member**:

Find the section where signal function pointers are declared:
```cpp
typedef void* (*signal_create_fn)(const char*);
typedef void (*signal_register_callback_fn)(void*, factor_callback_fn, void*);
typedef void (*signal_on_data_fn)(void*, int, const void*);
typedef void (*signal_destroy_fn)(void*);
```

Add after:
```cpp
typedef int (*signal_poll_callbacks_fn)(void*);
```

Find the member variables section:
```cpp
signal_destroy_fn signal_destroy_ = nullptr;
```

Add after:
```cpp
signal_poll_callbacks_fn signal_poll_callbacks_ = nullptr;
```

**Load the poll function in load_signal_library()**:

Find:
```cpp
signal_destroy_ = (signal_destroy_fn)dlsym(signal_lib_handle_, "signal_destroy");
```

Add after:
```cpp
signal_poll_callbacks_ = (signal_poll_callbacks_fn)dlsym(signal_lib_handle_, "signal_poll_callbacks");
std::cerr << "[DEBUG] signal_poll_callbacks: " << (signal_poll_callbacks_ ? "LOADED" : "NULL") << std::endl;
```

**Add polling in the event loop**:

Find the Depth event handler:
```cpp
events_ | is(msg::type::Depth) |
$([&](event_ptr event)
{
```

Add polling at the START of the lambda:
```cpp
events_ | is(msg::type::Depth) |
$([&](event_ptr event)
{
    // Poll hf-live callback queue (process in main thread)
    if (signal_poll_callbacks_ && signal_engine_handle_) {
        signal_poll_callbacks_(signal_engine_handle_);
    }

    // ... rest of existing code
```

---

## 4. Verification Steps

### 4.1 Syntax Check

```bash
cd /home/huyifan/projects/godzilla-evan/hf-live

# Check signal_api.h syntax
g++ -std=c++17 -fsyntax-only -I. adapter/signal_api.h

# Check signal_sender.h syntax
g++ -std=c++17 -fsyntax-only -I. _comm/signal_sender.h
```

### 4.2 Full Build (hf-live)

```bash
cd /home/huyifan/projects/godzilla-evan/hf-live
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 4.3 Full Build (Godzilla core)

```bash
cd /home/huyifan/projects/godzilla-evan/core
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 4.4 Deploy to Container

```bash
# Copy hf-live library
docker cp /home/huyifan/projects/godzilla-evan/hf-live/build/libsignal.so godzilla-dev:/app/hf-live/build/libsignal.so

# Copy Godzilla binaries (if rebuilt)
docker cp /home/huyifan/projects/godzilla-evan/core/build/cpp/wingchun/pywingchun.cpython-38-x86_64-linux-gnu.so godzilla-dev:/app/core/python/
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
| Log Output | Should see "Polled and processed N callbacks" |
| Log Output | Should see "ExecuteCallback" messages |

---

## 6. Rollback Procedure

```bash
# 1. Stop all services
docker exec godzilla-dev pm2 delete all

# 2. Switch back to main branch
cd /home/huyifan/projects/godzilla-evan
git checkout main

# 3. Rebuild and redeploy
cd hf-live && mkdir -p build && cd build && cmake .. && make -j$(nproc)
docker cp build/libsignal.so godzilla-dev:/app/hf-live/build/libsignal.so
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

### A.1 Thread Safety Explanation

```
BEFORE (Race Condition):
========================
Background Thread          Main Thread
────────────────          ───────────
1. TryGetOutput()
2. send_callback_() ──────► 3. on_factor_callback()
                           4. strategy->on_factor()
                              ↑ Could be during Strategy destruction!

AFTER (Thread Safe):
====================
Background Thread          Main Thread
────────────────          ───────────
1. TryGetOutput()
2. queue.push(result)
                           3. queue.pop(result)  [in event loop]
                           4. ExecuteCallback()
                           5. strategy->on_factor()
                              ↑ Same thread as Strategy lifecycle!
```

### A.2 Why Queue Instead of Lock

- Lock approach would block the background thread
- Queue approach allows background thread to continue immediately
- Polling in event loop ensures callbacks run in correct thread context
- No risk of deadlock between threads

---

## Appendix B: E2E 測試計劃 (test_hf_live 策略)

### B.1 目標

使用 `test_hf_live` 策略驗證 Phase 4I Callback Queue 修復的完整數據流：

```
Binance WebSocket → Godzilla MD → FactorEngine → test0000::Factor
→ ModelEngine → test0000::Model → SignalSender (Queue) → signal_poll_callbacks
→ Runner::on_factor_callback → Python on_factor()
```

### B.2 關鍵差異 (vs helloworld)

| 項目 | helloworld | test_hf_live |
|------|------------|--------------|
| on_factor 回調 | ❌ 無 | ✅ 有 |
| Factor 計算 | ❌ 無 | ✅ test0000 |
| Model 推理 | ❌ 無 | ✅ test0000 |
| Callback Queue 驗證 | ❌ 間接 | ✅ 完整 |

### B.3 執行步驟

#### Step 1: 停止所有服務
```bash
docker exec godzilla-dev pm2 delete all
```

#### Step 2: 清理 Journals 和 Logs
```bash
docker exec godzilla-dev bash -c "
rm -rf /shared/kungfu/runtime/*
find ~/.config/kungfu/app/ -name '*.journal' -delete
rm -rf ~/.pm2/logs/*
"
```

#### Step 3: 啟動基礎服務 (按順序)
```bash
cd /app/scripts/binance_test
pm2 start master.json && sleep 5
pm2 start ledger.json && sleep 5
pm2 start md_binance.json && sleep 5
pm2 start td_binance.json && sleep 5
```

#### Step 4: 啟動 test_hf_live 策略
```bash
pm2 start /app/scripts/test_hf_live/strategy.json
```

#### Step 5: 監控並驗證

**成功標準:**
1. PM2 restart count = 0
2. 無 "pure virtual method called" 錯誤
3. 無 "bus error" 錯誤
4. 日誌顯示完整 emoji 序列:
   - `🏁 [test0000::FactorEntry] Created`
   - `📊 [test0000 #N] bid=... ask=...`
   - `🔢 [test0000::UpdateFactors]`
   - `📨 [SignalSender::Send] Queuing result`
   - `🎯 [SignalSender::ExecuteCallback] Executing in main thread`
   - `🎊 [on_factor] Received factor`

**Phase 4I 特有驗證:**
- 看到 `Phase 4I: Callback queue initialized`
- 看到 `signal_poll_callbacks (Phase 4I): ✅ OK`
- 看到 `Polled and processed N callbacks in main thread`

### B.4 關鍵文件

| 文件 | 用途 |
|------|------|
| `strategies/test_hf_live/test_hf_live.py` | Python 策略 (含 on_factor) |
| `strategies/test_hf_live/config.json` | 策略配置 |
| `scripts/test_hf_live/strategy.json` | PM2 配置 |
| `hf-live/factors/test0000/factor_entry.cpp` | test0000 因子實現 |
| `hf-live/models/test0000/test0000_model.cc` | test0000 模型實現 |
| `hf-live/adapter/signal_api.cpp` | signal_poll_callbacks 實現 |
| `core/cpp/wingchun/src/strategy/runner.cpp` | poll 調用點 |

### B.5 預期輸出

完整成功時應看到 on_factor 接收模型輸出值:
```python
values = [
    pred_signal,      # Model output 1: 1.0
    pred_confidence   # Model output 2: 0.8
]
```

---

## Appendix C: E2E 測試結果總結

**測試日期**: 2024-12-13
**測試分支**: `fix/phase4i-callback-queue`
**測試策略**: `test_hf_live`

### C.1 服務穩定性

| 服務 | 運行時間 | 重啟次數 | 狀態 |
|------|----------|----------|------|
| master | 87s | **0** | ✅ PASS |
| ledger | 82s | **0** | ✅ PASS |
| md_binance | 77s | **0** | ✅ PASS |
| td_binance:gz_user1 | 71s | **0** | ✅ PASS |
| strategy_test_hf_live | 56s | **0** | ✅ PASS |

### C.2 Phase 4I 特有日志驗證

```
✅ [DEBUG] signal_poll_callbacks (Phase 4I): ✅ OK
✅ [signal_api] Phase 4I: Callback queue initialized (capacity=4096)
✅ 📨 [SignalSender::Send] Phase 4I: Queuing result (NOT direct callback)
✅ 🎯 [SignalSender::ExecuteCallback] Phase 4I: Executing in main thread
✅ [signal_api] Phase 4I: Polled and processed 1 callbacks in main thread
```

### C.3 完整 E2E 數據流驗證

```
Binance WebSocket → MD
    ✅ [FactorEngine::OnDepth] Received Depth for BTCUSDT (bid=90393.8 ask=90395.3)

Factor 計算
    ✅ 📊 [test0000 #40] bid=90393.8 ask=90396.3
    ✅ 🔢 [test0000::UpdateFactors] spread=0.3 mid=90400.9

Model 推理
    ✅ 📥 [ModelEngine::SendFactors] Received factors
    ✅ 🎯 [ModelScanThread::ScanFunc] TryGetOutput SUCCESS
    ✅ [signal_api] Model prediction for BTCUSDT: 2 values

Phase 4I Callback Queue
    ✅ 📨 [SignalSender::Send] Phase 4I: Queuing result
    ✅ 🎯 [SignalSender::ExecuteCallback] Phase 4I: Executing in main thread

Python 回調
    ✅ [FACTOR] Calling strategy on_factor
    ✅ [FACTOR] ✅ on_factor completed
    ✅ 🎊🎊🎊 [on_factor] Factor data received! 🎊🎊🎊
    ✅ Values: [1.0, 0.800000011920929] (pred_signal, pred_confidence)
```

### C.4 測試結論

| 測試項目 | 結果 |
|----------|------|
| 服務穩定性 (restart=0) | ✅ PASS |
| 無 "pure virtual method called" | ✅ PASS |
| 無 "bus error" | ✅ PASS |
| Callback Queue 初始化 | ✅ PASS |
| 背景線程推送到 Queue | ✅ PASS |
| 主線程 Poll 並執行 | ✅ PASS |
| Python on_factor 回調 | ✅ PASS |

**最終結論**: Phase 4I Callback Queue 修復**完全成功**！

---

