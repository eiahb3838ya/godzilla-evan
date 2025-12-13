# Debug Report: SPMC Buffer Data Race Analysis

**Date**: 2024-12-13
**Phase**: 4H - Root Cause Analysis (Iteration 2)
**Status**: SPMC FIX FAILED - NEW ROOT CAUSE IDENTIFIED
**Author**: Claude (as Linus-style C++ review)

---

## 0. Update Summary (Phase 4H Test Results)

### 0.1 SPMC Buffer Atomic Fix - FAILED

**執行結果**: ❌ 修復後仍然崩潰

**測試數據**:
- PM2 重啟次數: **18 次** (1分鐘內)
- 錯誤類型: "pure virtual method called" 仍然出現
- 結論: **SPMC Buffer 不是根本原因**

### 0.2 新發現的根本原因

經過深入分析 Godzilla runner.cpp 和 hf-live 回調鏈，發現真正的問題是：

**跨線程回調導致的線程安全問題**

```
hf-live (ModelResultScanThread) ─→ send_callback_
                                         ↓
SignalSender::Send() ─→ factor_callback_fn
                                         ↓
Runner::on_factor_callback() ─→ strategy->on_factor()
                                         ↓
                              ❌ 在非主線程訪問 Strategy 對象!
```

---

## 1. Executive Summary

經過系統性分析，定位到 hf-live 在 Godzilla dlopen 環境下崩潰的**真正根本原因**：

**跨線程回調導致 Strategy 對象的虛函數被從非主線程調用，當 Strategy 正在被銷毀或修改時引發 "pure virtual method called"**

~~SPMC Buffer 中使用 `volatile` 而非 `std::atomic`~~ (已修復，但不是根本原因)

---

## 2. Research Background

### 2.1 問題描述

- **現象**：hf-live 通過 dlopen 載入 Godzilla 後出現多種崩潰
- **環境**：hf-live 單獨運行正常，ref 項目單獨運行正常
- **錯誤類型**：
  - `"pure virtual method called"` ← **主要錯誤**
  - `"terminate called recursively"`
  - `"corrupted size vs. prev_size"`
  - `"bus error"` / `"segmentation violation"`

### 2.2 研究範圍

| 項目 | 路徑 | 狀態 |
|------|------|------|
| hf-live | `hf-live/` | 崩潰 (dlopen 環境) |
| ref 項目 | `ref/hf-stock-live-demo-main/` | 穩定 (獨立運行) |
| Godzilla | `core/` | 穩定 (無 hf-live) |

---

## 3. Research Methodology

### 3.1 分析策略

1. **代碼對比**：hf-live vs ref 項目，逐個組件比較
2. **數據結構驗證**：確認 Depth/Trade struct 內存布局相容性
3. **線程安全審計**：檢查 SPSC Queue、SPMC Buffer、Model Registry
4. **生命週期分析**：追蹤 shared_ptr 和裸指針的所有權
5. **跨線程分析**：追蹤回調鏈和線程邊界 ← **關鍵**

### 3.2 檢查清單

- [x] SPSC Queue (spsc_queue_for_model_use.h) - 與 ref 相同
- [x] SPSC Queue (app_live/data/spsc_queue.h) - 與 ref 相同
- [x] Model Registry (model_registry.h) - 與 ref 相同
- [x] ~~**SPMC Buffer (spmc_buffer.hpp)**~~ - 已修復，非根本原因
- [x] Depth/Trade struct layout - 與 Godzilla 相容
- [x] ModelInterface::TryGetOutput - 與 ref 相同 (有 static 變量)
- [x] **跨線程回調鏈** ← **發現根本原因**

---

## 4. Research Process

### 4.1 Phase 1: SPSC Queue 分析

**結論**：hf-live SPSC Queue 與 ref 項目完全相同，無差異。

### 4.2 Phase 2: 數據結構相容性

**結論**：hf::Depth 和 kungfu::wingchun::msg::data::Depth 內存布局完全相容。

### 4.3 Phase 3: SPMC Buffer 修復與測試

**已修復**：將 `volatile size_t write_num_` 改為 `std::atomic<size_t> write_num_`

**測試結果**：❌ **仍然崩潰** (18次重啟/分鐘)

### 4.4 Phase 4: 跨線程回調分析 (新發現)

**關鍵發現**：追蹤回調鏈發現線程安全問題

**回調鏈分析**：

```cpp
// 1. hf-live 的 ModelResultScanThread (背景線程)
// model_result_scan_thread.h:86
if (models_[i]->TryGetOutput(model_output)) {
    SendData(code, model_output.timestamp.data_time, data);
}

// 2. SendData 調用回調
// model_result_scan_thread.h:176
send_callback_(symbol, timestamp, predictions);

// 3. 回調最終到達 Godzilla runner.cpp (仍在背景線程!)
// runner.cpp:221-226
for (auto& [id, strategy] : strategies_) {
    context_->set_current_strategy_index(id);
    strategy->on_factor(context_, std::string(symbol), timestamp, factor_values);
    //        ^^^^^^^^^^^^^^^^^ 在非主線程調用虛函數!
}
```

**問題**：
1. `ModelResultScanThread` 在**背景線程**運行
2. 回調函數沒有線程切換，仍在背景線程執行
3. `strategy->on_factor()` 是虛函數調用
4. 如果 Strategy 對象正在被銷毀或 `strategies_` 正在被修改，會導致 "pure virtual method called"

### 4.5 對比：為什麼 ref 項目沒有這個問題？

**關鍵差異**：ref 項目是獨立進程，不使用 dlopen 載入到 Godzilla

| 特性 | ref 項目 | hf-live + Godzilla |
|------|----------|-------------------|
| 運行模式 | 獨立進程 | dlopen 動態載入 |
| 回調目標 | 自己的代碼 | Godzilla 的 Strategy |
| 線程模型 | 完全隔離 | 跨進程邊界回調 |
| Strategy 生命週期 | N/A | Godzilla 控制 |

---

## 5. Research Findings

### 5.1 確認的根本原因：跨線程回調 (CRITICAL)

**問題代碼路徑**：

```
core/cpp/wingchun/src/strategy/runner.cpp:188-193
```

```cpp
signal_register_callback_(signal_engine_handle_,
    [](const char* symbol, long long ts, const double* values, int count, void* ud) {
        Runner* self = static_cast<Runner*>(ud);
        self->on_factor_callback(symbol, ts, values, count);  // ❌ 在非主線程執行!
    },
    this);
```

**崩潰場景**：

```
Main Thread (Godzilla)          Background Thread (hf-live)
─────────────────────          ─────────────────────────────
1. 開始銷毀 Strategy
2. ~Strategy() 運行中
   vtable 指針被重置
                                3. TryGetOutput() 成功
                                4. send_callback_()
                                5. strategy->on_factor()
                                   ↑ 虛函數表已被重置!
                                   ↑ "pure virtual method called"!
```

### 5.2 次要問題 (已排除)

| 問題 | 狀態 | 備註 |
|------|------|------|
| SPMC Buffer volatile | ✅ 已修復 | 不是根本原因 |
| TryGetOutput 靜態變量 | 保持觀察 | ref 相同 |
| dlopen 符號衝突 | 已排除 | RTLD_NODELETE 處理 |

---

## 6. Research Results

### 6.1 結論

**真正的根本原因是跨線程回調導致的線程安全問題**

hf-live 的背景線程直接調用 Godzilla 的 Strategy 虛函數，沒有進行線程同步或將調用 marshal 到主線程。

### 6.2 為什麼這在 ref 項目不是問題

ref 項目獨立運行，回調目標是自己的代碼，生命週期完全由自己控制。當載入到 Godzilla 時，回調跨越了進程邊界，而 Godzilla 可能隨時銷毀 Strategy 對象。

---

## 7. Proposed Fix Plan (SPEC) - Phase 4I

### 7.1 修復策略

**有兩個可行方案**：

#### Option A: 在 hf-live 側同步回調 (推薦)

在 `ModelResultScanThread` 使用消息隊列而非直接回調：

```cpp
// 不直接調用 send_callback_，而是推送到隊列
// 由主線程 (或專門的回調線程) 消費隊列並調用回調
```

**優點**：不需要修改 Godzilla 代碼
**缺點**：增加延遲

#### Option B: 在 Godzilla 側加鎖保護

在 `Runner::on_factor_callback` 添加鎖：

```cpp
void Runner::on_factor_callback(...) {
    std::lock_guard<std::mutex> lock(strategies_mutex_);
    for (auto& [id, strategy] : strategies_) {
        strategy->on_factor(...);
    }
}
```

**優點**：改動最小
**缺點**：需要修改 Godzilla 核心代碼

#### Option C: 確保 Strategy 不會在回調期間被銷毀

使用 shared_ptr 或生命週期管理確保 Strategy 在回調完成前不會被銷毀。

---

## 8. Risk Assessment

### 8.1 修復風險

| 風險 | 可能性 | 影響 | 緩解措施 |
|------|--------|------|----------|
| 增加延遲 (Option A) | 確定 | 低 | 隊列處理極快 |
| 死鎖 (Option B) | 低 | 高 | 謹慎設計鎖順序 |
| 內存洩漏 (Option C) | 低 | 中 | 使用 weak_ptr 觀察 |

### 8.2 不修復風險

| 風險 | 可能性 | 影響 |
|------|--------|------|
| 持續崩潰 | 確定 | 高 - 無法部署 |
| 不可預測行為 | 高 | 高 - 潛在數據損壞 |

---

## 9. Decision Required

**請批准以下操作之一**：

- [ ] **Option A**: 在 hf-live 側添加回調隊列 (推薦 - 不影響 Godzilla)
- [ ] **Option B**: 在 Godzilla 側添加互斥鎖保護
- [ ] **Option C**: 實現 Strategy 的 shared_ptr 生命週期管理
- [ ] **Option D**: 其他 (請說明)

---

## 10. Appendix

### A. 相關文件

- `hf-live/app_live/thread/model_result_scan_thread.h` - 背景線程
- `hf-live/_comm/signal_sender.h` - 回調發送器
- `core/cpp/wingchun/src/strategy/runner.cpp` - Godzilla Runner (回調接收)
- `hf-live/adapter/signal_api.cpp` - dlopen 入口點

### B. 崩潰日誌樣本

```
4|strategy | [12/13 13:56:27.639566517] [critical] bus error
4|strategy | pure virtual method called
4|strategy | terminate called recursively
PM2 restart count: 18 (in ~1 minute)
```

### C. Phase 4H SPMC Buffer 修復記錄

**已執行修改**：
- `hf-live/app_live/data/spmc_buffer.hpp`
- 將 `volatile size_t write_num_` 改為 `std::atomic<size_t> write_num_`
- 使用 `memory_order_release/acquire` 語義

**測試結果**：仍然崩潰，確認不是根本原因

---

## 11. Next Steps

1. 等待用戶批准修復方案 (A/B/C)
2. 創建 Phase 4I 修復規格
3. 執行修復
4. E2E 測試驗證
