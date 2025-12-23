# Debug Report: on_factor Callback Failure

**日期**: 2025-12-20
**問題編號**: HF-LIVE-05
**狀態**: 已診斷，待修復

---

## 問題描述

### 症狀
- C++ 層日誌顯示回調被調用：
  ```
  [FACTOR] 🎊 Received factor for BTCUSDT @ <timestamp> (count=2)
  [FACTOR] Calling strategy on_factor for strategy_id=1350253488
  [FACTOR] ✅ on_factor completed
  ```
- 但 Python `on_factor` 函數內部**任何代碼都沒有執行**
  - 第一行的 `print()` 沒有輸出
  - `context.log().info()` 沒有輸出
  - 沒有 `🤖 [LinearModel]` 日誌

### 影響範圍
- Phase 6 全市場數據的模型輸出無法到達 Python 策略層
- 交易信號無法被策略處理

---

## 根因分析

### 問題代碼位置

**文件**: `core/cpp/wingchun/pybind/pybind_wingchun.cpp:216-220`

```cpp
void on_factor(strategy::Context_ptr context, const std::string &symbol,
               long long timestamp, const std::vector<double> &values) override
{
    py::gil_scoped_acquire acquire;  // ⚠️ 問題根源！
    PYBIND11_OVERLOAD(void, strategy::Strategy, on_factor, context, symbol, timestamp, values);
}
```

### 技術分析

**GIL 雙重獲取導致靜默失敗**

執行流程：
```
signal_poll_callbacks (主線程，已持有 GIL)
    ↓
Runner::on_factor_callback (runner.cpp:219-236)
    ↓
strategy->on_factor(...) (runner.cpp:232)
    ↓
PyStrategy::on_factor (pybind_wingchun.cpp:216-220)
    ↓
py::gil_scoped_acquire acquire  ← ⚠️ 在已持有 GIL 的線程中再次獲取！
    ↓
PYBIND11_OVERLOAD 靜默失敗（不抛異常）
    ↓
Python on_factor 未被調用，C++ 繼續執行
    ↓
"[FACTOR] ✅ on_factor completed" 被輸出（誤導性日誌）
```

### 對照組證據

| 回調函數 | 代碼位置 | GIL acquire | 工作狀態 |
|---------|---------|-------------|---------|
| on_depth | 189-190 | ❌ 無 | ✅ 正常 (~500 次/分鐘) |
| on_ticker | 192-193 | ❌ 無 | ✅ 正常 |
| on_trade | 213-214 | ❌ 無 | ✅ 正常 |
| on_order | 201-202 | ❌ 無 | ✅ 正常 |
| on_transaction | 198-199 | ❌ 無 | ✅ 正常 |
| **on_factor** | **216-220** | ⚠️ **有** | ❌ **失效** |

**結論**: `on_factor` 是所有回調中**唯一**有 `py::gil_scoped_acquire` 的。

### 錯誤的設計假設

第 218 行注釋寫道：
> "必須：從 C++ 回調線程調用 Python 需要 GIL"

這個假設是**錯誤的**，因為：

1. **Phase 4I 機制**確保回調通過 SPSC 隊列傳遞到**主線程**執行
2. 主線程已持有 GIL（Python 策略正在運行）
3. 再次獲取 GIL 導致 pybind11 內部狀態混亂

---

## 調查過程

### 1. 初始觀察
- 配置修改為 `"BTCUSDT"` 後，C++ 層開始輸出 `[FACTOR]` 日誌
- 但 Python 層完全沒有反應

### 2. 調試嘗試
- 在 `on_factor` 第一行添加 `print(..., file=sys.stderr, flush=True)`
- 結果：仍然沒有輸出

### 3. 代碼審查
- 對比其他回調的 pybind11 綁定實現
- 發現只有 `on_factor` 有 `py::gil_scoped_acquire`

### 4. 確認根因
- 分析 Phase 4I 回調隊列機制：`signal_poll_callbacks` 在主線程執行
- 確認主線程已持有 GIL

---

## 修復方案

### 方案 1: 移除 GIL 獲取（推薦）

**修改文件**: `core/cpp/wingchun/pybind/pybind_wingchun.cpp`

**修改前**:
```cpp
void on_factor(strategy::Context_ptr context, const std::string &symbol, long long timestamp, const std::vector<double> &values) override
{
    py::gil_scoped_acquire acquire;  // 必須：從 C++ 回調線程調用 Python 需要 GIL
    PYBIND11_OVERLOAD(void, strategy::Strategy, on_factor, context, symbol, timestamp, values);
}
```

**修改後**:
```cpp
void on_factor(strategy::Context_ptr context, const std::string &symbol, long long timestamp, const std::vector<double> &values) override
{
    // Phase 4I 確保此回調在主線程執行，主線程已持有 GIL，無需再次獲取
    PYBIND11_OVERLOAD(void, strategy::Strategy, on_factor, context, symbol, timestamp, values);
}
```

### 編譯命令
```bash
cd /app/core && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) pywingchun
```

---

## 驗證標準

### 成功指標
1. ✅ Python on_factor 日誌出現：`🤖 [LinearModel] BTCUSDT @ ...`
2. ✅ 信號和置信度正常輸出
3. ✅ 其他回調（on_depth, on_trade, on_ticker）繼續正常工作

### 失敗處理
如果修復後仍不工作：
1. 檢查 `.so` 文件更新時間確認編譯成功
2. 檢查 LD_LIBRARY_PATH 是否正確
3. 添加 try-catch 包裝以捕獲異常

---

## 經驗教訓

### 1. pybind11 GIL 規則
- 從 C++ 後台線程調用 Python：需要 `py::gil_scoped_acquire`
- 從已持有 GIL 的主線程調用：**不能再獲取 GIL**

### 2. 回調架構設計
- Phase 4I 的 SPSC 隊列 + 主線程輪詢機制是正確的
- 但需要確保綁定層代碼與此機制一致

### 3. 日誌誤導性
- `[FACTOR] ✅ on_factor completed` 沒有 try-catch
- 未來應添加異常處理和更詳細的狀態日誌

---

## 相關文件

| 文件 | 作用 |
|-----|------|
| `core/cpp/wingchun/pybind/pybind_wingchun.cpp:216-220` | 問題代碼 |
| `core/cpp/wingchun/src/strategy/runner.cpp:219-236` | on_factor_callback |
| `hf-live/app_live/signal_api.cpp:264-293` | signal_poll_callbacks |
| `strategies/test_hf_live/test_hf_live.py:207-294` | Python on_factor |

---

## 參考資料

- [pybind11 GIL 文檔](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil)
- Phase 4I 回調隊列設計：見 `.serena/memories/callback-fix-analysis.md`
