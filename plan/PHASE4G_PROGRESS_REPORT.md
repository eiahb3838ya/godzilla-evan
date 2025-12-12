# Phase 4G: 懸空指針修復進度報告

**日期**: 2025-12-12
**狀態**: 🟡 部分修復,核心問題仍存在

---

## 📋 執行摘要

### ✅ 已修復
1. **signal_sender.h:59-60** - 添加 `symbol_copy` 和 `values_copy`
2. 編譯成功,修復已部署

### ❌ 核心問題仍存在
- **現象**: `double free or corruption (!prev)` + `free(): invalid pointer`
- **發生位置**: Python `on_factor` 回調執行時
- **崩潰頻率**: 每次 on_factor 回調後必然崩潰
- **Restart count**: 5 次 (60秒測試期間)

---

## 🔍 問題分析

### 修復歷程

#### 修復 1: signal_api.cpp 懸空指針 (已失敗)
**原始猜測**:
```cpp
// signal_api.cpp:57-66
std::vector<double> predictions(data_with_metadata.begin() + 11, ...);
SignalSender::GetInstance().Send(symbol.c_str(), timestamp,
                                 predictions.data(), predictions.size());
// ❌ predictions 析構 → predictions.data() 懸空
```

**修復方案**: 在 `SignalSender::Send()` 中複製 `values`
**結果**: ❌ 問題仍存在

---

#### 修復 2: signal_sender.h symbol 懸空指針 (已失敗)
**發現**:
```cpp
// model_result_scan_thread.h:95, 117
std::string code = model_output.assets[0];  // 局部變數
SendData(code, ...);
  → send_callback_(symbol, ...)  // 傳遞引用
    → SignalSender::Send(symbol.c_str(), ...)  // C 字串指針
```

**修復方案**:
```cpp
// signal_sender.h:59-60
std::string symbol_copy(symbol ? symbol : "");
std::vector<double> values_copy(values, values + count);
callback_(symbol_copy.c_str(), timestamp, values_copy.data(), count, user_data_);
```

**結果**: ❌ 問題仍存在

---

### 當前問題定位

#### 崩潰時序
```
✅ SignalSender::Send() 成功 (with safe data copy)
✅ [FACTOR] 🎊 Received factor for BTCUSDT
✅ [FACTOR] Calling strategy on_factor for strategy_id=1350253488
❌ double free or corruption (!prev)
❌ free(): invalid pointer
```

#### 問題根源: runner.cpp
**文件**: `core/cpp/wingchun/src/strategy/runner.cpp:220-226`

```cpp
void Runner::on_factor_callback(const char* symbol, long long timestamp,
                                const double* values, int count, void* user_data)
{
    // ... 調試日誌 ...

    // ❌ 關鍵問題: 創建局部 vector
    std::vector<double> factor_values(values, values + count);  // Line 220

    for (auto& [id, strategy] : strategies_)
    {
        std::cerr << "[FACTOR] Calling strategy on_factor for strategy_id=" << id << std::endl;
        context_->set_current_strategy_index(id);
        strategy->on_factor(context_, std::string(symbol), timestamp, factor_values);  // Line 225
        // ❌ factor_values 傳遞給 Python (可能通過 pybind11 綁定)
    }
    // ❌ factor_values 析構 (Line 227)

    std::cerr << "[FACTOR] ✅ on_factor completed" << std::endl;
}
```

#### 記憶體錯誤序列

1. **Line 220**: 創建 `std::vector<double> factor_values`
2. **Line 225**: 調用 `strategy->on_factor(..., factor_values)`
   - Pybind11 將 C++ `std::vector` 轉換為 Python list
   - **可能問題**: Pybind11 可能保存了 `factor_values.data()` 的裸指針而非複製數據
3. **Python 側** ([test_hf_live.py:185](strategies/test_hf_live/test_hf_live.py#L185)):
   ```python
   context.log().info(f"  Values: {values}")  # 使用 values
   ```
4. **Line 227**: `factor_values` 析構 → 底層數據被釋放
5. **Python GC**: 當 Python 嘗試清理 `values` 對象時 → 訪問已釋放的記憶體
6. **❌ 崩潰**: `double free or corruption (!prev)` + `free(): invalid pointer`

---

## 🎯 根本原因

### 核心問題
**Pybind11 綁定層的記憶體管理問題**:
- `strategy->on_factor()` 通過 pybind11 將 C++ `std::vector<double>` 傳遞給 Python
- Pybind11 可能**未正確複製數據**,而是保存了指向臨時對象的指針
- 當 C++ 側的 `factor_values` 析構後,Python 側持有懸空指針

### 為什麼之前的修復無效?
1. **signal_sender.h 的修復**只解決了 `SignalSender → Runner` 的數據傳遞
2. **但 Runner → Python** 的數據傳遞仍然有問題
3. 問題在於 **runner.cpp:220 創建的局部 vector**,而非 SignalSender

---

## 💡 可能的解決方案

### 方案 A: 修改 runner.cpp (侵入性大)
修改 `Runner::on_factor_callback` 以延長 `factor_values` 的生命週期:

```cpp
// 不推薦: 修改 Godzilla 核心代碼

// 選項 1: 使用 std::shared_ptr
auto factor_values = std::make_shared<std::vector<double>>(values, values + count);

// 選項 2: 使用類成員變數 (線程不安全)
factor_values_.assign(values, values + count);

// 選項 3: 顯式複製到 Python (需要修改 pybind11 綁定)
py::list py_values;
for (int i = 0; i < count; ++i) {
    py_values.append(values[i]);
}
```

**優點**: 可能徹底解決問題
**缺點**:
- 需要修改 Godzilla 核心代碼
- 可能影響其他策略
- 違反最小侵入性原則

---

### 方案 B: 修改 Python 策略 (推薦)
確保 Python 側立即複製數據:

```python
def on_factor(context, symbol, timestamp, values):
    # ✅ 立即複製數據到 Python list
    values_copy = list(values)  # 或 values_copy = values[:]

    # 後續使用 values_copy 而非 values
    context.log().info(f"  Values: {values_copy}")

    if len(values_copy) >= 5:
        spread = values_copy[0]
        ...
```

**優點**:
- 不需要修改核心代碼
- 安全且簡單
- 性能影響可忽略 (只有 2-5 個值)

**缺點**:
- 需要修改所有使用 `on_factor` 的策略
- 治標不治本 (如果 pybind11 綁定有問題,其他地方可能也有風險)

---

### 方案 C: 檢查 pybind11 綁定 (最徹底)
檢查 `strategy->on_factor()` 的 pybind11 綁定實現:

1. 查找 Strategy 類的 pybind11 綁定代碼
2. 確認 `on_factor` 方法的參數綁定方式
3. 確保使用 `py::arg("values").noconvert()` 或顯式複製

**需要查找**:
- `core/cpp/wingchun/src/bindings/` 或類似路徑
- 搜索 `PYBIND11_MODULE` 和 `on_factor`

---

## 📊 測試結果

### P0 測試 (60秒)

| 檢查項 | 預期 | 實際 | 狀態 |
|--------|------|------|------|
| 無記憶體錯誤 | 0 | 0 | ❌ (重啟前有錯誤) |
| Restart count = 0 | 0 | 5 | ❌ |
| 修復生效 | >0 | 1 | ✅ |
| on_factor 回調 | >0 | 1 | ✅ |
| 完整數據流 | ✅ | ✅ | ✅ |

### 關鍵觀察
1. ✅ SignalSender 修復生效 ("with safe data copy")
2. ✅ Python 成功接收 factor 數據
3. ❌ 每次 on_factor 回調後必然崩潰
4. ❌ 60秒內崩潰 5 次 → 平均 12 秒崩潰一次

### 崩潰模式
```
循環 1: Received factor → double free → 重啟
循環 2: Received factor (2次) → double free → 重啟
循環 3: Received factor → double free → 重啟
循環 4: Received factor (2次) → double free → 重啟
循環 5: Received factor → double free → (測試結束)
```

---

## 🚀 下一步建議

### 優先級 P0: 快速驗證方案 B
1. 修改 `strategies/test_hf_live/test_hf_live.py:171`
2. 在 `on_factor` 第一行添加: `values = list(values)`
3. 重新編譯並測試 60 秒
4. **預期**: 如果問題消失 → 確認是 pybind11 綁定問題

### 優先級 P1: 深入調查
1. 查找 pybind11 綁定代碼
2. 檢查 `on_factor` 的綁定實現
3. 確認是否需要修改綁定層

### 優先級 P2: 長期修復
1. 如果方案 B 有效 → 文檔化最佳實踐,要求所有策略複製 `values`
2. 如果方案 B 無效 → 考慮方案 A 或 C

---

## 📁 相關文件

### 已修改
- [hf-live/_comm/signal_sender.h](hf-live/_comm/signal_sender.h#L59-L60) - 添加 symbol_copy 和 values_copy

### 需要調查
- [core/cpp/wingchun/src/strategy/runner.cpp](core/cpp/wingchun/src/strategy/runner.cpp#L220-L226) - 問題根源
- [strategies/test_hf_live/test_hf_live.py](strategies/test_hf_live/test_hf_live.py#L171) - Python 策略
- `core/cpp/wingchun/src/bindings/` - Pybind11 綁定 (待確認路徑)

### 日誌位置
- 容器內: `/root/.pm2/logs/strategy-test-hf-live-error.log`
- PM2 狀態: `docker exec godzilla-dev pm2 list`

---

## 🔧 快速複現

```bash
# 1. 進入容器
docker exec -it godzilla-dev bash

# 2. 清理並重啟
cd /app/scripts/binance_test
bash graceful_shutdown.sh
./run.sh start
sleep 5
pm2 start /app/scripts/test_hf_live/strategy.json

# 3. 等待 60 秒
sleep 60

# 4. 檢查結果
pm2 list | grep strategy  # 查看 restart count (預期: >0)
tail -100 /root/.pm2/logs/strategy-test-hf-live-error.log | grep "double free"  # 預期: 有輸出
```

---

## 📈 修復影響評估

### 性能影響
- Symbol 複製: ~10ns (可忽略)
- Values 複製: ~30ns for 5 values (可忽略)
- **總影響**: < 0.01% CPU

### 穩定性影響
- ✅ 修復了 SignalSender 層的懸空指針
- ❌ **但核心問題仍未解決**
- 🟡 系統仍不穩定,無法用於生產環境

---

## 結論

當前修復**只解決了表面問題**,核心的記憶體管理問題在 **runner.cpp 和 pybind11 綁定層**。

**建議**: 優先測試方案 B (Python 側複製數據),這是最安全且侵入性最小的方案。如果有效,可作為短期解決方案;長期需要徹底檢查 pybind11 綁定實現。
