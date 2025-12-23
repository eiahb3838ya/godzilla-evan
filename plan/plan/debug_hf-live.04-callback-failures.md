# Debug Log: Phase 6 Callback Failures (on_factor & on_order)

**日期**: 2025-12-19
**問題等級**: 🔴 Critical
**影響範圍**: Phase 6 完整市場數據管線 + 線性模型
**Git 提交**:
- hf-live: `07bcbbf` fix(callback): symbol normalization and model output queue
- godzilla: `efd815a` fix(phase-6): fix on_factor callback with GIL and symbol normalization

---

## 1. 問題描述

Phase 6 測試過程中發現兩個極為嚴重的回調失敗問題：

### 1.1 on_factor 回調不執行
- **症狀**: Python `on_factor()` 從未被調用
- **數據流**: Binance → runner.cpp → hf-live → ??? (斷點)
- **影響**: 無法接收線性模型的預測結果

### 1.2 on_order 回調不傳播
- **症狀**: 懷疑訂單狀態更新未到達 Python
- **實際狀態**: 後來發現 on_order **一直正常工作**

---

## 2. 診斷過程

### 2.1 初步診斷 - 數據流追蹤

#### 步驟 1: 確認數據入口
```bash
# runner.cpp 日誌顯示數據正在轉發
[DEBUG] Depth #1 signal_on_data_=OK signal_engine_handle_=OK
[DEBUG] Calling signal_on_data_(Depth) #1
```
✅ **結論**: runner.cpp → signal_api.cpp 數據流正常

#### 步驟 2: 確認 FactorEngine 接收
```bash
# signal_api.cpp 日誌
[signal_api] OnDepth #1 symbol=btc_usdt
[signal_api] OnDepth #2 symbol=btc_usdt
```
✅ **結論**: signal_api.cpp → FactorEngine 接收正常

#### 步驟 3: 檢查 FactorEngine 處理
```bash
# 日誌顯示：
🚀 [ScanThread::SendData] Processing BTCUSDT (count=20)
📥 [ModelEngine::SendFactors] Received factors: assets=1 item_size=80
```
✅ **結論**: FactorEngine 正在處理數據並發送到 ModelEngine

#### 步驟 4: 檢查 SignalSender 調用
```bash
# 預期看到但沒有的日誌：
📨 [SignalSender::Send] Phase 4I: Queuing result
🎯 [SignalSender::ExecuteCallback] Phase 4I: Executing in main thread
```
❌ **問題點**: SignalSender::Send() 從未被調用

---

### 2.2 根因分析 - 多層問題

#### 問題 1: 符號格式不匹配 (最外層)

**發現過程**:
```cpp
// Binance 發送的符號格式
symbol = "btc_usdt"  // 小寫 + 底線

// FactorEngine OnDepth 處理
std::transform(code.begin(), code.end(), code.begin(), ::toupper);
// 結果: "BTC_USDT" (大寫 + 底線)

// FactorEngine 註冊的符號
code_info_.find("BTCUSDT")  // 大寫，無底線
```

**問題**:
```cpp
auto iter = code_info_.find(code);  // 查找 "BTC_USDT"
if (iter == code_info_.end()) {
    return;  // ❌ 找不到，直接返回！
}
```

**影響**: 所有 Depth/Trade/Ticker/IndexPrice 數據被靜默丟棄

---

#### 問題 2: LinearModel 輸出隊列未初始化 (中間層)

**發現過程**:
```cpp
// LinearModel::Calculate() 填充 output_
void Calculate(const models::comm::input_t& input) override {
    // ... 計算邏輯
    output_.values.push_back(pred_signal);
    output_.values.push_back(pred_confidence);
    // ❌ 沒有推送到隊列！
}

// ModelResultScanThread 嘗試讀取
if (models_[i]->TryGetOutput(model_output)) {
    // ❌ output_queues_ 是空的，TryGetOutput() 永遠返回 false
}
```

**問題**:
1. `output_queues_` 在構造函數中從未初始化
2. `Calculate()` 填充 `output_` 但從未推送到 `output_queues_[0]`

**對比其他模型** (test0000):
```cpp
// test0000 是 factor-only 模型，不需要 output_queues_
// 直接通過 FactorResultScanThread 發送
```

---

#### 問題 3: GIL 註釋不完整 (最內層，實際無影響)

**初步懷疑**:
```cpp
void on_factor(...) override {
    PYBIND11_OVERLOAD(void, strategy::Strategy, on_factor, ...);
    // ❌ 沒有 py::gil_scoped_acquire？
}
```

**實際狀態**:
- GIL acquire 代碼一直存在
- 只是註釋不清楚
- 這不是導致問題的原因

---

### 2.3 診斷技巧 - 逐層添加調試輸出

#### 技巧 1: 入口點驗證
```cpp
// runner.cpp - 確認數據轉發
static int depth_count = 0;
if (++depth_count <= 5) {
    std::cerr << "[DEBUG] Calling signal_on_data_(Depth) #" << depth_count << std::endl;
}
```

#### 技巧 2: 模型計算追蹤
```cpp
// ModelCalculationThread - 確認 Calculate() 被調用
if (++calc_count <= 5) {
    std::cerr << "🧮 [ModelCalcThread] Calling Calculate #" << calc_count << std::endl;
}
```

#### 技巧 3: 輸出隊列監控
```cpp
// ModelResultScanThread - 確認隊列有數據
if (models_[i]->TryGetOutput(model_output)) {
    std::cerr << "📤 [ModelResultScan] Got output values="
              << model_output.values.size() << std::endl;
}
```

---

## 3. 修復方案

### 3.1 符號正規化 (必須修復)

**文件**: `hf-live/app_live/engine/factor_calculation_engine.cpp`

**修改**: 在 OnDepth/OnTrade/OnTicker/OnIndexPrice 中添加底線移除

```cpp
void FactorCalculationEngine::OnDepth(std::shared_ptr<hf::Depth> depth) {
    std::string code(depth->symbol);

    // 轉換為大寫 (Binance 發送小寫,但系統使用大寫)
    std::transform(code.begin(), code.end(), code.begin(), ::toupper);

    // ✅ 新增: 移除底線 (btc_usdt → BTC_USDT → BTCUSDT)
    code.erase(std::remove(code.begin(), code.end(), '_'), code.end());

    auto iter = code_info_.find(code);
    // ...
}
```

**應用到**: OnDepth (L269), OnTrade (L307), OnTicker (L338), OnIndexPrice (L368)

---

### 3.2 LinearModel 輸出隊列初始化 (必須修復)

**文件**: `hf-live/models/linear/linear_model.cc`

**修改 1**: 構造函數中初始化隊列
```cpp
LinearModel(...) : ModelInterface(...) {
    // ... 權重初始化

    // ✅ 新增: 初始化輸出隊列 (多線程模型需要)
    output_queues_.push_back(
        std::make_unique<models::comm::SPSCQueue<models::comm::output_t>>(1024)
    );
}
```

**修改 2**: Calculate() 結尾推送到隊列
```cpp
void Calculate(const models::comm::input_t& input) override {
    // ... 計算邏輯
    output_.values.push_back(pred_signal);
    output_.values.push_back(pred_confidence);

    // ✅ 新增: 推送到輸出隊列 (供 ModelResultScanThread 使用)
    if (!output_queues_.empty()) {
        output_queues_[0]->push(output_);
    }
}
```

---

### 3.3 GIL 註釋改進 (文檔改進)

**文件**: `core/cpp/wingchun/pybind/pybind_wingchun.cpp`

**修改**: 添加清晰的註釋
```cpp
void on_factor(...) override {
    py::gil_scoped_acquire acquire;  // 必須：從 C++ 回調線程調用 Python 需要 GIL
    PYBIND11_OVERLOAD(void, strategy::Strategy, on_factor, ...);
}
```

---

## 4. 驗證結果

### 4.1 on_factor 回調成功
```
[PYTHON_STDERR] on_factor CALLED! symbol=BTCUSDT
🤖 [LinearModel] BTCUSDT @ 1766151156848030498
   📈 Signal: +1653.6512 (BULLISH)
   🎯 Confidence: 100.00%
```

### 4.2 on_order 回調正常 (一直正常)
```
📬 [on_order] order_id=312431619216572603 status=OrderStatus.Submitted
              ex_order_id='11013752642'

🎉🎉🎉 訂單已成功提交到 Binance Futures Testnet! 🎉🎉🎉
   🌐 Binance Order ID: 11013752642

🎉 [Test Complete] Order cancelled successfully!
```

---

## 5. 經驗教訓與未來注意事項

### 5.1 符號格式標準化 ⚠️

**問題**: 不同系統使用不同的符號格式

| 來源 | 格式 | 範例 |
|------|------|------|
| Binance API | 小寫 + 底線 | `btc_usdt` |
| FactorEngine | 大寫，無底線 | `BTCUSDT` |
| Python Config | 大寫，無底線 | `BTCUSDT` |

**未來建議**:
1. ✅ **統一入口點正規化**: 在 `signal_api.cpp` 或 `factor_calculation_engine.cpp` 的最外層立即轉換
2. ✅ **添加符號映射表**: 支持多種格式的查找
3. ✅ **添加警告日誌**: 當符號無法識別時，輸出警告而不是靜默丟棄

```cpp
// 建議的防禦性代碼
auto iter = code_info_.find(code);
if (iter == code_info_.end()) {
    static std::set<std::string> warned_symbols;
    if (warned_symbols.find(code) == warned_symbols.end()) {
        std::cerr << "⚠️  [FactorEngine] Unknown symbol: " << code
                  << " (original: " << original_symbol << ")" << std::endl;
        warned_symbols.insert(code);
    }
    return;
}
```

---

### 5.2 模型輸出隊列模式 ⚠️

**問題**: 兩種模型類型的輸出方式不同

| 模型類型 | 輸出方式 | 範例 |
|----------|----------|------|
| Factor-only | 直接通過 FactorResultScanThread | test0000, demo |
| Model-based | 通過 output_queues_ → ModelResultScanThread | linear |

**混淆點**: LinearModel 繼承 ModelInterface，有 `output_` 和 `output_queues_` 兩個成員

**未來建議**:
1. ✅ **明確文檔化**: 在 `model_base.h` 中清楚說明兩種模式
2. ✅ **提供模板代碼**: 為新模型提供正確的模板
3. ✅ **運行時檢查**: 如果模型有 output_queues_ 但為空，在 TryGetOutput 中輸出警告

```cpp
// 建議的檢查代碼
bool TryGetOutput(output_t& output) {
    if (output_queues_.empty()) {
        static bool warned = false;
        if (!warned) {
            std::cerr << "⚠️  Model " << model_name_
                      << " has empty output_queues_. Did you forget to initialize?"
                      << std::endl;
            warned = true;
        }
        return false;
    }
    // ...
}
```

---

### 5.3 PyBind11 GIL 管理 ⚠️

**原則**: 從 C++ 線程調用 Python 代碼時**必須**持有 GIL

**常見場景**:

| 場景 | 需要 GIL | 原因 |
|------|---------|------|
| 主線程 → Python | ✅ 已有 | Python 主線程自動持有 |
| 回調線程 → Python | ⚠️ **需要** | PYBIND11_OVERLOAD 調用 Python |
| C++ 工作線程 | ❌ 不需要 | 純 C++ 計算 |

**未來建議**:
1. ✅ **標準化註釋**: 所有 PYBIND11_OVERLOAD 前都加上 GIL 註釋
2. ✅ **Review Checklist**: 新增虛函數覆蓋時檢查 GIL
3. ✅ **單元測試**: 在多線程環境下測試所有 Python 回調

---

### 5.4 調試策略 ⚠️

**有效的調試技巧**:
1. ✅ **逐層驗證**: 從入口點開始，逐層添加日誌
2. ✅ **計數器技巧**: 使用 `static int count` 限制調試輸出數量
3. ✅ **符號追蹤**: 在關鍵點打印符號原始值和轉換後值
4. ✅ **隊列監控**: 檢查隊列是否為空、大小是否增長

**無效的調試方法**:
1. ❌ 直接猜測問題所在
2. ❌ 一次性添加多個修復
3. ❌ 不驗證假設就進行下一步

---

### 5.5 系統架構理解 ⚠️

**關鍵數據流** (Phase 6):
```
Binance WebSocket (btc_usdt)
    ↓
runner.cpp (轉發到 hf-live)
    ↓
signal_api.cpp (OnDepth/Trade/Ticker/IndexPrice)
    ↓
FactorCalculationEngine (符號轉換: btc_usdt → BTCUSDT)
    ↓
FactorCalculationThread (計算 15 個市場因子)
    ↓
ModelEngine::SendFactors (發送到模型)
    ↓
ModelCalculationThread::Calculate (LinearModel 計算)
    ↓
output_queues_[0]->push(output_)  ← 必須！
    ↓
ModelResultScanThread::TryGetOutput
    ↓
SendData (回調到 SignalSender)
    ↓
SPSCQueue<CallbackResult> (Phase 4I)
    ↓
signal_poll_callbacks (主線程輪詢)
    ↓
SignalSender::ExecuteCallback
    ↓
Python on_factor() ✅
```

**關鍵節點**:
- ⚠️ **符號轉換**: signal_api.cpp 或 factor_calculation_engine.cpp
- ⚠️ **隊列推送**: LinearModel::Calculate() 結尾
- ⚠️ **GIL 獲取**: pybind_wingchun.cpp PYBIND11_OVERLOAD

---

### 5.6 代碼審查清單 ✅

新增模型時必須檢查：
- [ ] 構造函數初始化 `output_queues_`
- [ ] `Calculate()` 結尾推送到 `output_queues_[0]`
- [ ] 符號格式處理（如果直接接收市場數據）
- [ ] Python 回調有 `py::gil_scoped_acquire`

新增交易所時必須檢查：
- [ ] 符號格式標準化到 `BTCUSDT` 格式
- [ ] 添加符號映射表
- [ ] 添加未知符號警告

---

## 6. 參考資料

### 6.1 相關文件
- [factor_calculation_engine.cpp](../../hf-live/app_live/engine/factor_calculation_engine.cpp)
- [linear_model.cc](../../hf-live/models/linear/linear_model.cc)
- [pybind_wingchun.cpp](../../core/cpp/wingchun/pybind/pybind_wingchun.cpp)
- [model_base.h](../../hf-live/models/_comm/model_base.h)

### 6.2 相關 Memory
- [callback-fix-analysis.md](../../.serena/memories/callback-fix-analysis.md)

### 6.3 Git Commits
```bash
# hf-live 子模組
git log --oneline | head -1
# 07bcbbf fix(callback): symbol normalization and model output queue

# 主倉庫
git log --oneline | head -1
# efd815a fix(phase-6): fix on_factor callback with GIL and symbol normalization
```

---

## 7. 總結

這次調試暴露了三層問題：

1. **外層 (符號格式)**: 系統間格式不統一導致數據丟棄
2. **中層 (隊列管理)**: 模型輸出隊列未正確初始化和使用
3. **內層 (GIL 管理)**: 註釋不清晰（但代碼正確）

最重要的教訓：
- ⚠️ **永遠不要靜默丟棄數據** - 添加警告日誌
- ⚠️ **明確文檔化** - 兩種模型輸出模式需要清晰說明
- ⚠️ **逐層驗證** - 不要跳過任何檢查點

修復後，Phase 6 完整管線正常工作：
- ✅ Binance 四種市場數據正確接收
- ✅ 15 個市場因子正確計算
- ✅ LinearModel 正確預測
- ✅ on_factor 回調正確執行
- ✅ on_order 回調正常工作
