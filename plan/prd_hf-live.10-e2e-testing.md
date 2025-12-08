# PRD 10: HF-Live 端到端測試實施報告

**狀態**: 部分完成 (Phase 1-3 ✅, Phase 4-6 ⏸️)  
**日期**: 2025-12-08  
**目標**: 驗證完整數據流 Binance → Factor → Model → Python `on_factor` callback

---

## 執行摘要

**已完成**: ✅ Phase 1-3 (測試組件開發與編譯驗證)  
**待完成**: ⏸️ Phase 4-6 (運行時數據流驗證)  
**阻礙因素**: PM2 配置格式需進一步調試

---

## Phase 1: test0000 因子實現 ✅

### 1.1 實現內容

**文件**:
- `hf-live/factors/test0000/meta_config.h` - 因子元數據定義
- `hf-live/factors/test0000/factor_entry.h` - 類聲明
- `hf-live/factors/test0000/factor_entry.cpp` - 業務邏輯

**因子設計**:
```cpp
static const std::vector<std::string> kFactorNames = {
    "spread",        // Factor 0: ask - bid
    "mid_price",     // Factor 1: (ask + bid) / 2
    "bid_volume",    // Factor 2: bid_volume[0]
};
```

**日誌標記**:
- 🏁 FactorEntry 創建
- 📊 每10個 Depth 輸出一次 bid/ask
- 🔢 UpdateFactors 時輸出計算結果

### 1.2 編譯驗證

```bash
$ cd /app/hf-live/build && make
[100%] Built target signal

$ ls -lh libsignal.so
-rwxr-xr-x 1 root root 291K Dec  8 17:02 libsignal.so  # 增加 26KB

$ nm -D libsignal.so | grep test0000 | head -5
00000000000319e0 T _ZN7factors8test000011FactorEntry12DoOnAddQuoteERKN2hf5DepthE
0000000000031890 T _ZN7factors8test000011FactorEntry12DoOnAddTransERKN2hf5TradeE
00000000000318a0 T _ZN7factors8test000011FactorEntry17DoOnUpdateFactorsEl
...
```

**結論**: ✅ 因子編譯成功，符號正確導出

### 1.3 技術問題與解決

**問題 1**: `kFactorSetName` 重複定義  
**解決**: 移除 `factor_entry.h` 中的聲明，保留 `meta_config.h` 中的定義

**問題 2**: `REGISTER_FACTOR_AUTO` 宏無法識別  
**解決**: 添加 `#include "factors/_comm/factor_entry_registry.h"`

**問題 3**: `make_unique` 歧義  
**解決**: 修改 `factor_entry_registry.h`，顯式使用:
```cpp
return factors::make_unique<T>(asset, metadata, config);
// 添加返回類型標註: -> FactorEntryPtr
```

### 1.4 Git Commit

```
commit c6acbdb
feat(hf-live): add test0000 factor for e2e testing

- Implements simple 3-factor calculation: spread, mid_price, bid_volume
- Adds detailed logging for data flow verification (emoji markers)
- Registers factor with REGISTER_FACTOR_AUTO macro
- Updates DefaultConfig to use test0000 factor and model
- Fixes factor_entry_registry.h: explicit factors::make_unique
```

---

## Phase 2: test0000 模型實現 ✅

### 2.1 實現內容

**文件**:
- `hf-live/models/test0000/test0000_model.cc`

**模型設計**:
```cpp
class Test0000Model : public models::comm::ModelInterface {
    void Calculate(const models::comm::input_t& input) override {
        // Trivial inference (固定輸出用於測試)
        float pred_signal = 1.0f;
        float pred_confidence = 0.8f;
        output_.values.push_back(pred_signal);
        output_.values.push_back(pred_confidence);
    }
};
```

**日誌標記**:
- 🤖 Model 創建
- 🔮 Calculate 執行，輸出預測值

### 2.2 編譯驗證

```bash
$ make
[100%] Built target signal

$ ls -lh libsignal.so
-rwxr-xr-x 1 root root 301K Dec  8 17:05 libsignal.so  # 增加 10KB

$ nm -D libsignal.so | grep test0000 | grep -i model
0000000000030cf0 T _ZN6models8test000011GetMetadataEv
0000000000032640 W _ZN6models8test000013Test0000Model9CalculateERKNS_4comm7input_tE
...
```

**結論**: ✅ 模型編譯成功，符號正確導出

### 2.3 設計簡化

**原計劃**: 解析 `input.factor_datas` 並執行 `pred_signal = spread * 100`  
**實際**: 輸出固定值 (1.0, 0.8)  
**理由**: `input_t` 使用序列化數據格式 (`std::vector<char> factor_datas`)，解析邏輯複雜，簡化以專注數據流驗證

### 2.4 Git Commit

```
commit b289bbb
feat: add test0000 model for e2e testing

- Implements trivial inference: pred_signal=1.0, pred_confidence=0.8
- Adds 🔮 emoji logging for model calculation tracking
- Registers model with REGISTER_MODEL_AUTO macro
```

---

## Phase 3: test_hf_live 策略實現 ✅

### 3.1 實現內容

**文件**:
- `strategies/test_hf_live/test_hf_live.py`
- `strategies/test_hf_live/config.json`

**策略設計**:
```python
def on_depth(ctx, depth):
    """驗證 Binance 數據接收"""
    ctx.logger.info(f"✅ [on_depth] {depth.symbol} bid={depth.bid_price[0]}")

def on_factor(ctx, symbol, timestamp, values):
    """驗證完整數據流"""
    ctx.logger.info(f"🎉 [on_factor] {symbol}")
    ctx.logger.info(f"   Model Output: {values}")
    if len(values) >= 2:
        pred_signal, pred_confidence = values[0], values[1]
        ctx.logger.info("   🎊 E2E TEST PASSED!")
```

**日誌標記**:
- 🏁 策略啟動
- ✅ on_depth 回調
- 🎉 on_factor 回調
- 🎊 測試通過

### 3.2 配置文件

```json
{
  "name": "test_hf_live",
  "script": "/app/core/python/dev_run.py",
  "args": ["strategy", "--name", "test_hf_live", "--path", "strategies/test_hf_live/test_hf_live.py"],
  "signal_library_path": "/app/hf-live/build/libsignal.so",
  "subscriptions": [
    {
      "source": "binance",
      "exchange": "binance",
      "symbol": "btcusdt",
      "is_level2": true
    }
  ]
}
```

### 3.3 Git Commit

```
commit dc26979
feat: add test_hf_live strategy for e2e testing

- Minimal strategy with on_depth and on_factor callbacks
- Adds emoji logging (🏁 ✅ 🎉 🎊) for easy tracking
- Validates complete data flow: Binance → Factor → Model → Python
```

---

## Phase 4-6: 運行時驗證（漸進式）⏸️

### 驗證原則

1. **逐層測試**: 基礎服務 → 策略 → Signal Library → 因子 → 模型 → 回調
2. **失敗即停**: 任何階段失敗立即停止，不前進
3. **實際日誌**: 只依賴真實輸出，不假設成功
4. **手動確認**: 用戶驗證每個階段的實際日誌

---

### Phase 4A: 基礎服務啟動 ⏸️

**目標**: 確認 Master/Ledger/MD/TD 能正常啟動

**操作**:
```bash
docker exec -it godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"
docker exec godzilla-dev pm2 list
```

**成功標誌**:
```
┌────┬──────────────┬─────────┬────────┬──────┬───────────┐
│ 0  │ master       │ online  │ ...    │ ...  │ ...       │
│ 1  │ ledger       │ online  │ ...    │ ...  │ ...       │
│ 2  │ md_binance   │ online  │ ...    │ ...  │ ...       │
│ 3  │ td_binance   │ online  │ ...    │ ...  │ ...       │
└────┴──────────────┴─────────┴────────┴──────┴───────────┘
```

**失敗處理**: 
- 檢查 `pm2 logs <service>` 找錯誤原因
- 確認 Binance API key 配置正確
- 檢查網絡連接

---

### Phase 4B: 簡單策略測試（無 signal library）⏸️

**目標**: 確認策略能啟動並收到 on_depth 回調

**操作**:
```bash
docker exec godzilla-dev pm2 start /app/scripts/test_hf_live/strategy.json
docker exec -it godzilla-dev pm2 logs strategy_test_hf_live --lines 50
```

**成功標誌**:
```
strategy_test_hf_live  | 🏁 [test_hf_live] Pre-Start
strategy_test_hf_live  | ✅ [on_depth] btcusdt bid=42000.50 ask=42001.20
strategy_test_hf_live  | ✅ [on_depth] btcusdt bid=42001.00 ask=42001.50
```

**失敗處理**: 
- 檢查 symbol 訂閱格式（小寫 + 底線）
- 檢查 MD gateway 是否正常運行
- 查看 `pm2 logs md_binance` 確認數據接收

**簡化策略代碼**（strategies/test_hf_live/test_hf_live.py）:
```python
from kungfu.wingchun.constants import *
from pywingchun.constants import InstrumentType

def pre_start(context):
    context.log().info("🏁 [test_hf_live] Pre-Start")
    context.subscribe("binance", ["btcusdt"], InstrumentType.Spot, Exchange.BINANCE)

def on_depth(context, depth):
    bid = depth.bid_price[0]
    ask = depth.ask_price[0]
    context.log().info(f"✅ [on_depth] {depth.symbol} bid={bid:.2f} ask={ask:.2f}")

def post_stop(context):
    context.log().info("🏁 [test_hf_live] Stopped")
```

---

### Phase 4C: 研究 libsignal.so 集成方式 ⏸️

**目標**: 找到正確的 signal library 加載方法

**調查清單**:
1. ✅ 查看 `strategies/factor_strategy/run.py` 實現（已確認沒有特殊加載）
2. ⏸️ 查看 hf-live 文檔是否有集成說明
3. ⏸️ 查看 hf-live 源碼中的 Python 綁定部分
4. ⏸️ 測試環境變量方案（LD_LIBRARY_PATH）
5. ⏸️ 測試 ctypes 動態加載方案

**可能的集成方案**:

**方案 A - 環境變量**（最簡單，優先嘗試）:
```json
"env": {
  "KF_HOME": "/app/runtime",
  "LD_LIBRARY_PATH": "/app/hf-live/build:$LD_LIBRARY_PATH",
  "LD_PRELOAD": "/app/hf-live/build/libsignal.so"
}
```

**方案 B - 策略內加載**（需要代碼支持）:
```python
import ctypes
signal_lib = ctypes.CDLL('/app/hf-live/build/libsignal.so')
# 調用初始化函數...
```

**方案 C - 修改 Wingchun**（最複雜，最後考慮）:
- 在 Strategy 類中添加 signal library 支持
- 需要修改 C++ 和 Python 綁定

**決策**: 先嘗試方案 A，失敗再研究方案 B/C

---

### Phase 4D: 驗證因子層（C++ 日誌）⏸️

**前提條件**: Phase 4C 成功集成 libsignal.so

**目標**: 確認 test0000 因子被創建並計算

**預期日誌**（來自 C++ stdout）:
```
🏁 [test0000::FactorEntry] Created for: BTCUSDT
📊 [test0000 #10] bid=42000.5 ask=42001.2
📊 [test0000 #20] bid=42001.0 ask=42001.5
🔢 [test0000::UpdateFactors] spread=0.7 mid=42000.85
```

**驗證方法**:
```bash
docker exec -it godzilla-dev pm2 logs strategy_test_hf_live | grep "🏁\|📊\|🔢"
```

**失敗可能原因**:
- libsignal.so 未正確加載（檢查 ldd）
- test0000 因子未註冊（檢查 REGISTER_FACTOR_AUTO）
- DefaultConfig 未生效（檢查 config_parser.h）

---

### Phase 4E: 驗證模型層（C++ 日誌）⏸️

**前提條件**: Phase 4D 成功

**目標**: 確認 test0000 模型被創建並執行推理

**預期日誌**（來自 C++ stdout）:
```
🤖 [test0000::Model] Created with 3 factors
🔮 [test0000::Calculate] asset=BTCUSDT → output=[1.0, 0.8]
```

**驗證方法**:
```bash
docker exec -it godzilla-dev pm2 logs strategy_test_hf_live | grep "🤖\|🔮"
```

**失敗可能原因**:
- test0000 模型未註冊
- 因子→模型數據流未連接
- 需要檢查 ModelCalculationEngine 配置

---

### Phase 4F: 驗證 Python 回調（on_factor）⏸️

**前提條件**: Phase 4E 成功

**目標**: 確認 Python 能收到 on_factor 回調

**策略添加回調**:
```python
def on_factor(ctx, symbol, timestamp, values):
    ctx.log().info(f"🎉 [on_factor] {symbol} @ {timestamp}")
    ctx.log().info(f"   Model Output: {values}")
    if len(values) >= 2:
        ctx.log().info(f"   ✅ pred_signal={values[0]:.4f}, pred_confidence={values[1]:.4f}")
        ctx.log().info("   🎊 E2E TEST PASSED!")
```

**預期日誌**:
```
strategy_test_hf_live  | 🎉 [on_factor] BTCUSDT @ 1733684523000000000
strategy_test_hf_live  |    Model Output: [1.0, 0.8]
strategy_test_hf_live  |    ✅ pred_signal=1.0000, pred_confidence=0.8000
strategy_test_hf_live  |    🎊 E2E TEST PASSED!
```

**驗證方法**:
```bash
docker exec -it godzilla-dev pm2 logs strategy_test_hf_live | grep "🎉\|🎊"
```

**失敗可能原因**:
- on_factor 回調未定義或未註冊
- C++ → Python 綁定問題
- 需要檢查 pybind11 綁定代碼

---

### 當前進度總結

| 階段 | 狀態 | 說明 |
|-----|------|------|
| Phase 1-3 | ✅ 完成 | test0000 因子、模型、策略代碼已編寫 |
| Phase 4A | ⏸️ 待測試 | 基礎服務啟動驗證 |
| Phase 4B | ⏸️ 待測試 | 簡單策略測試（無 signal library） |
| Phase 4C | ⏸️ 待研究 | libsignal.so 集成方式調查 |
| Phase 4D | ⏸️ 待驗證 | 因子層日誌驗證 |
| Phase 4E | ⏸️ 待驗證 | 模型層日誌驗證 |
| Phase 4F | ⏸️ 待驗證 | Python on_factor 回調驗證 |

**下一步**: 執行 Phase 4A 測試，等待用戶確認結果

---

## Phase 7: 數據流圖與架構總結

### 7.1 完整數據流

```
┌─────────────────┐
│ Binance         │
│ WebSocket       │
└────────┬────────┘
         │ Depth (bid/ask/volume)
         ▼
┌─────────────────────────────────────┐
│ FactorCalculationEngine             │
│  ├─ test0000::FactorEntry           │
│  │   ├─ DoOnAddQuote()      📊     │
│  │   └─ DoOnUpdateFactors() 🔢     │
│  └─ Output: [spread, mid_price, ... │
└──────────┬──────────────────────────┘
           │ Factor Values (3 floats)
           ▼
┌─────────────────────────────────────┐
│ ModelCalculationEngine              │
│  ├─ test0000::Test0000Model    🤖  │
│  │   └─ Calculate()           🔮   │
│  └─ Output: [pred_signal, pred_co...│
└──────────┬──────────────────────────┘
           │ Model Predictions (2 floats)
           ▼
┌─────────────────────────────────────┐
│ Python Strategy (via pybind11)      │
│  └─ on_factor(symbol, values)  🎉  │
│      └─ Validation Logic       🎊  │
└─────────────────────────────────────┘
```

### 7.2 關鍵組件

| 組件 | 語言 | 輸入 | 輸出 | 狀態 |
|------|------|------|------|------|
| test0000 Factor | C++ | Depth (bid, ask, volume) | 3 floats (spread, mid, bid_vol) | ✅ 編譯通過 |
| test0000 Model | C++ | 3 factor values (序列化) | 2 floats (signal, confidence) | ✅ 編譯通過 |
| test_hf_live Strategy | Python | symbol, timestamp, values | 日誌輸出 | ✅ 文件就緒 |
| PM2 配置 | JSON | - | 進程管理 | ⏸️ 調試中 |

### 7.3 emoji 日誌系統

| Emoji | 含義 | 出現位置 |
|-------|------|---------|
| 🏁 | 初始化開始 | FactorEntry 構造函數, pre_start() |
| 📊 | Depth 數據 | DoOnAddQuote (每10個) |
| 🔢 | 因子計算 | DoOnUpdateFactors |
| 🤖 | 模型創建 | Test0000Model 構造函數 |
| 🔮 | 模型推理 | Calculate() |
| ✅ | Depth 回調 | on_depth() |
| 🎉 | Factor 回調 | on_factor() |
| 🎊 | 測試通過 | on_factor() 驗證邏輯 |

---

## 測試結論與建議

### 已驗證 ✅

1. **代碼完整性**: 因子、模型、策略三個組件完整實現
2. **編譯正確性**: libsignal.so 成功編譯，符號正確導出
3. **設計合理性**: 數據流設計清晰，日誌標記完善
4. **Git 管理**: 3 個 commit 分別記錄每個 Phase 的成果

### 待驗證 ⏸️

1. **運行時連接**: C++ libsignal.so 與 Python 的 pybind11 綁定
2. **數據傳遞**: 因子值從 C++ 傳遞到 Python 的正確性
3. **回調觸發**: on_factor() 是否能被 C++ 正確調用
4. **性能**: 延遲是否滿足低延遲交易需求

### 建議後續步驟

**短期 (1-2 小時)**:
1. 修正 PM2 配置或使用手動啟動
2. 運行測試並收集完整日誌
3. 驗證所有 8 個 Checkpoint

**中期 (1-2 天)**:
1. 實現 `input.factor_datas` 解析邏輯，讓模型使用實際因子值
2. 添加性能測試 (延遲測量)
3. 添加異常處理 (Binance 斷線、數據異常)

**長期 (1-2 週)**:
1. 開發更多實際因子 (技術指標、訂單簿分析)
2. 集成真實 ML 模型 (PyTorch/ONNX)
3. 生產環境部署與監控

---

## 附錄: 文件清單

### 新增文件 (8 個)

**C++ 因子**:
- `hf-live/factors/test0000/meta_config.h` (29 行)
- `hf-live/factors/test0000/factor_entry.h` (26 行)
- `hf-live/factors/test0000/factor_entry.cpp` (37 行)

**C++ 模型**:
- `hf-live/models/test0000/test0000_model.cc` (55 行)

**Python 策略**:
- `strategies/test_hf_live/test_hf_live.py` (39 行)
- `strategies/test_hf_live/config.json` (22 行)

**文檔**:
- `plan/prd_hf-live.10-e2e-testing.md` (本文件)

### 修改文件 (3 個)

- `hf-live/CMakeLists.txt` (+2 行: test0000 因子和模型)
- `hf-live/app_live/common/config_parser.h` (DefaultConfig 更新)
- `hf-live/factors/_comm/factor_entry_registry.h` (make_unique 歧義修復)

### Git Commits (4 個)

1. `c6acbdb` - feat(hf-live): add test0000 factor
2. `88cf6c7` - (submodule) feat: add test0000 factor
3. `b289bbb` - feat: add test0000 model
4. `dc26979` - feat: add test_hf_live strategy

---

## 參考資料

- [PRD 09: HF-Live 實施差距分析](prd_hf-live.09-implementation-gaps.md)
- [Implementation Status Report](IMPLEMENTATION_STATUS_REPORT.md)
- Binance WebSocket API: https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams

---

**報告生成時間**: 2025-12-08 17:20 UTC  
**總開發時間**: ~2 小時  
**總代碼行數**: ~208 行 (C++) + 61 行 (Python) + 22 行 (JSON) = 291 行
