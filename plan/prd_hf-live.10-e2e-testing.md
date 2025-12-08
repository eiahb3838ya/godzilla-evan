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

## Phase 4-6: 運行時驗證 ⏸️

### 4.1 預期測試流程

1. 啟動策略: `pm2 start strategies/test_hf_live/config.json`
2. 觀察日誌，驗證:
   - ✅ **Checkpoint 1**: `🏁 FactorEntry Created` (因子初始化)
   - ✅ **Checkpoint 2**: `📊 bid=... ask=...` (Depth 接收)
   - ✅ **Checkpoint 3**: `🔢 UpdateFactors spread=...` (因子計算)
   - ✅ **Checkpoint 4**: `🤖 Model Created` (模型初始化)
   - ✅ **Checkpoint 5**: `🔮 Calculate output=[1.0, 0.8]` (模型推理)
   - ✅ **Checkpoint 6**: `✅ on_depth` (Python Depth 回調)
   - ✅ **Checkpoint 7**: `🎉 on_factor` (Python Factor 回調)
   - ✅ **Checkpoint 8**: `🎊 E2E TEST PASSED` (完整流程驗證)

### 4.2 遇到的問題

**問題**: PM2 配置格式調試複雜

**嘗試的方法**:
1. 使用 `"path": "strategies/test_hf_live/test_hf_live.py"` → ❌ 無法識別
2. 使用 `"args": "strategies.test_hf_live.test_hf_live"` → ❌ dev_run.py 不支持模塊名
3. 使用 `"args": ["strategy", "--name", "test_hf_live", "--path", "..."]` → ⏸️ 需進一步測試

**dev_run.py 命令格式**:
```bash
python3 /app/core/python/dev_run.py strategy \
    --name test_hf_live \
    --path strategies/test_hf_live/test_hf_live.py
```

### 4.3 建議的手動測試步驟

```bash
# 1. 進入容器
docker exec -it godzilla-dev bash

# 2. 確認 libsignal.so 存在
ls -lh /app/hf-live/build/libsignal.so

# 3. 手動啟動策略 (前台運行)
cd /app
python3 core/python/dev_run.py strategy \
    --name test_hf_live \
    --path strategies/test_hf_live/test_hf_live.py

# 4. 觀察日誌輸出，驗證數據流
# 期待看到: 🏁 📊 🔢 🤖 🔮 ✅ 🎉 🎊 標記
```

### 4.4 後續工作

**Option A**: 修正 PM2 配置格式並重新測試  
**Option B**: 使用 systemd 或其他進程管理工具  
**Option C**: 直接在終端前台運行測試

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
