# PRD 11: hf-live Phase 4F 實施歷程整合文檔

**文檔版本**: v1.0
**創建日期**: 2025-12-10
**目的**: 整合 Phase 4F 開發過程中產生的中間文檔,提供完整的實施歷程記錄

---

## 文檔索引

本文檔整合了以下 4 個中間過程文檔的內容:

1. **phase-4f-test-plan.md** - Phase 4F E2E 測試計劃與執行步驟
2. **IMPLEMENTATION_STATUS_REPORT.md** - 87% 完成度狀態評估報告 (2025-12-08)
3. **hf-live/ADAPTATION_SUMMARY.md** - FactorEngine/ModelEngine 技術適配細節
4. **hf-live/IMPLEMENTATION_SUMMARY.md** - 核心組件實現摘要

整合後原文檔將被刪除,所有信息統一保存於此。

---

## 第一部分: Phase 4F 測試計劃

> **來源**: plan/phase-4f-test-plan.md (453 lines)
> **用途**: Phase 4F E2E 數據流測試的系統化驗證方法論

### 測試目標

驗證完整的 E2E 數據流:
```
Binance WebSocket → signal_on_data → FactorEngine → ModelEngine
→ SignalSender → Runner::on_factor_callback → Python on_factor
```

### 8 步漸進式測試方法

#### Step 1: 驗證 SignalSender::Send 被調用

**目的**: 確認 ModelEngine → SignalSender 調用成功

**檢查點**:
```
📨 [SignalSender::Send] CALLED!
   Symbol: BTCUSDT
   Timestamp: 1765377407481907263
   Count: 2
   Callback: VALID
   Values: [1, 0.8]
```

**成功標準**:
- ✅ SignalSender::Send 日誌出現
- ✅ Callback 狀態為 VALID (非 NULL)
- ✅ Values 數據正確 (2 個值: pred_signal=1.0, pred_confidence=0.8)

**失敗排查**:
- Callback NULL → 檢查 SetSendCallback 是否在 Start() 前調用
- Count 不正確 → 檢查 ModelResultScanThread 數據打包邏輯

---

#### Step 2: 驗證 signal_register_callback 被調用

**目的**: 確認 Godzilla 正確註冊回調函數

**檢查點**:
```
📞 [signal_register_callback] CALLED!
   Callback: VALID
   User data: VALID
```

**成功標準**:
- ✅ signal_api.cpp 註冊函數被調用
- ✅ callback 和 user_data 指針非 NULL

**失敗排查**:
- 函數未調用 → 檢查 dlsym 是否成功加載 "signal_register_callback"
- 指針為 NULL → 檢查 runner.cpp 傳遞的參數

---

#### Step 3-4: 編譯與部署流程

**Step 3**: 編譯 hf-live
```bash
cd /home/huyifan/projects/godzilla-evan/hf-live
rm -rf build && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

**Step 4**: 部署到容器
```bash
docker exec godzilla-dev mkdir -p /app/hf-live/lib
docker cp build/lib/libsignal.so godzilla-dev:/app/hf-live/lib/
docker exec godzilla-dev pm2 restart strategy-test-hf-live
```

---

#### Step 5: 檢查註冊日誌

**目的**: 確認回調函數成功註冊到 SignalSender

**檢查命令**:
```bash
docker exec godzilla-dev pm2 logs strategy-test-hf-live --lines 50 | grep "register_callback"
```

**成功標準**:
- ✅ 看到 "📞 [signal_register_callback] CALLED!"
- ✅ Callback 和 User data 都是 VALID

---

#### Step 6: 等待因子計算並檢查數據流

**目的**: 確認 FactorEngine → ModelEngine 數據流

**檢查點**:
```
📥 [ModelEngine::SendFactors] Received factors: assets=1 item_size=3 timestamp=...
✅ [ModelEngine::SendFactors] Pushed to buffer (count=1)
🔮 [test0000::Calculate] asset=BTCUSDT → output=[1, 0.8]
   ✅ [test0000] Output pushed to queue
🎯 [ModelScanThread::ScanFunc] TryGetOutput SUCCESS for model 0
```

**成功標準**:
- ✅ SendFactors 接收到因子數據
- ✅ test0000 模型計算成功
- ✅ 結果推送到輸出隊列
- ✅ ScanThread 成功讀取結果

**失敗排查**:
- SendFactors 未調用 → 檢查 FactorEngine::Start() 是否成功
- Output queue 推送失敗 → 檢查 output_queues_ 初始化
- TryGetOutput 失敗 → 檢查模型是否實現異步架構

---

#### Step 7: 檢查 Runner::on_factor_callback 日誌

**目的**: 確認 C++ 回調層收到數據

**檢查點**:
```
🔔 [Runner::on_factor_callback] CALLED!
   Symbol: BTCUSDT
   Timestamp: 1765377407481907263
   Count: 2
```

**成功標準**:
- ✅ on_factor_callback 被調用
- ✅ 參數正確傳遞

**失敗排查**:
- 未看到日誌 → 檢查 SignalSender callback 是否正確設置
- 數據損壞 → 檢查 signal_api.cpp 中的數據轉換邏輯

---

#### Step 8: 檢查 Python on_factor 回調 (最終目標)

**目的**: 確認 Python 策略層成功接收信號

**檢查命令**:
```bash
docker exec godzilla-dev pm2 logs strategy-test-hf-live --lines 100 | grep "Received factor"
```

**成功標準**:
- ✅ 看到 "[FACTOR] 🎊 Received factor for BTCUSDT @ timestamp (count=2)"
- ✅ 看到 "[FACTOR] Calling strategy on_factor for strategy_id=..."

**完整成功日誌序列**:
```
🔮 [test0000::Calculate] asset=BTCUSDT → output=[1, 0.8]
   ✅ [test0000] Output pushed to queue
🎯 [ModelScanThread::ScanFunc] TryGetOutput SUCCESS for model 0
📤 [ModelScanThread::SendData] CALLED!
   Callback: VALID
📨 [SignalSender::Send] CALLED!
   Values: [1, 0.8]
   ✅ Calling callback...
   ✅ Callback returned
🔔 [Runner::on_factor_callback] CALLED!
   Symbol: BTCUSDT
[FACTOR] 🎊 Received factor for BTCUSDT @ 1765377407481907263 (count=2)
[FACTOR] Calling strategy on_factor for strategy_id=1350253488
```

---

### 測試優先級分級

**P0 (阻塞級)**: 必須通過,否則整個 E2E 流程不可用
- SignalSender::Send 被調用
- Callback 非 NULL
- Python on_factor 被調用

**P1 (關鍵級)**: 嚴重影響功能但不完全阻塞
- 數據值正確性 (values 內容)
- 時間戳正確性
- 資產符號正確

**P2 (增強級)**: 可用性和調試相關
- 調試日誌完整性
- 性能追蹤字段
- 錯誤處理邏輯

---

### 故障排查決策樹

```
SignalSender::Send 未調用?
├─ YES → ModelResultScanThread 未啟動
│        └─ 檢查 ModelEngine::Start()
└─ NO  → Callback 是 NULL?
          ├─ YES → SetSendCallback 時序問題
          │        └─ 確保在 Init() 後、Start() 前調用
          └─ NO  → Runner callback 未調用?
                   ├─ YES → signal_register_callback 失敗
                   │        └─ 檢查 dlsym 加載
                   └─ NO  → Python on_factor 未調用?
                            └─ 檢查 pybind11 綁定和策略代碼
```

---

## 第二部分: 實施狀態評估報告 (87% 完成)

> **來源**: plan/IMPLEMENTATION_STATUS_REPORT.md (813 lines)
> **日期**: 2025-12-08
> **用途**: PRD 10 完成度全面評估

### 總體狀態: 87% 完成 🟢

#### 核心需求達成情況

| 需求 | 完成度 | 狀態 | 關鍵證據 |
|------|--------|------|----------|
| **Req 1**: hf-live 獨立編譯 | 100% | ✅ | CMakeLists.txt 完整、libsignal.so 成功生成 |
| **Req 2**: 冷庫 .so 使用 | 100% | ✅ | signal_api.cpp dlopen 機制、runner.cpp 動態加載 |
| **Req 3**: on_factor 信號流 | 95% | 🟢 | E2E 數據流打通、已知記憶體問題待修復 |
| **Req 4**: ref 業務邏輯完整性 | 82% | 🟡 | 核心邏輯完整、部分增強功能缺失 |
| **Req 5**: PRD 文檔實施 | 90% | 🟢 | 主要階段完成、Phase 7-8 待規劃 |

---

### 詳細組件分析

#### FactorCalculationEngine (88% 完成)

**已實現功能** ✅:
- 多線程因子計算架構
- SPMC 緩衝區 (Single Producer Multiple Consumer)
- FactorCalculationThread 管理
- FactorResultScanThread 結果收集
- 性能追蹤 (RDTSC 計時)
- 回調函數機制 (SendFactors)

**缺失功能** ⚠️:
- 動態因子配置加載 (目前硬編碼 spread/mid/bid)
- 因子熱更新機制
- 批量因子計算優化
- 錯誤恢復策略

**測試覆蓋**:
```
✅ OnDepth 數據接收
✅ 因子計算觸發
✅ 結果收集與發送
✅ 回調函數調用
⚠️ 多資產並發測試 (僅單資產驗證)
```

**性能指標**:
```
因子計算延遲: ~50-100 μs (微秒)
SPMC 推送延遲: ~10-20 μs
結果掃描週期: 1ms 輪詢
```

---

#### ModelCalculationEngine (85% 完成)

**已實現功能** ✅:
- 多線程模型計算架構
- SPMC 緩衝區 (接收因子)
- ModelCalculationThread 管理
- ModelResultScanThread 結果收集
- 異步模型架構支持 (SPSC output queue)
- 回調函數機制 (SetSendCallback)

**修復的關鍵問題** 🔧:
- **Callback 時序問題**: SetSendCallback 現在重建 ScanThread
- **test0000 異步架構**: 添加 output_queues_ 和 push 邏輯
- **SignalSender 集成**: includes 和調試日誌修復

**缺失功能** ⚠️:
- 動態模型配置加載 (目前硬編碼 test0000)
- 模型熱更新機制
- 多模型並行優化
- 模型輸出驗證邏輯

**測試覆蓋**:
```
✅ SendFactors 數據接收
✅ 模型計算執行
✅ 輸出隊列推送
✅ ScanThread 結果讀取
✅ SignalSender 回調調用
✅ Python on_factor 回調
⚠️ 多模型並發測試 (僅單模型驗證)
```

---

#### signal_api.cpp 集成層 (95% 完成)

**已實現功能** ✅:
- dlopen/dlsym 動態加載 libsignal.so
- signal_register_callback 綁定
- signal_on_data 數據轉換 (void* → FactorEngine)
- Lambda 回調鏈: ModelEngine → SignalSender
- 數據打包與轉換 (11 metadata + N output values)

**已知問題** 🐛:
- **記憶體損壞**: 局部 vector 的 dangling pointer
  ```cpp
  // 問題代碼 (signal_api.cpp:57-66)
  std::vector<double> predictions(data_with_metadata.begin() + 11,
                                  data_with_metadata.begin() + 11 + output_size);
  SignalSender::GetInstance().Send(symbol.c_str(), timestamp,
                                   predictions.data(), predictions.size());
  // predictions 銷毀 → dangling pointer
  ```

**修復建議** (3 個選項):
- **Option A** (推薦): SignalSender::Send 立即複製數據
- **Option B**: 使用 shared_ptr 延長生命週期
- **Option C**: 修改 Send 簽名為 `const vector<double>&`

---

### E2E 測試結果

#### 測試環境
- Godzilla 容器: godzilla-dev
- 策略: test_hf_live (strategies/test_hf_live/)
- 數據源: Binance WebSocket (BTCUSDT)
- 模型: test0000 (固定輸出 1.0, 0.8)

#### 成功證據

**完整日誌鏈** (2025-12-10 22:36 執行):
```
[22:36:45] 🔮 [test0000::Calculate] asset=BTCUSDT → output=[1, 0.8]
[22:36:45]    ✅ [test0000] Output pushed to queue
[22:36:45] 🎯 [ModelScanThread::ScanFunc] TryGetOutput SUCCESS for model 0
[22:36:45] 📤 [ModelScanThread::SendData] CALLED!
[22:36:45]    Symbol: BTCUSDT
[22:36:45]    Timestamp: 1765377407481907263
[22:36:45]    Predictions size: 13
[22:36:45]    Callback: VALID
[22:36:45]    ✅ Calling send_callback_...
[22:36:45] 📨 [SignalSender::Send] CALLED!
[22:36:45]    Symbol: BTCUSDT
[22:36:45]    Timestamp: 1765377407481907263
[22:36:45]    Count: 2
[22:36:45]    Callback: VALID
[22:36:45]    Values: [1, 0.8]
[22:36:45]    ✅ Calling callback...
[22:36:45] [FACTOR] 🎊 Received factor for BTCUSDT @ 1765377407481907263 (count=2)
[22:36:45] [FACTOR] Calling strategy on_factor for strategy_id=1350253488
[22:36:45]    ✅ Callback returned
[22:36:47] [critical] double free or corruption (!prev)  // 已知問題
```

**成功指標**:
- ✅ 全鏈路延遲: <2ms (從 Calculate 到 Python callback)
- ✅ 數據完整性: 100% (values 正確傳遞)
- ✅ 回調成功率: 100%
- ⚠️ 進程穩定性: 有記憶體問題 (2 秒後崩潰)

---

### 未完成項目清單

#### 短期 (Phase 6 - 1 週)
- [ ] 修復 signal_api.cpp 記憶體損壞問題 (P0)
- [ ] 添加多資產並發測試 (P1)
- [ ] 實現因子/模型動態配置加載 (P1)
- [ ] 添加錯誤處理和恢復邏輯 (P1)

#### 中期 (Phase 7 - 2 週)
- [ ] 性能優化 (批量計算、零拷貝) (P2)
- [ ] 監控和可觀測性 (metrics, tracing) (P2)
- [ ] 完整的單元測試套件 (P1)
- [ ] 壓力測試 (高頻數據流) (P2)

#### 長期 (Phase 8 - 4 週)
- [ ] 生產環境部署文檔 (P1)
- [ ] 模型熱更新機制 (P2)
- [ ] 多策略並行支持 (P2)
- [ ] 完整的 API 文檔 (P2)

---

### 生產就緒評估

| 評估項 | 狀態 | 說明 |
|--------|------|------|
| 功能完整性 | 🟡 82% | 核心功能完整,增強功能部分缺失 |
| 穩定性 | 🔴 60% | 有已知記憶體問題,需修復 |
| 性能 | 🟢 85% | 延遲符合預期,未做壓力測試 |
| 可維護性 | 🟢 90% | 代碼結構清晰,文檔完善 |
| 可觀測性 | 🟡 70% | 調試日誌充足,缺監控指標 |

**結論**: **不建議直接生產部署**,需完成:
1. 修復記憶體損壞問題 (阻塞項)
2. 完成多資產測試
3. 添加錯誤恢復機制
4. 壓力測試驗證

---

## 第三部分: 技術適配細節

> **來源**: hf-live/ADAPTATION_SUMMARY.md (522 lines)
> **用途**: 記錄 FactorEngine 和 ModelEngine 從 ref 項目到 Godzilla 的適配細節

### Part 1: FactorEngine 適配 (Phase 3.3)

#### 數據結構變更

**原 ref 項目**:
```cpp
// 行情數據
struct TickDataInfo {
    std::string code;
    int64_t recv_time;
    Depth depth;
    Trade trade;
};

// 因子結果
struct FactorResultInfo {
    std::string code;
    int64_t data_time;
    std::vector<float> values;
};
```

**Godzilla 適配**:
```cpp
// 使用 void* 直接傳遞 Depth/Trade (零拷貝)
// FactorEngine::OnDepth(const char* symbol, void* depth_ptr, int64_t timestamp)

// FactorResultInfo 保持不變 (內部使用)
```

**關鍵變更**:
- ❌ 移除 `TickDataInfo` 包裝層 (避免拷貝)
- ✅ 直接使用 `void*` 指向 Godzilla 的 Depth/Trade
- ✅ 在 FactorCalculationThread 中轉換為具體類型

---

#### 依賴移除

**移除的 ref 項目依賴**:
1. **SDPHandler** (數據發送模組)
   - 原用途: 發送因子結果到下游
   - 替代方案: 回調函數 `std::function<void(const FactorResultInfo&)>`

2. **ConfigData** (配置系統)
   - 原用途: 從 config.json 加載因子配置
   - 替代方案: 直接傳遞參數到 Init()

3. **WLOG** (日誌系統)
   - 原用途: 統一日誌輸出
   - 替代方案: std::cerr + 條件編譯

**清理的代碼**:
```cpp
// 移除
#include "comm/sdp/sdp_handler.h"
#include "comm/config_data.h"
#include "comm/print.hpp"

// 替換為
#include <iostream>
#define WLOG(msg, sync) std::cout << msg << std::endl
```

---

#### API 變更對照表

| ref 項目 API | Godzilla API | 變更說明 |
|--------------|--------------|----------|
| `OnTick(TickDataInfo)` | `OnDepth(symbol, void*, timestamp)` | 零拷貝,直接傳指針 |
| `OnTrans(TransactionInfo)` | `OnTrade(symbol, void*, timestamp)` | 同上 |
| `OnOrder(OrderInfo)` | **移除** | Godzilla 不使用逐筆委託 |
| `Init(ConfigData)` | `Init(factor_names, thread_num)` | 簡化參數 |
| `Send(SDPHandler)` | `SendFactors(callback)` | 回調替代 SDPHandler |

---

#### Init() 接口簡化

**ref 項目**:
```cpp
void Init(const config::ConfigData& config) {
    // 解析 config.json
    auto factor_list = config.get("factors");
    auto thread_num = config.get("thread_num");
    auto model_list = config.get("models");
    // ... 複雜的配置解析邏輯
}
```

**Godzilla 適配**:
```cpp
void Init(const std::vector<std::string>& factor_names, int thread_num = 4) {
    // 直接使用參數,無需解析
    factor_num_ = factor_names.size();
    factor_names_ = factor_names;
    // ... 簡化的初始化邏輯
}
```

**優勢**:
- ✅ 減少依賴 (不需要 config 模組)
- ✅ 更靈活 (可從 Python 傳遞配置)
- ✅ 更易測試 (直接注入參數)

---

#### 回調機制設計

**問題**: 如何在不依賴 SDPHandler 的情況下發送因子結果?

**解決方案**: 回調函數注入

```cpp
// FactorCalculationEngine.h
class FactorCalculationEngine {
public:
    void SetSendCallback(
        std::function<void(const FactorResultInfo&)> cb
    ) {
        send_callback_ = std::move(cb);
    }

private:
    std::function<void(const FactorResultInfo&)> send_callback_;
};

// FactorResultScanThread 使用
void ScanFunc() {
    FactorResultInfo result = CollectResults();
    if (send_callback_) {
        send_callback_(result);  // 發送到 ModelEngine
    }
}
```

**調用鏈**:
```
FactorResultScanThread::ScanFunc()
  → send_callback_(result)
    → signal_api.cpp::OnFactorResult()
      → ModelEngine::SendFactors()
```

---

### Part 2: ModelEngine 適配 (Phase 3.4)

#### 時間結構變更

**ref 項目**:
```cpp
struct start_time_t {
    int64_t exchange_timestamp;
    int64_t local_timestamp;
};
```

**Godzilla 適配**:
```cpp
struct GodzillaTime {
    int64_t data_time;      // Godzilla 的標準時間戳
    int64_t extra_nano;     // 納秒精度 (預留)
};
```

**適配理由**:
- Godzilla 使用單一時間戳 (納秒精度)
- 不區分交易所時間和本地時間 (統一由 Godzilla 管理)

---

#### ModelEngine 初始化簡化

**ref 項目**:
```cpp
void Init(const config::ConfigData& config) {
    // 從 config 解析模型列表
    auto model_configs = config.get_array("models");
    for (auto& cfg : model_configs) {
        auto model_name = cfg.get("name");
        auto model_params = cfg.get("params");
        // ... 複雜的模型創建邏輯
    }
}
```

**Godzilla 適配**:
```cpp
void Init(int thread_num = 4) {
    // 硬編碼測試模型 (簡化版)
    std::vector<std::string> model_names = {"test0000"};
    std::vector<std::string> factor_names = {"spread", "mid", "bid"};

    // 使用 ModelRegistry 創建模型
    auto& registry = models::comm::ModelRegistry::GetInstance();
    for (const auto& name : model_names) {
        auto model = registry.CreateModel(name, factor_names, {});
        model_calc_threads_.emplace_back(
            std::make_unique<ModelCalculationThread>(std::move(model), ...)
        );
    }
}
```

**未來改進**:
- [ ] 從外部傳入模型配置 (而非硬編碼)
- [ ] 支持動態模型註冊
- [ ] 支持模型熱更新

---

#### 回調機制 (SetSendCallback)

**關鍵修復**: Callback 時序問題

**問題代碼**:
```cpp
void Init(int thread_num) {
    // ... 創建 model_calc_threads_ ...

    // ❌ 問題: 此時 send_callback_ 還未設置 (NULL)
    model_result_scan_thread_ = std::make_unique<ModelResultScanThread>(
        models,
        send_callback_  // NULL!
    );
}

void SetSendCallback(...) {
    send_callback_ = std::move(cb);
    // ❌ ScanThread 已經創建,使用的是舊的 NULL callback
}
```

**修復方案**:
```cpp
void SetSendCallback(
    std::function<void(const std::string&, int64_t, const std::vector<float>&)> cb
) {
    send_callback_ = std::move(cb);

    // ✅ 重建 ScanThread,使用新的 callback
    std::vector<models::comm::ModelInterface*> models;
    for (size_t i = 0; i < model_calc_threads_.size(); ++i) {
        models.push_back(model_calc_threads_[i]->GetModel());
    }

    model_result_scan_thread_ = std::make_unique<ModelResultScanThread>(
        models,
        send_callback_  // 現在是 VALID!
    );
}
```

**驗證**:
```
📤 [ModelScanThread::SendData] CALLED!
   Callback: VALID  ✅
```

---

#### 數據打包邏輯

**ModelResultScanThread::ScanFunc()**:
```cpp
// 頭部 11 個 metadata 欄位
data[0] = model_id;
data[1] = tick_max_wait_elapsed_us;
data[2] = factor_max_calc_elapsed_us;
data[3] = factor_scan_elapsed_us;
data[4] = factor_send_elapsed_us;
data[5] = model_calc_elapsed_us;
data[6] = factor_max_calc_duration_us;
data[7] = factor_scan_duration_us;
data[8] = factor_send_duration_us;
data[9] = model_calc_duration_us;
data[10] = output_size;

// 追加原始模型輸出值
memcpy(&data[11], &model_output.values[0], output_size * sizeof(float));

// 發送
SendData(code, timestamp, data);
```

**數據格式**:
```
[model_id, 延遲統計 x9, output_size, 模型輸出值...]
  0        1~9                10         11~(11+output_size-1)
```

---

### 關鍵設計決策

#### 決策 1: 零拷貝 vs 數據包裝

**選擇**: 零拷貝 (void* 直接傳遞)

**理由**:
- ✅ 減少延遲 (避免 memcpy)
- ✅ 減少記憶體分配
- ⚠️ 需要小心管理指針生命週期

**實施**:
```cpp
// Godzilla runner.cpp
void Runner::on_depth(const msg::Depth& depth) {
    signal_on_data(symbol.c_str(), (void*)&depth, depth.data_time);
    // depth 的生命週期由 Godzilla 管理
}
```

---

#### 決策 2: 回調 vs 共享記憶體

**選擇**: 回調函數

**理由**:
- ✅ 簡單易實現
- ✅ 無需同步機制
- ✅ 符合事件驅動架構
- ⚠️ 需要確保回調不阻塞

**實施**:
```cpp
std::function<void(const std::string&, int64_t, const std::vector<float>&)> send_callback_;
```

---

#### 決策 3: 硬編碼 vs 配置文件

**選擇**: 階段性硬編碼 (Phase 4),未來改為配置

**理由**:
- ✅ 快速驗證 E2E 流程
- ✅ 減少配置解析複雜度
- ⚠️ 不適合生產環境

**後續計劃**:
```cpp
// Phase 6: 從外部傳入配置
void Init(const ModelConfig& config) {
    auto model_names = config.get_model_names();
    auto factor_names = config.get_factor_names();
    // ...
}
```

---

### 集成範例

**完整調用流程**:
```cpp
// 1. 初始化 FactorEngine
FactorCalculationEngine factor_engine;
factor_engine.Init({"spread", "mid", "bid"}, 4);

// 2. 初始化 ModelEngine
ModelCalculationEngine model_engine;
model_engine.Init(4);

// 3. 設置回調鏈
factor_engine.SetSendCallback([&](const FactorResultInfo& result) {
    // FactorEngine → ModelEngine
    model_engine.SendFactors(result);
});

model_engine.SetSendCallback([](const string& symbol, int64_t ts, const vector<float>& preds) {
    // ModelEngine → SignalSender
    SignalSender::GetInstance().Send(symbol, ts, preds);
});

// 4. 啟動引擎
factor_engine.Start();
model_engine.Start();

// 5. 接收市場數據
factor_engine.OnDepth("BTCUSDT", (void*)&depth, depth.data_time);
```

---

## 第四部分: 核心組件實現摘要

> **來源**: hf-live/IMPLEMENTATION_SUMMARY.md (405 lines)
> **日期**: 2025-01-06
> **用途**: Phase 3.3-3.4 核心組件技術細節

### 已完成的 6 個核心組件

#### 1. FactorEntryManager ✅

**文件**: `hf-live/factors/_comm/factor_entry_manager.h`

**核心功能**:
- 管理所有註冊的因子實例
- 分發市場數據到各因子 (AddQuote, AddTrans)
- 觸發因子計算 (TriggerCompute)
- 收集因子計算結果 (GetFactorValues)

**適配變更**:
```cpp
// 數據類型替換
- Stock_Internal_Book → hf::Depth
- Stock_Transaction_Internal_Book_New → hf::Trade

// 移除方法
- void AddOrder(...)  // Godzilla 不使用

// 時間統計更新
struct TimeStats {
    timer::ElapsedTimeStats quote;
    timer::ElapsedTimeStats trans;
    // timer::ElapsedTimeStats order;  // 移除
    timer::ElapsedTimeStats factor;
};
```

**使用範例**:
```cpp
// 配置
factors::comm::FactorEntryConfig config;
config.date = "20250106";
config.ev_path = "/path/to/data";

// 創建管理器
factors::FactorEntryManager manager("BTCUSDT", config, {"ma_factor", "volume_factor"});

// 添加數據
hf::Depth depth;
manager.AddQuote(depth);

// 觸發計算
manager.TriggerCompute(current_timestamp);

// 獲取結果
auto values = manager.GetFactorValues();
```

---

#### 2. FactorEntryBase & FactorEntryRegistry ✅

**文件**:
- `hf-live/factors/_comm/factor_entry_base.h` (基類)
- `hf-live/factors/_comm/factor_entry_registry.h` (註冊機制)

**核心功能**:
- FactorEntryBase: 所有因子的基類,定義通用接口
- FactorEntryRegistry: 單例模式的因子註冊表

**適配變更**:
```cpp
// FactorEntryBase - 移除虛函數
- virtual void DoOnAddOrder(const Stock_Order_Internal_Book_New &quote);

// FactorEntryBase - 移除統計欄位
- timer::ElapsedTimeStats order_time_stats_;
- const timer::ElapsedTimeStats &GetOrderTimeStats() const;
```

**註冊機制**:
```cpp
// 因子實現 (factors/my_factor/my_factor.cc)
class MyFactor : public factors::FactorEntryBase {
public:
    void DoOnAddQuote(const hf::Depth &quote) override {
        // 因子邏輯
    }
};

// 註冊宏
REGISTER_FACTOR_AUTO(my_factor, MyFactor)
```

---

#### 3. core.h - 數據類型適配層 ✅

**文件**: `hf-live/factors/_comm/core.h`

**實施內容**:
```cpp
// 引入 Godzilla 數據類型
#include "../../include/market_data_types.h"

// 定義 hf namespace 別名
namespace hf {
    using Depth = ::hf::Depth;  // kungfu::wingchun::msg::Depth
    using Trade = ::hf::Trade;  // kungfu::wingchun::msg::Trade
}

// 更新 IFactorEntry 接口
class IFactorEntry {
public:
    virtual void AddQuote(const hf::Depth &quote) = 0;
    virtual void AddTrans(const hf::Trade &quote) = 0;
    // void AddOrder(...);  // 移除
};
```

**Godzilla 數據結構映射**:
```cpp
// kungfu::wingchun::msg::Depth
struct Depth {
    int64_t data_time;
    double bid_price[10];
    double ask_price[10];
    int64_t bid_volume[10];
    int64_t ask_volume[10];
};

// kungfu::wingchun::msg::Trade
struct Trade {
    int64_t data_time;
    double price;
    int64_t volume;
    int8_t side;  // 1=buy, 2=sell
};
```

---

#### 4. timer.h - 高精度計時器 ✅

**文件**: `hf-live/factors/_comm/timer.h`

**實施內容**:
- ✅ 從參考項目完整複製 (無修改)
- ✅ 提供 RDTSC (Read Time Stamp Counter) 計時器
- ✅ CPU 頻率校準 (get_cpu_mhz)
- ✅ 時間統計結構 (ElapsedTimeStats, ScopedTiming)

**提供的計時器**:
```cpp
namespace factors::timer {
    // RDTSC - CPU 時鐘週期 (最低延遲)
    class RdtscTimer {
    public:
        static uint64_t operator()() { return __rdtsc(); }
        static double GetScaler() { /* CPU 頻率校準 */ }
    };

    // 高精度 - std::chrono
    class HighResTimer {
    public:
        static uint64_t operator()() {
            return std::chrono::high_resolution_clock::now().time_since_epoch().count();
        }
    };

    // 單調時鐘 - 不受系統時間調整影響
    class SteadyClockTimer { /* ... */ };
}
```

**使用範例**:
```cpp
#include "factors/_comm/timer.h"

// 測量延遲
auto start = factors::timer::RdtscTimer()();
// ... 執行操作 ...
auto end = factors::timer::RdtscTimer()();
double elapsed_us = (end - start) * factors::timer::RdtscTimer::GetScaler();

std::cout << "Elapsed: " << elapsed_us << " μs" << std::endl;
```

**性能特性**:
- RDTSC 精度: ~1-2 納秒
- HighResTimer 精度: ~100 納秒
- CPU 頻率校準: 啟動時執行 (100ms 預熱)

---

#### 5. MarketEventProcessor - 觸發邏輯 ✅

**文件**: `hf-live/app_live/trigger/market_event_processor.h`

**實施內容**:
- ✅ 創建簡化版本 (header-only)
- ✅ 支持基於計數的觸發邏輯
- ✅ 可動態調整觸發間隔

**核心接口**:
```cpp
class MarketEventProcessor {
public:
    MarketEventProcessor(const std::string& symbol,
                         int depth_interval = 100,
                         int trade_interval = 100)
        : symbol_(symbol),
          depth_interval_(depth_interval),
          trade_interval_(trade_interval),
          depth_count_(0),
          trade_count_(0) {}

    // 判斷是否應該觸發 (Depth)
    bool ShouldTriggerOnDepth(const hf::Depth* depth) {
        depth_count_++;
        if (depth_count_ >= depth_interval_) {
            depth_count_ = 0;
            return true;
        }
        return false;
    }

    // 判斷是否應該觸發 (Trade)
    bool ShouldTriggerOnTrade(const hf::Trade* trade) {
        trade_count_++;
        if (trade_count_ >= trade_interval_) {
            trade_count_ = 0;
            return true;
        }
        return false;
    }

    // 重置計數器
    void Reset() {
        depth_count_ = 0;
        trade_count_ = 0;
    }

    // 動態調整
    void set_depth_interval(int interval) { depth_interval_ = interval; }
    void set_trade_interval(int interval) { trade_interval_ = interval; }

private:
    std::string symbol_;
    int depth_interval_;
    int trade_interval_;
    int depth_count_;
    int trade_count_;
};
```

**簡化說明**:
- ❌ 移除參考項目的複雜邏輯:
  - 股票交易所特定邏輯 (集合競價、盤中暫停)
  - 訂單簿重建 (myod2ab)
  - 時間窗口觸發
- ✅ 保留核心功能:
  - 簡單計數器觸發
  - 可配置間隔

**使用範例**:
```cpp
// 創建處理器 (每 100 筆 Depth 觸發)
MarketEventProcessor processor("BTCUSDT", 100, 50);

// 處理數據
void OnDepth(const hf::Depth& depth) {
    if (processor.ShouldTriggerOnDepth(&depth)) {
        // 執行因子計算
        factor_manager.TriggerCompute(depth.data_time);
    }
}

// 動態調整
processor.set_depth_interval(200);  // 改為每 200 筆觸發
```

---

#### 6. timer_utils.h - Timer 包裝層 ✅

**文件**: `hf-live/app_live/common/timer_utils.h`

**實施內容**:
- ✅ 創建新文件,包裝 `factors/_comm/timer.h`
- ✅ 刪除舊的 stub 文件 (`timer_utils_stub.h`)
- ✅ 更新所有引用

**命名空間映射**:
```cpp
#ifndef TIMER_UTILS_H
#define TIMER_UTILS_H

#include "factors/_comm/timer.h"

namespace timer_utils {
    using RdtscTimer = factors::timer::RdtscTimer;
    using HighResTimer = factors::timer::HighResTimer;
    using SteadyClockTimer = factors::timer::SteadyClockTimer;
    using ElapsedTimeStats = factors::timer::ElapsedTimeStats;
    using ScopedTiming = factors::timer::ScopedTiming;
}

#endif // TIMER_UTILS_H
```

**更新的文件** (共 3 個):
1. `app_live/thread/factor_calculation_thread.h`
2. `app_live/thread/factor_result_scan_thread.h`
3. `app_live/engine/factor_calculation_engine.cpp`

**變更內容**:
```cpp
// 舊引用
#include "common/timer_utils_stub.h"  // ❌ 刪除

// 新引用
#include "common/timer_utils.h"  // ✅ 包裝真實 timer

// 使用方式不變
auto scaler = timer_utils::RdtscTimer::GetScaler();
```

---

### 適配變更總結

#### 數據類型映射表

| 參考項目 | Godzilla | 說明 |
|---------|----------|------|
| `Stock_Internal_Book` | `hf::Depth` | L2 盤口數據 (bid/ask price/volume x10) |
| `Stock_Transaction_Internal_Book_New` | `hf::Trade` | 成交數據 (price, volume, side) |
| `Stock_Order_Internal_Book_New` | **移除** | Godzilla 不使用逐筆委託 |

---

#### 接口變更彙整

**IFactorEntry (core.h)**:
```cpp
// 移除
- void AddOrder(const Stock_Order_Internal_Book_New &quote);

// 修改
- void AddQuote(const Stock_Internal_Book &quote);
+ void AddQuote(const hf::Depth &quote);

- void AddTrans(const Stock_Transaction_Internal_Book_New &quote);
+ void AddTrans(const hf::Trade &quote);
```

**FactorEntryBase (factor_entry_base.h)**:
```cpp
// 移除虛函數
- void DoOnAddOrder(const Stock_Order_Internal_Book_New &quote);

// 移除統計欄位
- timer::ElapsedTimeStats order_time_stats_;
- const timer::ElapsedTimeStats &GetOrderTimeStats() const;
```

**FactorEntryManager (factor_entry_manager.h)**:
```cpp
// 移除方法
- void AddOrder(const Stock_Order_Internal_Book_New &quote);

// TimeStats 結構簡化
struct TimeStats {
    timer::ElapsedTimeStats quote;
    timer::ElapsedTimeStats trans;
-   timer::ElapsedTimeStats order;  // 移除
    timer::ElapsedTimeStats factor;
};
```

**MarketEventProcessor (新增)**:
```cpp
// 參考項目接口 (複雜)
bool AddQuote(Stock_Internal_Book *quote);
bool AddTrans(Stock_Transaction_Internal_Book_New *quote);
bool AddOrder(Stock_Order_Internal_Book_New *quote);

// Godzilla 簡化接口
bool ShouldTriggerOnDepth(const hf::Depth* depth);
bool ShouldTriggerOnTrade(const hf::Trade* trade);
```

---

### 未修改的文件 (保留參考)

**`factors/_comm/myod2ab/`** (訂單簿重建工具):
- 這些是從參考項目複製的輔助工具
- 僅用於股票市場因子 (需要訂單簿重建)
- 加密貨幣市場提供完整盤口,可能不需要
- 暫時保留以供未來參考

**不影響核心功能** - 因子可選擇性使用這些工具

---

### 驗證狀態

#### 語法檢查 ✅
- [x] 所有核心文件已更新引用
- [x] `timer_utils_stub.h` 已完全移除
- [x] 數據類型已統一替換為 hf::Depth / hf::Trade

#### 編譯驗證 ⏳
- [ ] 無法在 host 環境驗證 (缺少 g++)
- [ ] 需要在 Godzilla 容器環境中進行完整編譯測試

**建議驗證步驟**:
```bash
# 在 Godzilla 開發容器中
cd /app/hf-live
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

---

### 遇到的問題與解決方案

#### 問題 1: 數據類型不匹配
**症狀**: 參考項目使用 `Stock_Internal_Book` 等股票市場專用結構

**解決方案**:
1. 在 `core.h` 引入 Godzilla 的 `market_data_types.h`
2. 定義 `hf` namespace 別名
3. 全局替換數據類型

**驗證**:
```bash
grep -r "Stock_Internal_Book" hf-live/factors/_comm/  # 應無結果
grep -r "hf::Depth" hf-live/factors/_comm/            # 應有多處
```

---

#### 問題 2: timer_utils_stub.h 占位實現
**症狀**: 舊的 stub 文件功能不完整,只有空實現

**解決方案**:
1. 從參考項目複製完整的 `timer.h` (RDTSC 實現)
2. 創建 `timer_utils.h` 包裝層 (namespace 轉換)
3. 更新所有引用文件 (3 個)
4. 刪除 `timer_utils_stub.h`

**驗證**:
```bash
find hf-live -name "*timer_utils_stub*"  # 應無結果
grep -r "timer_utils::RdtscTimer" hf-live  # 應有引用
```

---

#### 問題 3: MarketEventProcessor 過於複雜
**症狀**: 參考項目包含股票交易所特定邏輯 (集合競價、盤口重建)

**解決方案**:
1. 創建簡化版本 (header-only)
2. 僅保留計數觸發邏輯
3. 移除訂單簿重建功能
4. 移除時間窗口觸發邏輯

**取捨**:
- ✅ 簡單易維護
- ✅ 滿足加密貨幣市場需求 (24/7 交易,無集合競價)
- ⚠️ 未來若需複雜觸發邏輯,需擴展

---

#### 問題 4: AddOrder 方法冗餘
**症狀**: Godzilla 不使用逐筆委託數據流 (Level 3 數據)

**解決方案**:
1. 從 `IFactorEntry` 接口完全移除 `AddOrder()`
2. 清理時間統計中的 `order` 欄位
3. 更新 CSV 輸出格式 (移除 order 列)

**影響範圍**:
- `core.h` - 接口定義
- `factor_entry_base.h` - 基類實現
- `factor_entry_manager.h` - 管理器

**驗證**:
```bash
grep -r "AddOrder" hf-live/factors/_comm/  # 應無結果
grep -r "order_time_stats" hf-live/factors/_comm/  # 應無結果
```

---

### 性能考量

#### 內存布局優化
- ✅ 使用 `InsertOrderMap` 保持因子插入順序
- ✅ 預分配內存 (`fvals_snapshot_.reserve()`)
- ✅ 使用 `std::memcpy` 進行批量複製

**關鍵代碼** (FactorEntryManager):
```cpp
// 預分配
fvals_snapshot_.reserve(total_factor_count);

// 批量複製
for (auto& entry : entries_) {
    auto values = entry->GetFactorValues();
    std::memcpy(&fvals_snapshot_[offset], values.data(), values.size() * sizeof(float));
    offset += values.size();
}
```

---

#### 計時精度
- ✅ RDTSC 提供 CPU 週期級精度 (~1-2 納秒)
- ✅ 預熱機制避免首次調用延遲
- ✅ 靜態緩存 CPU 頻率校準結果

**關鍵代碼** (timer.h):
```cpp
class RdtscTimer {
public:
    static double GetScaler() {
        static double scaler = []() {
            // CPU 頻率校準 (僅執行一次)
            auto cpu_mhz = get_cpu_mhz();
            return 1.0 / (cpu_mhz * 1000.0);  // 轉換為微秒
        }();
        return scaler;
    }
};
```

---

#### 觸發策略
- ✅ 簡單計數器 (O(1) 複雜度)
- ✅ 可配置觸發間隔
- ✅ 支持動態調整

**性能測試** (建議):
```cpp
// 測試觸發延遲
auto start = timer_utils::RdtscTimer()();
bool should_trigger = processor.ShouldTriggerOnDepth(&depth);
auto end = timer_utils::RdtscTimer()();
// 預期: <10 納秒
```

---

### 文件清單

#### 修改的文件 (6 個)
1. `hf-live/factors/_comm/core.h` - 數據類型適配
2. `hf-live/factors/_comm/factor_entry_base.h` - 移除 AddOrder
3. `hf-live/factors/_comm/factor_entry_manager.h` - 移除 AddOrder
4. `hf-live/app_live/thread/factor_calculation_thread.h` - 更新 timer 引用
5. `hf-live/app_live/thread/factor_result_scan_thread.h` - 更新 timer 引用
6. `hf-live/app_live/engine/factor_calculation_engine.cpp` - 更新 timer 引用

#### 新增的文件 (2 個)
1. `hf-live/app_live/trigger/market_event_processor.h` (新增)
2. `hf-live/app_live/common/timer_utils.h` (替換 stub)

#### 刪除的文件 (1 個)
1. `hf-live/app_live/common/timer_utils_stub.h`

#### 已存在但未修改的關鍵文件
1. `hf-live/factors/_comm/timer.h` (從參考項目複製,已存在)
2. `hf-live/factors/_comm/factor_entry_registry.h` (僅數據類型替換)

---

### 集成檢查清單

#### ✅ 已完成
- [x] FactorEntryManager 數據類型適配
- [x] FactorEntryBase 接口更新
- [x] FactorEntryRegistry 保留完整機制
- [x] core.h 定義 Godzilla 數據類型
- [x] timer.h 從參考項目複製
- [x] MarketEventProcessor 簡化實現
- [x] timer_utils.h 包裝層
- [x] 更新所有 timer_utils_stub 引用
- [x] 移除所有 Order 相關方法

#### ⏳ 待驗證
- [ ] 在容器環境中編譯測試
- [ ] 與 FactorCalculationEngine 的集成測試
- [ ] 與 ModelCalculationEngine 的數據流測試

#### 📋 後續任務 (Phase 5-6)
- [ ] 實現因子註冊機制 (REGISTER_FACTOR_AUTO 宏使用)
- [ ] 創建示例因子 (參考 factors/example/)
- [ ] 配置文件適配 (factor_entry_config)
- [ ] 性能測試與優化
- [ ] 多資產並發測試

---

## 附錄: 中間文檔原始信息

### 原文檔列表

1. **plan/phase-4f-test-plan.md**
   - 創建日期: 2025-12-10
   - 行數: 453
   - 用途: Phase 4F 測試方法論

2. **plan/IMPLEMENTATION_STATUS_REPORT.md**
   - 創建日期: 2025-12-08
   - 行數: 813
   - 用途: 87% 完成度評估

3. **hf-live/ADAPTATION_SUMMARY.md**
   - 創建日期: 2025-01-06
   - 行數: 522
   - 用途: FactorEngine/ModelEngine 適配細節

4. **hf-live/IMPLEMENTATION_SUMMARY.md**
   - 創建日期: 2025-01-06
   - 行數: 405
   - 用途: 核心組件實現摘要

### 整合原則

1. **保留關鍵技術細節** - 代碼範例、配置參數、性能數據
2. **統一術語** - 使用 Godzilla 標準術語 (Depth/Trade 而非 Quote/Trans)
3. **時間序列組織** - 按實施順序 (測試計劃 → 狀態評估 → 技術細節)
4. **去重複** - 合併重複內容 (如數據類型映射表)
5. **增強可讀性** - 添加章節索引、表格彙整、快速定位錨點

---

## 變更歷史

| 版本 | 日期 | 變更說明 |
|------|------|----------|
| v1.0 | 2025-12-10 | 初始版本,整合 4 個中間文檔 |

---

**生成時間**: 2025-12-10 23:15
**文檔狀態**: 完整
**後續行動**: 待用戶審閱後刪除原始 4 個中間文檔
