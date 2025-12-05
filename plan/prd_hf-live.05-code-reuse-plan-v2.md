# hf-live 代碼復用計劃 (完整版)

## 文檔元信息
- **版本**: v2.0 (完整版 - 包含 Model)
- **日期**: 2025-12-03
- **目標**: 逐一盤點 ref/hf-stock-live-demo-main 可復用的代碼資源 (含 Factor + Model 完整流程)
- **前置**: [prd_hf-live.04-project-config.md](prd_hf-live.04-project-config.md)

---

## 一、業務邏輯修正

### 1.1 錯誤理解 vs 正確理解

**❌ 錯誤理解 (之前的版本)**:
```
Factor → 直接發送給策略大師
```

**✅ 正確理解 (ref 項目實際邏輯)**:
```
行情數據 (Depth/Trade)
  ↓
FactorCalculationEngine (多線程計算因子)
  ↓
FactorResultScanThread (收集因子結果)
  ↓
【分支點】🔥
  ├─→ 選項 A: 直接發送因子 (可選)
  │     └─→ SignalSender::Send(factors) → on_factor(factors)
  │
  └─→ 選項 B: 經過模型預測 (常規做法) 🔥🔥🔥
        ↓
      ModelCalculationEngine::SendFactors(factors)
        ↓
      ModelCalculationThread (多線程 ONNX/自定義模型)
        ↓
      ModelResultScanThread (收集模型預測值)
        ↓
      SignalSender::Send(predictions) → on_factor(predictions)
        ↑
      策略大師收到的是預測值,但他不知道!
      (on_factor 既可以接收原始因子,也可以接收模型預測值)
```

### 1.2 核心認知修正

| 之前理解 | 正確理解 |
|---------|---------|
| ❌ Model 是可選的,未來才需要 | ✅ Model 是**核心組件**,常規流程必備 |
| ❌ on_factor 只接收因子 | ✅ on_factor 既可接收因子,也可接收預測值 |
| ❌ 策略大師知道收到的是什麼 | ✅ 策略大師**不知道**收到的是因子還是預測值 |
| ❌ FactorResultScanThread 直接發送 | ✅ FactorResultScanThread 發送到 ModelCalculationEngine |
| ❌ Model 相關代碼可以刪除 | ✅ Model 相關代碼必須 80%+ 復用 🔥 |

---

## 二、ref 項目完整結構 (含 Model)

### 2.1 完整數據流

```
[1] 行情數據
      ↓
[2] FactorCalculationEngine::OnTick/OnTrans/OnOrder
      ↓ (SPMC Buffer)
[3] FactorCalculationThread[n] (多線程計算)
      ↓ (SPSC Queue)
[4] FactorResultScanThread::CollectAndSend()
      ↓
[5] 【分支點】
      ├─→ 直接發送: sdp_handler->send_factor_v2(factors)
      └─→ 發送到模型: model_calc_engine->SendFactors(factors) 🔥
            ↓ (SPMC Buffer)
      [6] ModelCalculationThread[n] (ONNX 預測)
            ↓ (SPSC Queue)
      [7] ModelResultScanThread::ScanFunc()
            ↓
      [8] sdp_handler->send_factor_v2(predictions) 🔥
            ↓
      [9] 策略大師: on_factor(predictions)
```

### 2.2 關鍵組件關係

```cpp
// ref/app_live/strategy.cpp
FactorCalculationEngine* g_factor_calc_engine;
ModelCalculationEngine* g_model_calc_engine;  // 🔥 關鍵組件!

int my_st_init_v3(...) {
    g_factor_calc_engine = new FactorCalculationEngine(config);
    g_model_calc_engine = new ModelCalculationEngine();       // 🔥 必須初始化
    g_model_calc_engine->Init(date, thread_num, config, sdp_handler);

    g_factor_calc_engine->Start();
    g_model_calc_engine->Start();  // 🔥 啟動模型計算線程
}
```

---

## 三、復用策略分類 (修正版)

### 3.1 分類標準 (與之前相同)

| 類別 | 標準 | 復用率 | 示例 |
|------|------|--------|------|
| **A. 直接復制** | 無需修改,100% 復用 | 100% | utils, timer, 基礎數據結構 |
| **B. 微調復用** | 修改少量參數/命名 | 80-95% | CMakeLists, factor_entry_base |
| **C. 適配復用** | 保留結構,替換數據類型 | 50-80% | engine, threads |
| **D. 重寫** | 僅參考設計,重新實現 | <50% | entry.cpp, signal_sender |
| **E. 刪除** | 不需要的功能 | 0% | HDF5 |

### 3.2 復用統計 (修正版)

| 模塊 | 之前理解 | 正確理解 |
|------|---------|---------|
| **factors/** | 復用 90% ✅ | 復用 90% ✅ (無變化) |
| **models/** | ❌ 刪除,未來可能需要 | ✅ **復用 75%+** 🔥🔥🔥 |
| **app_live/engine/** | 只需 factor_calculation_engine | ✅ **需要兩個 engine** 🔥 |
| **app_live/thread/** | 只需 factor_*_thread | ✅ **需要 4 個 thread** 🔥 |

---

## 四、models/ 復用計劃 (新增完整章節)

### 4.1 models/ 目錄結構

```
ref/hf-stock-live-demo-main/models/
├── CMakeLists.txt                    # 模型構建配置
├── ModelModuleTemplate.cmake         # 模型模板
│
├── _comm/                            # 🔥 模型基礎框架 (必須復用)
│   ├── model_base.h                  # 🔥 ModelInterface 接口
│   ├── model_registry.h              # 🔥 模型註冊器
│   ├── model_display.cc              # 模型顯示
│   ├── timer.h                       # 計時器
│   └── spsc_queue_for_model_use.h    # 模型專用隊列
│
├── demo0000/                         # 示例模型 1 (ONNX)
│   ├── model.h
│   ├── model.cc
│   ├── meta_config.h
│   └── CMakeLists.txt
│
└── demo0001/                         # 示例模型 2
    └── ...
```

### 4.2 models/_comm/ 復用計劃

#### 4.2.1 直接復制 (A)

| ref 文件 | hf-live 目標 | 復用率 | 說明 |
|---------|-------------|--------|------|
| `_comm/timer.h` | `models/_comm/timer.h` | **100%** | 計時器 |
| `_comm/model_display.cc` | `models/_comm/model_display.cc` | **100%** | 模型顯示 |
| `_comm/spsc_queue_for_model_use.h` | `models/_comm/spsc_queue_for_model_use.h` | **100%** | 隊列 |

**操作**:
```bash
cp -r ref/hf-stock-live-demo-main/models/_comm \
      hf-live/models/_comm
```

#### 4.2.2 微調復用 (B)

| ref 文件 | hf-live 目標 | 復用率 | 修改內容 |
|---------|-------------|--------|---------|
| `_comm/model_base.h` | `models/_comm/model_base.h` | **90%** | 🔥 刪除 SDPHandler 引用 |
| `_comm/model_registry.h` | `models/_comm/model_registry.h` | **95%** | 保留註冊邏輯 |

**關鍵修改 (model_base.h)**:
```cpp
// ref/models/_comm/model_base.h
#include "sdp_handler/core/sdp_handler.h"  // ❌ 刪除

namespace models {
namespace comm {

struct input_t {
    // ... (保留完整結構)
    size_t item_size;
    std::vector<char> factor_datas;  // 🔥 因子數據
    std::vector<std::string> assets;
    start_time_t start_time;         // ❌ sdp 平台時間結構
    uint64_t start_tsc;
    // ... timing fields
};

struct output_t {
    // ... (保留完整結構)
    std::vector<pval_t> values;      // 🔥 預測值
    std::vector<std::string> assets;
    start_time_t start_time;         // ❌ 需要適配
    uint64_t start_tsc;
    // ... timing fields
};

class ModelInterface {
public:
    virtual ~ModelInterface() = default;

    // 🔥 核心接口 (100% 保留)
    virtual void SendInput(const input_t& input) = 0;
    virtual bool TryGetOutput(output_t& output) = 0;
    virtual bool IsOutputEmpty() const = 0;
    virtual size_t GetOutputSize() const = 0;
    virtual std::vector<std::string> GetOutputNames() const = 0;
};

} // namespace comm
} // namespace models

// hf-live/models/_comm/model_base.h
// #include "sdp_handler/core/sdp_handler.h"  // ✅ 刪除

namespace models {
namespace comm {

// 🔥 適配時間結構
struct GodzillaTime {
    int64_t data_time;    // Godzilla 時間 (納秒)
    int64_t local_time;   // 本地時間 (納秒)
};

struct input_t {
    size_t item_size;
    std::vector<char> factor_datas;  // ✅ 保留
    std::vector<std::string> assets; // ✅ 保留
    GodzillaTime start_time;         // ✅ 適配時間結構
    uint64_t start_tsc;              // ✅ 保留
    // ... (保留所有 timing fields)
};

struct output_t {
    std::vector<pval_t> values;      // ✅ 保留
    std::vector<std::string> assets; // ✅ 保留
    GodzillaTime start_time;         // ✅ 適配時間結構
    uint64_t start_tsc;              // ✅ 保留
    // ... (保留所有 timing fields)
};

class ModelInterface {
    // ✅ 100% 保留接口定義
};

} // namespace comm
} // namespace models
```

### 4.3 app_live/engine/model_calculation_engine.* 復用

#### 4.3.1 適配復用 (C)

| ref 文件 | hf-live 目標 | 復用率 | 修改內容 |
|---------|-------------|--------|---------|
| `model_calculation_engine.h` | `app_live/engine/model_calculation_engine.h` | **80%** | 刪除 SDPHandler 參數 |
| `model_calculation_engine.cc` | `app_live/engine/model_calculation_engine.cc` | **75%** | 適配初始化邏輯 |

**關鍵修改**:
```cpp
// ref/app_live/engine/model_calculation_engine.h
class ModelCalculationEngine {
public:
    void Init(const std::string& date,
              int thread_num,
              const config::ConfigData& config,
              SDPHandler* sdp_handler);  // ❌ sdp 平台連接器

    void SendFactors(const models::comm::input_t& input_data);  // ✅ 保留
};

// hf-live/app_live/engine/model_calculation_engine.h
class ModelCalculationEngine {
public:
    void Init(const std::string& date,
              int thread_num,
              const config::ConfigData& config);  // ✅ 刪除 sdp_handler

    void SendFactors(const models::comm::input_t& input_data);  // ✅ 保留

    // 🔥 新增: 設置發送回調
    void SetSendCallback(std::function<void(const char*, int64_t, const std::vector<float>&)> cb);
};
```

### 4.4 app_live/thread/model_*_thread.h 復用

#### 4.4.1 適配復用 (C)

| ref 文件 | hf-live 目標 | 復用率 | 修改內容 |
|---------|-------------|--------|---------|
| `model_calculation_thread.h` | `app_live/thread/model_calculation_thread.h` | **85%** | 🔥 保留多線程邏輯 |
| `model_result_scan_thread.h` | `app_live/thread/model_result_scan_thread.h` | **75%** | 🔥 替換發送接口 |

**關鍵修改 (model_result_scan_thread.h)**:
```cpp
// ref/app_live/thread/model_result_scan_thread.h
class ModelResultScanThread {
public:
    ModelResultScanThread(
        const std::vector<models::comm::ModelInterface*>& models,
        SDPHandler* sdp_handler  // ❌ sdp 平台連接器
    );

private:
    void SendData(const char *ticker, start_time_t *t,
                  std::vector<models::pval_t> &data) {
        if (SDPHandler* p = sdp_handler_) {
            p->send_factor_v2(ticker, t, data);  // ❌ sdp 平台 API
        }
    }

    SDPHandler* sdp_handler_;  // ❌
};

// hf-live/app_live/thread/model_result_scan_thread.h
class ModelResultScanThread {
public:
    ModelResultScanThread(
        const std::vector<models::comm::ModelInterface*>& models,
        std::function<void(const char*, int64_t, const std::vector<float>&)> send_callback  // ✅ 回調
    );

private:
    void SendData(const char *symbol, int64_t timestamp,
                  std::vector<models::pval_t> &data) {
        if (send_callback_) {
            send_callback_(symbol, timestamp, data);  // ✅ SignalSender::Send()
        }
    }

    std::function<void(const char*, int64_t, const std::vector<float>&)> send_callback_;  // ✅
};
```

### 4.5 models/demo0000/ (示例模型) 復用

#### 4.5.1 適配復用 (C)

| ref 文件 | hf-live 目標 | 復用率 | 說明 |
|---------|-------------|--------|------|
| `demo0000/model.h` | `models/demo/model.h` | **90%** | ONNX 模型封裝 |
| `demo0000/model.cc` | `models/demo/model.cc` | **90%** | ONNX Runtime 調用 |
| `demo0000/meta_config.h` | `models/demo/meta_config.h` | **95%** | 模型元信息 |
| `demo0000/CMakeLists.txt` | `models/demo/CMakeLists.txt` | **90%** | 構建配置 |

**關鍵**: ONNX Runtime 邏輯完全可復用,僅需調整輸入輸出接口

---

## 五、完整復用計劃 (修正版)

### 5.1 sdp_handler/ → hf-live/handler/

**(與之前版本相同,無變化)**

### 5.2 app_live/ → hf-live/app_live/ (修正版)

#### 5.2.1 engine/ 復用

| ref 文件 | hf-live 目標 | 復用率 | 修改 |
|---------|-------------|--------|------|
| `factor_calculation_engine.h` | `app_live/engine/factor_calculation_engine.h` | **70%** | 替換數據類型 |
| `factor_calculation_engine.cpp` | `app_live/engine/factor_calculation_engine.cpp` | **65%** | OnTick → OnDepth |
| `model_calculation_engine.h` | `app_live/engine/model_calculation_engine.h` | **80%** | 🔥 刪除 SDPHandler |
| `model_calculation_engine.cc` | `app_live/engine/model_calculation_engine.cc` | **75%** | 🔥 適配發送回調 |

#### 5.2.2 thread/ 復用 (修正版)

| ref 文件 | hf-live 目標 | 復用率 | 說明 |
|---------|-------------|--------|------|
| `factor_calculation_thread.h` | `app_live/thread/factor_calculation_thread.h` | **75%** | 保留多線程邏輯 |
| `factor_result_scan_thread.h` | `app_live/thread/factor_result_scan_thread.h` | **70%** | 🔥 發送到 ModelEngine |
| `model_calculation_thread.h` | `app_live/thread/model_calculation_thread.h` | **85%** | 🔥 保留 ONNX 邏輯 |
| `model_result_scan_thread.h` | `app_live/thread/model_result_scan_thread.h` | **75%** | 🔥 替換發送接口 |
| `thread_allocator.hpp` | `app_live/thread/thread_allocator.hpp` | **100%** | 直接復制 |

### 5.3 factors/ → hf-live/factors/

**(與之前版本相同,無變化)**

### 5.4 models/ → hf-live/models/ (新增完整章節)

| ref 文件 | hf-live 目標 | 復用率 | 說明 |
|---------|-------------|--------|------|
| `CMakeLists.txt` | `models/CMakeLists.txt` | **90%** | 🔥 模型構建配置 |
| `ModelModuleTemplate.cmake` | `models/ModelModuleTemplate.cmake` | **95%** | 🔥 模型模板 |
| `_comm/model_base.h` | `models/_comm/model_base.h` | **90%** | 🔥 核心接口 |
| `_comm/model_registry.h` | `models/_comm/model_registry.h` | **95%** | 🔥 模型註冊器 |
| `_comm/model_display.cc` | `models/_comm/model_display.cc` | **100%** | 模型顯示 |
| `_comm/timer.h` | `models/_comm/timer.h` | **100%** | 計時器 |
| `_comm/spsc_queue_for_model_use.h` | `models/_comm/spsc_queue_for_model_use.h` | **100%** | 隊列 |
| `demo0000/*` | `models/demo/*` | **90%** | 🔥 ONNX 示例模型 |

---

## 六、完整實施步驟 (修正版)

### 6.1 階段 0-2: 基礎復制 (與之前相同)

**(參考之前版本,無變化)**

### 6.2 階段 3: 適配 Factor 模塊 (1-2 天)

**(與之前版本相同,無變化)**

### 6.3 階段 4: 復制 Model 模塊 (1-2 天) 🔥 新增

```bash
# 1. 復制 models/_comm (100% + 微調)
cp -r ref/hf-stock-live-demo-main/models/_comm \
      hf-live/models/_comm

# 2. 微調 model_base.h
vim hf-live/models/_comm/model_base.h
# 刪除 #include "sdp_handler/core/sdp_handler.h"
# 替換 start_time_t → GodzillaTime

# 3. 復制 ModelCalculationEngine
cp ref/hf-stock-live-demo-main/app_live/engine/model_calculation_engine.* \
   hf-live/app_live/engine/

# 4. 適配初始化邏輯
vim hf-live/app_live/engine/model_calculation_engine.h
# void Init(..., SDPHandler*) → void Init(...) + void SetSendCallback(...)

# 5. 復制 ModelCalculationThread
cp ref/hf-stock-live-demo-main/app_live/thread/model_calculation_thread.h \
   hf-live/app_live/thread/

# 6. 復制 ModelResultScanThread
cp ref/hf-stock-live-demo-main/app_live/thread/model_result_scan_thread.h \
   hf-live/app_live/thread/

# 7. 適配發送接口
vim hf-live/app_live/thread/model_result_scan_thread.h
# SDPHandler* → std::function<> send_callback_

# 8. 復制示例模型
cp -r ref/hf-stock-live-demo-main/models/demo0000 \
      hf-live/models/demo

# 9. 復制 CMakeLists
cp ref/hf-stock-live-demo-main/models/CMakeLists.txt \
   hf-live/models/

# 10. 復制模型模板
cp ref/hf-stock-live-demo-main/models/ModelModuleTemplate.cmake \
   hf-live/models/
```

### 6.4 階段 5: 集成 Factor + Model (0.5-1 天)

```cpp
// hf-live/app_live/entry.cpp
extern "C" {
    void* signal_create(const char* config_json) {
        // 1. 創建 FactorCalculationEngine
        auto* factor_engine = new FactorCalculationEngine(config);

        // 2. 創建 ModelCalculationEngine 🔥
        auto* model_engine = new ModelCalculationEngine();
        model_engine->Init(date, thread_num, config);

        // 3. 設置模型發送回調 🔥
        model_engine->SetSendCallback([](const char* symbol, int64_t ts, const std::vector<float>& preds) {
            SignalSender::Send(symbol, ts, preds.data(), preds.size());  // 發送預測值
        });

        // 4. 設置 FactorResultScanThread 發送到模型 🔥
        factor_engine->SetModelEngine(model_engine);  // 連接兩個 engine

        // 5. 啟動
        factor_engine->Start();
        model_engine->Start();

        // 6. 返回 handle (包含兩個 engine)
        auto* handle = new SignalHandle{factor_engine, model_engine};
        return handle;
    }
}
```

### 6.5 階段 6: 配置與構建 (0.5-1 天)

**(與之前版本類似,需要添加 ONNX Runtime 依賴)**

---

## 七、復用統計 (修正版)

### 7.1 完整統計

| 模塊 | 文件數 | 復用率 | 說明 |
|------|--------|--------|------|
| **A. 直接復制** | ~45 | 100% | utils, timer, queues, model_comm |
| **B. 微調復用** | ~25 | 85-95% | config, factor_base, model_base |
| **C. 適配復用** | ~20 | 50-80% | engines, threads |
| **D. 重寫** | ~10 | 20-40% | entry.cpp, signal_sender |
| **E. 刪除** | ~15 | 0% | HDF5, sdp 平台相關 |
| **總計** | ~115 | **平均 72%** 🔥 | (之前: 65%) |

### 7.2 關鍵模塊復用率

| 模塊 | 之前預估 | 實際需求 |
|------|---------|---------|
| factors/ | 90% ✅ | 90% ✅ |
| **models/** | ❌ 0% (刪除) | ✅ **75%** 🔥🔥🔥 |
| factor_engine | 70% ✅ | 70% ✅ |
| **model_engine** | ❌ 0% (刪除) | ✅ **80%** 🔥 |
| factor_threads | 75% ✅ | 75% ✅ |
| **model_threads** | ❌ 0% (刪除) | ✅ **80%** 🔥 |

### 7.3 時間估算 (修正版)

| 階段 | 之前估算 | 修正估算 | 差異 |
|------|---------|---------|------|
| 階段 0-2: 基礎復制 | 5-8 小時 | 5-8 小時 | 無變化 |
| 階段 3: 適配 Factor | 1-2 天 | 1-2 天 | 無變化 |
| 階段 4: 復制 Model | ❌ 0 | ✅ **1-2 天** 🔥 | +1-2 天 |
| 階段 5: 集成 | 0.5-1 天 | 0.5-1 天 | 無變化 |
| 階段 6: 配置構建 | 0.5-1 天 | 1-1.5 天 | +0.5 天 (ONNX) |
| **總計** | **3-5 天** | **4-7 天** 🔥 | +1-2 天 |

---

## 八、驗證清單 (修正版)

### 8.1 編譯驗證

- [ ] `make clean && make` 成功編譯
- [ ] 生成 `build/libsignal.so`
- [ ] `ldd build/libsignal.so` 包含 ONNX Runtime 依賴 🔥
- [ ] 文件大小合理 (< 10MB,含 ONNX)

### 8.2 代碼驗證

- [ ] 所有數據類型已替換 (Stock_* → Depth/Trade)
- [ ] 所有 sdp 平台相關代碼已刪除
- [ ] 因子模板可正常實例化
- [ ] **模型接口正確** 🔥
- [ ] **ModelCalculationEngine 可接收因子數據** 🔥
- [ ] SignalSender 接口正確

### 8.3 功能驗證

- [ ] FactorCalculationEngine 可接收 Depth 數據
- [ ] 多線程計算邏輯正常
- [ ] FactorResultScanThread 可收集結果
- [ ] **FactorResultScanThread 可發送到 ModelEngine** 🔥
- [ ] **ModelCalculationEngine 可接收因子數據** 🔥
- [ ] **ModelCalculationThread 可執行 ONNX 預測** 🔥
- [ ] **ModelResultScanThread 可收集預測值** 🔥
- [ ] SignalSender::Send() 可觸發回調

---

## 九、關鍵認知修正

### 9.1 業務邏輯

| 項目 | 之前理解 | 正確理解 |
|------|---------|---------|
| 核心流程 | Factor → 策略 | Factor → **Model** → 策略 🔥 |
| on_factor 接收 | 因子值 | 因子值 OR **預測值** 🔥 |
| Model 地位 | 可選組件 | **核心組件** 🔥 |
| 策略大師認知 | 知道收到的是因子 | **不知道**收到的是什麼 🔥 |

### 9.2 代碼復用

| 模塊 | 之前計劃 | 正確計劃 |
|------|---------|---------|
| models/_comm/ | ❌ 刪除 | ✅ 90% 復用 🔥 |
| model_calculation_engine | ❌ 刪除 | ✅ 80% 復用 🔥 |
| model_*_thread | ❌ 刪除 | ✅ 80% 復用 🔥 |
| ONNX Runtime | ❌ 不需要 | ✅ **必須依賴** 🔥 |

### 9.3 架構設計

**正確的 hf-live 架構**:
```
hf-live/
├── factors/                          # 因子計算
│   ├── _comm/                        # 因子框架 (90% 復用)
│   └── demo/                         # 示例因子
│
├── models/                           # 🔥 模型預測 (75% 復用)
│   ├── _comm/                        # 🔥 模型框架
│   │   ├── model_base.h              # 🔥 ModelInterface
│   │   ├── model_registry.h          # 🔥 模型註冊器
│   │   └── ...
│   └── demo/                         # 🔥 ONNX 示例模型
│
├── app_live/
│   ├── engine/
│   │   ├── factor_calculation_engine.*  # 因子引擎
│   │   └── model_calculation_engine.*   # 🔥 模型引擎
│   │
│   └── thread/
│       ├── factor_calculation_thread.h
│       ├── factor_result_scan_thread.h
│       ├── model_calculation_thread.h   # 🔥 模型計算線程
│       └── model_result_scan_thread.h   # 🔥 模型結果線程
│
└── handler/                          # 數據處理
```

---

## 十、總結

### 核心修正

1. ✅ **Model 是核心組件**,不是可選組件
2. ✅ **常規流程**: Factor → Model → 策略 (不是 Factor → 策略)
3. ✅ **models/ 必須 75%+ 復用**,不能刪除
4. ✅ **4 個 thread 全部需要**: factor_calc, factor_scan, model_calc, model_scan
5. ✅ **2 個 engine 全部需要**: FactorCalculationEngine, ModelCalculationEngine

### 時間影響

- **之前估算**: 3-5 天 (誤刪 Model)
- **修正估算**: 4-7 天 (包含 Model)
- **差異**: +1-2 天 (Model 模塊復制與適配)

### 復用率

- **之前**: 65% (誤刪 Model)
- **修正**: 72% (包含 Model)
- **提升**: +7% (Model 模塊貢獻)

---

**版本**: v2.0 (完整版)
**日期**: 2025-12-03
**狀態**: 已修正 Model 相關認知錯誤
