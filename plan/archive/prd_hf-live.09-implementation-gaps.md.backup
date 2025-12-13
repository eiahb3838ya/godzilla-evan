# PRD 09: hf-live Implementation Gaps - 錯誤分析與修復計劃

**文檔版本**: v1.0
**創建時間**: 2025-12-07
**狀態**: 🔴 CRITICAL - 需要立即修復
**預計修復時間**: 2.5-4.5 天

---

## 📋 執行摘要

在 Phase 4 編譯成功後 (libsignal.so 33KB)，發現 **hf-live 實現僅完成 ref 代碼的 57%**，存在嚴重功能缺失。

### 核心問題

| 模塊 | 完成度 | 缺失行數 | 嚴重性 |
|------|--------|---------|--------|
| FactorCalculationEngine | 39% | 243/400 行 | **P0 - 數據流中斷** |
| ModelCalculationEngine | 45% | 98/180 行 | **P0 - 模型無法運行** |
| 依賴文件 (app_live/common) | 12.5% | 7/8 缺失 | **P1 - 編譯依賴** |

**影響**: 系統可編譯但 **完全無法運行** - 數據流在初始化階段即中斷。

---

## 🔍 問題發現過程

### 用戶驗證請求 (5 項檢查)

1. ✅ **編譯獨立性**: hf-live 使用 CMake 獨立編譯，不依賴 ref 路徑
2. ✅ **Godzilla 隔離**: Godzilla 僅使用 libsignal.so，不接觸源碼
3. ❌ **數據流完整性**: Factor → Model → on_factor **流程中斷**
4. ❌ **PRD 完整性**: 違反 "完整複製 ref 代碼" 要求
5. ❌ **代碼複製率**: 僅 57% 代碼從 ref 複製，大量 TODO 未實現

### 關鍵發現

檢查 `hf-live/app_live/engine/factor_calculation_engine.cpp:11-26`:

```cpp
void FactorCalculationEngine::Init(int thread_num, ModelCalculationEngine* model_calc_engine) {
    // TODO: adapt to Godzilla - 簡化版初始化
    // 需要從外部傳入:
    // 1. asset_codes_ (資產列表)
    // 2. factor 配置信息
    // 目前先留空,等待後續集成時填充

    calc_thread_num_ = std::max(1, thread_num);

    // TODO: adapt to Godzilla - 需要實現
    // AssignWorkLoads(thread_num);
    // AssignThreadMapping();
    // 創建緩衝區和隊列
    // 創建計算線程
    // 創建掃描線程
}
```

**對比 ref 版本** (`ref/hf-stock-live-demo-main/app_live/engine/factor_calculation_engine.cpp`):

```cpp
void FactorCalculationEngine::Init(...) {
    trade_date_ = date;
    asset_codes_ = codes;
    InitConfig(config);  // 39 行配置解析

    // Factor static initialization
    factors::comm::factor_manager::FactorEntryManager::StaticInit(factor_group_num_, ...);

    AssignWorkLoads(thread_num);  // 43 行負載分配
    AssignThreadMapping();  // 45 行線程映射

    // 創建 SPMC 緩衝區
    for (int i = 0; i < asset_group_num_; i++) {
        data_buffers_.emplace_back(std::make_shared<SPMCBuffer<TickDataInfo>>(5000));
    }

    // 創建 SPSC 隊列
    for (int i = 0; i < calc_thread_num_; i++) {
        result_queues_.emplace_back(std::make_shared<SPSCQueue<FactorResultInfo>>(500000));
    }

    // 創建計算線程
    for (int i = 0; i < calc_thread_num_; i++) {
        calc_threads_.emplace_back(std::make_unique<FactorCalculationThread>(...));
    }

    // 創建掃描線程
    scan_thread_ = std::make_unique<FactorResultScanThread>(...);
}
```

**缺失內容**:
- `InitConfig()` 方法完全缺失 (39 行)
- 無 `data_buffers_` 創建 (7 行)
- 無 `result_queues_` 創建 (4 行)
- 無 `calc_threads_` 創建 (17 行)
- 無 `scan_thread_` 創建 (14 行)

---

## 📊 詳細差距分析

### 1. FactorCalculationEngine (P0 - 關鍵)

| 方法 | hf-live | ref | 缺失 | 狀態 |
|------|---------|-----|------|------|
| `Init()` | 15 行 (TODO) | 82 行 | 67 行 (82%) | ❌ 未實現 |
| `InitConfig()` | **不存在** | 39 行 | 39 行 (100%) | ❌ 完全缺失 |
| `OnDepth()` | 26 行 | 26 行 | 0 行 | ✅ 完整 |
| `OnTrade()` | 21 行 | 21 行 | 0 行 | ✅ 完整 |
| `AssignWorkLoads()` | 17 行 | 43 行 | 26 行 (60%) | ⚠️ 簡化版 |
| `AssignThreadMapping()` | 32 行 | 45 行 | 13 行 (29%) | ⚠️ 簡化版 |
| `Start()` | 10 行 | 12 行 | 2 行 | ✅ 基本完整 |
| `Stop()` | 18 行 | 56 行 | 38 行 (68%) | ⚠️ 缺統計 |

**總計**: 157/400 行 (39% 完成度)

#### 關鍵缺失邏輯

**1.1 InitConfig() - 完全缺失**
```cpp
// ref 版本 (39 行)
void FactorCalculationEngine::InitConfig(const config::ConfigData& config) {
    auto factor_conf = config.factor_calc_engine_conf;
    factor_group_num_ = factor_conf.factor_groups.size();

    factor_group_names_.clear();
    for (auto& fg : factor_conf.factor_groups) {
        factor_group_names_.push_back(fg.name);

        for (auto& fn : fg.factors) {
            all_factor_names_.push_back(fn.name);
            all_factor_windows_.push_back(fn.window);
            all_factor_step_lens_.push_back(fn.step_len);
        }
    }
}
```

**hf-live**: ❌ 方法不存在，無配置解析邏輯

**1.2 Init() - 緩衝區/隊列/線程創建缺失**

| 組件 | ref 實現 | hf-live 狀態 | 影響 |
|------|----------|-------------|------|
| `data_buffers_` | 7 行，為每個 asset_group 創建 SPMC buffer | ❌ 未創建 | OnDepth/OnTrade 無法推送數據 |
| `result_queues_` | 4 行，為每個線程創建 SPSC queue | ❌ 未創建 | 計算結果無法傳遞 |
| `calc_threads_` | 17 行，創建 FactorCalculationThread | ❌ 未創建 | 無計算線程 |
| `scan_thread_` | 14 行，創建 FactorResultScanThread | ❌ 未創建 | 無結果掃描線程 |

**數據流中斷點**:
```cpp
// hf-live/app_live/engine/factor_calculation_engine.cpp:71-81
void FactorCalculationEngine::OnDepth(const hf::Depth* depth) {
    // ...
    if (grp_idx > -1 && grp_idx < static_cast<int>(data_buffers_.size())) {
        // ❌ data_buffers_ 是空的! size() == 0
        // 數據永遠不會被推送
        data_buffers_[grp_idx]->push(qdi);  // 永遠不執行
    }
}
```

---

### 2. ModelCalculationEngine (P0 - 關鍵)

| 方法 | hf-live | ref | 缺失 | 狀態 |
|------|---------|-----|------|------|
| `Init()` | 8 行 (TODO) | 42 行 | 34 行 (81%) | ❌ 未實現 |
| `InitConfig()` | **不存在** | 56 行 | 56 行 (100%) | ❌ 完全缺失 |
| `Start()` | 6 行 | 8 行 | 2 行 | ✅ 基本完整 |
| `Stop()` | 16 行 | 34 行 | 18 行 (53%) | ⚠️ 缺統計 |
| `SendFactors()` | 4 行 | 6 行 | 2 行 | ✅ 基本完整 |

**總計**: 82/180 行 (45% 完成度)

#### 關鍵缺失邏輯

**2.1 InitConfig() - 完全缺失**
```cpp
// ref 版本 (56 行)
void ModelCalculationEngine::InitConfig(const config::ConfigData& config) {
    auto model_conf = config.model_calc_engine_conf;
    model_num_ = model_conf.models.size();
    trading_date_ = config.trading_date;

    model_names_.clear();
    for (auto& m : model_conf.models) {
        model_names_.push_back(m.name);
        model_output_names_.push_back(m.output_names);

        // 註冊到 model_column_names_
        for (auto& on : m.output_names) {
            model_column_names_.push_back(m.name + "_" + on);
        }
    }
}
```

**hf-live**: ❌ 方法不存在，無模型配置解析

**2.2 Init() - 線程創建缺失**

```cpp
// ref 版本 (42 行)
void ModelCalculationEngine::Init(int thread_num) {
    InitConfig(config);

    // 創建 SPMC 緩衝區
    factor_data_buffer_ = std::make_shared<SPMCBuffer<input_t>>(20000);

    // 創建模型計算線程
    for (size_t i = 0; i < model_num_; i++) {
        model_calc_threads_.emplace_back(
            std::make_unique<ModelCalculationThread>(
                i, model_names_[i], model_output_names_[i],
                factor_data_buffer_, ...
            )
        );
    }

    // 創建結果掃描線程
    model_result_scan_thread_ = std::make_unique<ModelResultScanThread>(...);
}
```

**hf-live 版本** (8 行):
```cpp
void ModelCalculationEngine::Init(int thread_num) {
    // TODO: adapt to Godzilla - 簡化版初始化
    // 需要從外部傳入模型列表
    // 目前先留空
}
```

**影響**: 無線程創建，模型無法運行

---

### 3. 依賴文件缺失 (P1 - 高優先級)

**app_live/common/ 目錄狀態**:

| 文件 | hf-live | ref | 狀態 |
|------|---------|-----|------|
| `timer_utils.h` | ✅ | ✅ | 完整 |
| `print.hpp` | ❌ | ✅ | **缺失** - WLOG/TO_STRING 宏 |
| `tools.h` | ❌ | ✅ | **缺失** - CreateDirRecursive |
| `tools.cpp` | ❌ | ✅ | **缺失** |
| `hdf5_utils.h` | ❌ | ✅ | **可選** - 僅用於結果保存 |
| `hdf5_utils.cpp` | ❌ | ✅ | **可選** |
| `config_parser.h` | ❌ | ✅ | **缺失** - 需適配 |
| `json_parser.h` | ❌ | ✅ | **缺失** - 需適配 |

**完成度**: 1/8 (12.5%)

#### 3.1 print.hpp - 日誌宏缺失

**影響**: 所有使用 `WLOG`, `TO_STRING` 的代碼無法編譯

**ref 版本功能**:
```cpp
// ref/hf-stock-live-demo-main/app_live/common/print.hpp
#define WLOG(msg) std::cerr << "[" << __FUNCTION__ << "] " << msg << std::endl
#define TO_STRING(val) std::to_string(val)
```

**依賴位置**:
- `factor_calculation_thread.cpp` - 日誌輸出
- `model_calculation_thread.cpp` - 日誌輸出
- `factor_result_scan_thread.cpp` - 性能統計

#### 3.2 tools.h/cpp - 工具函數缺失

**影響**: 無法創建輸出目錄

**ref 版本功能**:
```cpp
namespace tools {
    void CreateDirRecursive(const std::string& path);
}
```

**依賴位置**:
- `model_result_scan_thread.cpp` - 創建結果目錄

#### 3.3 config_parser.h - 配置解析缺失

**影響**: InitConfig() 無法實現

**ref 版本結構**:
```cpp
namespace config {
    struct FactorGroupConfig {
        std::string name;
        std::vector<FactorConfig> factors;
    };

    struct FactorCalcEngineConfig {
        std::vector<FactorGroupConfig> factor_groups;
    };

    struct ModelCalcEngineConfig {
        std::vector<ModelConfig> models;
    };

    struct ConfigData {
        std::string trading_date;
        FactorCalcEngineConfig factor_calc_engine_conf;
        ModelCalcEngineConfig model_calc_engine_conf;
    };
}
```

**Godzilla 適配方案**: 創建簡化版 `GodzillaConfig` 替代 `ConfigData`

---

## 🎯 根本原因分析

### 過度簡化策略失敗

**原始 PRD 要求**:
> "完整複製 ref/hf-stock-live-demo-main 代碼，僅做最小化適配"

**實際執行**:
- ❌ 大量方法標記為 TODO 而非複製實現
- ❌ 核心初始化邏輯被註釋掉
- ❌ 依賴文件未完整複製
- ❌ 配置系統被簡化為空實現

### 錯誤假設

1. **假設**: "配置可以稍後從外部傳入"
   **實際**: 初始化必須完整，否則數據流中斷

2. **假設**: "緩衝區和線程可以延後創建"
   **實際**: 無緩衝區/線程 = 系統無法運行

3. **假設**: "簡化版可以先編譯通過"
   **實際**: 編譯通過 ≠ 功能可用

---

## 🔧 修復計劃

### Priority 0: 恢復數據流 (1-2 天)

**目標**: 使 OnDepth → Factor → Model → on_factor 流程可運行

#### Task 0.1: FactorCalculationEngine::Init() 完整實現
- **文件**: `hf-live/app_live/engine/factor_calculation_engine.cpp`
- **方法**: 從 ref 複製 Init() 完整邏輯 (82 行)
- **關鍵步驟**:
  ```cpp
  void FactorCalculationEngine::Init(int thread_num, ModelCalculationEngine* model_calc_engine) {
      // 1. 設置基本參數
      trade_date_ = "20250101";  // TODO: 從外部傳入
      asset_codes_ = {"BTCUSDT", "ETHUSDT"};  // TODO: 從外部傳入

      // 2. 配置解析 (適配版)
      InitConfig(godzilla_config);

      // 3. Factor 靜態初始化
      factors::comm::factor_manager::FactorEntryManager::StaticInit(...);

      // 4. 負載分配
      AssignWorkLoads(thread_num);
      AssignThreadMapping();

      // 5. 創建 SPMC 緩衝區
      for (int i = 0; i < asset_group_num_; i++) {
          data_buffers_.emplace_back(std::make_shared<SPMCBuffer<TickDataInfo>>(5000));
      }

      // 6. 創建 SPSC 隊列
      for (int i = 0; i < calc_thread_num_; i++) {
          result_queues_.emplace_back(std::make_shared<SPSCQueue<FactorResultInfo>>(500000));
      }

      // 7. 創建計算線程
      for (int i = 0; i < calc_thread_num_; i++) {
          calc_threads_.emplace_back(std::make_unique<FactorCalculationThread>(...));
      }

      // 8. 創建掃描線程
      scan_thread_ = std::make_unique<FactorResultScanThread>(...);
  }
  ```

#### Task 0.2: InitConfig() 實現 (適配版)
- **文件**: `hf-live/app_live/engine/factor_calculation_engine.cpp`
- **方法**: 新增 InitConfig() 方法
- **適配策略**: 使用 GodzillaConfig 替代 config::ConfigData
- **代碼**:
  ```cpp
  void FactorCalculationEngine::InitConfig(const GodzillaConfig& config) {
      factor_group_num_ = config.factor_groups.size();

      factor_group_names_.clear();
      all_factor_names_.clear();
      all_factor_windows_.clear();
      all_factor_step_lens_.clear();

      for (auto& fg : config.factor_groups) {
          factor_group_names_.push_back(fg.name);

          for (auto& fn : fg.factors) {
              all_factor_names_.push_back(fn.name);
              all_factor_windows_.push_back(fn.window);
              all_factor_step_lens_.push_back(fn.step_len);
          }
      }
  }
  ```

#### Task 0.3: ModelCalculationEngine::Init() 完整實現
- **文件**: `hf-live/app_live/engine/model_calculation_engine.cpp`
- **方法**: 從 ref 複製 Init() 完整邏輯 (42 行)
- **關鍵步驟**:
  ```cpp
  void ModelCalculationEngine::Init(int thread_num) {
      // 1. 配置解析
      InitConfig(godzilla_config);

      // 2. 創建 SPMC 緩衝區
      factor_data_buffer_ = std::make_shared<SPMCBuffer<input_t>>(20000);

      // 3. 創建模型計算線程
      for (size_t i = 0; i < model_num_; i++) {
          model_calc_threads_.emplace_back(
              std::make_unique<ModelCalculationThread>(...)
          );
      }

      // 4. 創建結果掃描線程
      model_result_scan_thread_ = std::make_unique<ModelResultScanThread>(...);
  }
  ```

#### Task 0.4: ModelCalculationEngine::InitConfig() 實現
- **文件**: `hf-live/app_live/engine/model_calculation_engine.cpp`
- **方法**: 新增 InitConfig() 方法 (56 行適配版)
- **代碼**:
  ```cpp
  void ModelCalculationEngine::InitConfig(const GodzillaConfig& config) {
      model_num_ = config.models.size();
      trading_date_ = config.trading_date;

      model_names_.clear();
      model_output_names_.clear();
      model_column_names_.clear();

      for (auto& m : config.models) {
          model_names_.push_back(m.name);
          model_output_names_.push_back(m.output_names);

          for (auto& on : m.output_names) {
              model_column_names_.push_back(m.name + "_" + on);
          }
      }
  }
  ```

**預計時間**: 1-2 天
**驗證方式**: 編譯通過 + 單元測試數據流

---

### Priority 1: 補充依賴文件 (0.5-1 天)

#### Task 1.1: print.hpp 實現
- **文件**: `hf-live/app_live/common/print.hpp`
- **代碼**:
  ```cpp
  #pragma once
  #include <iostream>
  #include <string>

  #define WLOG(msg) std::cerr << "[" << __FUNCTION__ << "] " << msg << std::endl
  #define TO_STRING(val) std::to_string(val)
  ```

#### Task 1.2: tools.h/cpp 實現
- **文件**: `hf-live/app_live/common/tools.h`, `tools.cpp`
- **代碼**:
  ```cpp
  // tools.h
  #pragma once
  #include <string>

  namespace tools {
      void CreateDirRecursive(const std::string& path);
  }

  // tools.cpp
  #include "tools.h"
  #include <sys/stat.h>
  #include <cstring>

  void tools::CreateDirRecursive(const std::string& path) {
      size_t pos = 0;
      while ((pos = path.find('/', pos + 1)) != std::string::npos) {
          std::string sub = path.substr(0, pos);
          mkdir(sub.c_str(), 0755);
      }
      mkdir(path.c_str(), 0755);
  }
  ```

#### Task 1.3: config_parser.h 實現 (Godzilla 適配版)
- **文件**: `hf-live/app_live/common/config_parser.h`
- **代碼**:
  ```cpp
  #pragma once
  #include <string>
  #include <vector>

  struct FactorConfig {
      std::string name;
      int window;
      int step_len;
  };

  struct FactorGroupConfig {
      std::string name;
      std::vector<FactorConfig> factors;
  };

  struct ModelConfig {
      std::string name;
      std::vector<std::string> output_names;
  };

  struct GodzillaConfig {
      std::string trading_date;
      std::vector<std::string> symbols;
      std::vector<FactorGroupConfig> factor_groups;
      std::vector<ModelConfig> models;
  };
  ```

#### Task 1.4: json_parser.h 實現 (簡化版)
- **文件**: `hf-live/app_live/common/json_parser.h`
- **功能**: 解析 JSON 配置到 GodzillaConfig
- **依賴**: 使用 nlohmann/json 或手寫簡單解析器

**預計時間**: 0.5-1 天
**驗證方式**: 編譯通過 + 配置解析單元測試

---

### Priority 2: 完善 AssignWorkLoads/AssignThreadMapping (0.5 天)

#### Task 2.1: AssignWorkLoads() 完整實現
- **文件**: `hf-live/app_live/engine/factor_calculation_engine.cpp`
- **從 ref 複製**: 完整的 43 行邏輯

#### Task 2.2: AssignThreadMapping() 完整實現
- **文件**: `hf-live/app_live/engine/factor_calculation_engine.cpp`
- **從 ref 複製**: 完整的 45 行邏輯

**預計時間**: 0.5 天
**驗證方式**: 檢查 code_info_ 和 codes_in_asset_group_ 正確性

---

### Priority 3: 可選功能 (0.5 天)

#### Task 3.1: 刪除 HDF5 依賴
- **影響文件**:
  - `model_result_scan_thread.cpp` - 移除 SaveResultsToH5()
  - `factor_result_scan_thread.cpp` - 移除 SaveResultsToH5()
- **原因**: Godzilla 不需要保存歷史結果到 HDF5

#### Task 3.2: 簡化性能統計
- **文件**: `factor_calculation_engine.cpp`, `model_calculation_engine.cpp`
- **方法**: Stop() 方法中移除詳細統計輸出

**預計時間**: 0.5 天
**驗證方式**: 編譯通過

---

## ⏱️ 總時間估算

| Priority | 任務 | 時間估算 |
|----------|------|---------|
| **P0** | 恢復數據流 (Init + InitConfig) | 1-2 天 |
| **P1** | 補充依賴文件 (print, tools, config) | 0.5-1 天 |
| **P2** | 完善負載分配邏輯 | 0.5 天 |
| **P3** | 可選功能清理 | 0.5 天 |

**總計**: **2.5-4.5 天**

---

## 🚨 風險評估

| 風險 | 可能性 | 影響 | 緩解措施 |
|------|--------|------|---------|
| ConfigData → GodzillaConfig 轉換失敗 | 中 | 高 | 先實現最小可用配置結構 |
| FactorEntryManager 適配問題 | 低 | 中 | ref 代碼可直接複製 |
| ONNX 模型加載失敗 | 中 | 高 | 使用 dummy 模型測試框架 |
| 線程競爭條件 | 低 | 高 | ref 代碼已驗證，直接複製 |

---

## ✅ 驗證清單

### 階段 1: 編譯驗證
- [ ] `cmake ..` 成功
- [ ] `make` 成功
- [ ] `libsignal.so` 大小 > 100KB (當前 33KB 過小)

### 階段 2: 單元測試
- [ ] FactorCalculationEngine::Init() 創建所有緩衝區
- [ ] ModelCalculationEngine::Init() 創建所有線程
- [ ] OnDepth() 數據成功推送到 data_buffers_
- [ ] FactorCalculationThread 能夠計算因子
- [ ] ModelCalculationThread 能夠運行推理

### 階段 3: 端到端測試
- [ ] Godzilla 加載 libsignal.so 成功
- [ ] 市場數據觸發 OnDepth/OnTrade
- [ ] 因子計算完成
- [ ] 模型推理完成
- [ ] Python strategy 的 on_factor() 接收到預測值

---

## 📈 預防措施

### 未來開發規範

1. **代碼複製原則**:
   - ✅ 優先完整複製 ref 代碼
   - ✅ 適配僅限於命名空間/類型轉換
   - ❌ 禁止將完整實現替換為 TODO

2. **驗證流程**:
   - ✅ 每個模塊完成後立即驗證功能
   - ✅ 使用 diff 工具對比 ref 和 hf-live
   - ✅ 編譯成功 ≠ 功能完成

3. **文檔同步**:
   - ✅ PRD 文檔必須包含完成度檢查
   - ✅ 每個 Phase 結束時執行完整性審計

### 自動化檢查腳本

```bash
#!/bin/bash
# verify_completeness.sh

echo "=== hf-live Implementation Completeness Check ==="

# 1. 檢查 TODO 數量
todo_count=$(grep -r "TODO: adapt to Godzilla" hf-live/ | wc -l)
echo "Remaining TODOs: $todo_count (目標: 0)"

# 2. 檢查關鍵方法存在性
check_method() {
    file=$1
    method=$2
    if grep -q "^void.*::$method" "$file"; then
        echo "✅ $method found in $file"
    else
        echo "❌ $method MISSING in $file"
    fi
}

check_method "hf-live/app_live/engine/factor_calculation_engine.cpp" "InitConfig"
check_method "hf-live/app_live/engine/model_calculation_engine.cpp" "InitConfig"

# 3. 檢查緩衝區創建
if grep -q "data_buffers_.emplace_back" hf-live/app_live/engine/factor_calculation_engine.cpp; then
    echo "✅ data_buffers_ creation found"
else
    echo "❌ data_buffers_ creation MISSING"
fi

# 4. 檢查線程創建
if grep -q "calc_threads_.emplace_back" hf-live/app_live/engine/factor_calculation_engine.cpp; then
    echo "✅ calc_threads_ creation found"
else
    echo "❌ calc_threads_ creation MISSING"
fi

echo "=== Check Complete ==="
```

---

## 📝 總結

**問題嚴重性**: 🔴 **P0 - 系統無法運行**

**核心原因**: 過度簡化導致數據流中斷

**修復策略**: 完整複製 ref 代碼，最小化適配

**預計時間**: 2.5-4.5 天

**成功標準**: OnDepth → Factor → Model → on_factor 完整數據流可運行

---

**文檔結束**
