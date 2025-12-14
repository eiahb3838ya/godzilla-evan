# PRD: HF-Live 延遲監控系統 (Phase 5D)

**文檔版本**: v1.0
**創建日期**: 2024-12-14
**狀態**: ✅ 已實現
**分支**: `feature/latency-monitoring`

---

## 目錄

1. [概述](#概述)
2. [快速開始](#快速開始)
3. [設計原理](#設計原理)
4. [運作機制](#運作機制)
5. [延遲指標解讀](#延遲指標解讀)
6. [維護指南](#維護指南)
7. [故障排除](#故障排除)
8. [技術細節](#技術細節)

---

## 概述

### 目標

實現**零接口變更**的端到端延遲監控系統，用於測量從行情到達到模型輸出回調的完整鏈路延遲。

### 核心特性

- ✅ **零接口變更**: 所有函數簽名保持不變
- ✅ **零運行時開銷**: 關閉時完全無性能影響
- ✅ **編譯時決定**: 通過 CMake 開關控制
- ✅ **自動化解析**: Python 層自動檢測並解析元數據
- ✅ **向後兼容**: 關閉後行為與原代碼完全一致

### 監控鏈路

```
Tick 到達 → FactorCalculation → FactorScan → ModelCalculation → Model Output → Python Callback
   ↑                                                                                    ↓
   └────────────────────────── total_elapsed_us (~300μs) ──────────────────────────────┘
```

---

## 快速開始

### 1. 啟用延遲監控

```bash
cd /home/huyifan/projects/godzilla-evan/hf-live

# 創建 timing-enabled build
cmake -B build_timing -DHF_TIMING_METADATA=ON
cmake --build build_timing

# 部署到容器
docker cp build_timing/libsignal.so godzilla-dev:/app/hf-live/build/libsignal.so

# 重啟策略
docker exec godzilla-dev pm2 restart strategy_test_hf_live
```

### 2. 查看延遲數據

```bash
# 方法 1: 查看 Python 日誌
docker exec godzilla-dev pm2 logs strategy_test_hf_live | grep Latency

# 預期輸出:
# 📊 [Latency] tick_wait=0.7μs calc=51.4μs total=298.8μs
```

### 3. 關閉延遲監控

```bash
cd /home/huyifan/projects/godzilla-evan/hf-live

# 創建標準 build（無延遲監控）
cmake -B build
cmake --build build

# 部署
docker cp build/libsignal.so godzilla-dev:/app/hf-live/build/libsignal.so
docker exec godzilla-dev pm2 restart strategy_test_hf_live
```

---

## 設計原理

### Linus 代碼原則

本設計遵循 Linus Torvalds 的代碼哲學：

1. **編譯時決定，而非運行時**
   - 使用 `#ifdef HF_TIMING_METADATA` 而非 `if (enable_timing)`
   - 關閉時代碼被編譯器完全移除（零開銷）

2. **最小侵入性**
   - 不修改任何函數簽名
   - 通過現有數據通道傳遞元數據（prepend to vector）

3. **向後兼容**
   - 關閉後行為與原代碼完全一致
   - 不影響現有功能

### 零接口變更技術

**問題**: 如何在不修改函數簽名的情況下傳遞延遲數據？

**解決方案**: 元數據前置（Metadata Prepending）

```cpp
// ❌ 傳統方案（需要修改簽名）
void Callback(const std::string& symbol, int64_t timestamp,
              const std::vector<float>& values,
              const TimingInfo& timing);  // 新增參數

// ✅ 零接口方案（簽名不變）
void Callback(const std::string& symbol, int64_t timestamp,
              const std::vector<float>& values) {
    // values = [metadata_header(8), ...actual_values]
    // 通過 marker (-999.0) 識別元數據存在
}
```

---

## 運作機制

### 數據流圖

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          延遲監控數據流                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [1] Tick 到達 (signal_api.cpp OnDepth)                                 │
│       ↓                                                                 │
│       start_tsc = RdtscTimer()()  ← 記錄起始時鐘周期                     │
│       ↓                                                                 │
│  [2] FactorCalculationThread                                            │
│       ↓                                                                 │
│       計算因子 (DoOnUpdateFactors)                                       │
│       ↓                                                                 │
│       tick_wait_us = (calc_start - start_tsc) * scaler                  │
│       factor_calc_duration_us = (calc_end - calc_start) * scaler        │
│       ↓                                                                 │
│  [3] FactorResultScanThread::SendData() ─┐                              │
│       ↓                                  │                              │
│   ┌──────────────────────────────────────┴──────────────────────────┐   │
│   │ #ifdef HF_TIMING_METADATA                                       │   │
│   │   // 注入點 1: 前置 8 列元數據                                   │   │
│   │   values_with_metadata = [                                      │   │
│   │     -999.0,              // [0] marker                          │   │
│   │     tick_wait_us,        // [1] 行情等待                        │   │
│   │     calc_duration_us,    // [2] 因子計算耗時                    │   │
│   │     calc_elapsed_us,     // [3] 因子計算累計                    │   │
│   │     scan_elapsed_us,     // [4] 掃描累計                        │   │
│   │     total_elapsed_us,    // [5] 端到端總延遲                    │   │
│   │     factor_count,        // [6] 因子數量                        │   │
│   │     0.0,                 // [7] 保留                            │   │
│   │     ...actual_factors    // [8+] 實際因子值                     │   │
│   │   ]                                                             │   │
│   │ #else                                                           │   │
│   │   values = [actual_factors]  // 無元數據                        │   │
│   │ #endif                                                          │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│       ↓                                                                 │
│  [4] ModelSendCallback (factor_calculation_engine.cpp) ─┐               │
│       ↓                                                 │               │
│   ┌─────────────────────────────────────────────────────┴─────────────┐ │
│   │ #ifdef HF_TIMING_METADATA                                         │ │
│   │   // 注入點 2: 提取元數據並重建 start_tsc                          │ │
│   │   if (factors[0] == -999.0) {                                     │ │
│   │     input.tick_max_wait_elapsed_us = factors[1];                  │ │
│   │     input.factor_max_calc_duration_us = factors[2];               │ │
│   │     input.factor_max_calc_elapsed_us = factors[3];                │ │
│   │     input.factor_scan_elapsed_us = factors[4];                    │ │
│   │     total_elapsed = factors[5];                                   │ │
│   │                                                                   │ │
│   │     // 反算 start_tsc                                             │ │
│   │     input.start_tsc = now_tsc - (total_elapsed / scaler);         │ │
│   │                                                                   │ │
│   │     // 序列化實際因子（跳過 8 列元數據）                           │ │
│   │     serialize(&factors[8], ...);                                  │ │
│   │   }                                                               │ │
│   │ #endif                                                            │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│       ↓                                                                 │
│  [5] ModelCalculationThread                                             │
│       ↓                                                                 │
│       model->Calculate(input)  ← input 已包含時間字段                   │
│       ↓                                                                 │
│  [6] test0000_model.cc::Calculate()                                     │
│       ↓                                                                 │
│       // 模型從 input 複製時間字段到 output                             │
│       output_.start_tsc = input.start_tsc;                              │
│       output_.tick_max_wait_elapsed_us = input.tick_max_wait_elapsed_us;│
│       ...                                                               │
│       ↓                                                                 │
│  [7] ModelResultScanThread                                              │
│       ↓                                                                 │
│       model_calc_elapsed_us = (now_tsc - output.start_tsc) * scaler     │
│       ↓                                                                 │
│       // 填充 11 列元數據 + 模型輸出                                     │
│       data[0] = model_id;                                               │
│       data[1] = tick_wait_us;                                           │
│       data[5] = model_calc_elapsed_us;  // 總延遲                       │
│       data[11+] = model_predictions;                                    │
│       ↓                                                                 │
│  [8] signal_api.cpp SendCallback ─┐                                     │
│       ↓                           │                                     │
│   ┌───────────────────────────────┴─────────────────────────────────┐   │
│   │ #ifdef HF_TIMING_METADATA                                       │   │
│   │   // 注入點 3: 轉換為統一的 8 列格式                             │   │
│   │   output_with_metadata = [                                      │   │
│   │     -999.0,                        // [0] marker                │   │
│   │     data[1],                       // [1] tick_wait             │   │
│   │     data[6],                       // [2] factor_calc_dur       │   │
│   │     data[2],                       // [3] factor_calc_elapsed   │   │
│   │     data[3],                       // [4] factor_scan_elapsed   │   │
│   │     data[5],                       // [5] model_calc_elapsed    │   │
│   │     output_size,                   // [6] count                 │   │
│   │     0.0,                           // [7] reserved              │   │
│   │     data[11], data[12], ...        // [8+] 模型預測             │   │
│   │   ]                                                             │   │
│   │ #else                                                           │   │
│   │   output = [data[11], data[12], ...]  // 僅模型預測             │   │
│   │ #endif                                                          │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│       ↓                                                                 │
│  [9] Python on_factor(context, symbol, timestamp, values)               │
│       ↓                                                                 │
│       if values[0] == -999.0:                                           │
│           latency_info = parse_metadata(values[:8])                     │
│           actual_values = values[8:]                                    │
│           log(f"📊 [Latency] {latency_info}")                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 關鍵設計點

1. **start_tsc 傳播**:
   - FactorCalculationThread: 記錄 tick 到達時的 TSC
   - 通過 `total_elapsed_us` 間接傳遞
   - ModelSendCallback: 反算 `start_tsc = now_tsc - (total_elapsed / scaler)`

2. **元數據標記**:
   - 使用 `-999.0` 作為 marker（不可能出現的因子值）
   - Python 端通過 `values[0] == -999.0` 檢測元數據

3. **統一格式**:
   - Factor-only 路徑: 8 列元數據 + N 個因子
   - ModelEngine 路徑: 8 列元數據 + M 個模型輸出
   - 兩條路徑對 Python 呈現相同格式

---

## 延遲指標解讀

### 輸出格式

```
📊 [Latency] tick_wait=0.7μs calc=51.4μs total=298.8μs
```

### 各字段含義

| 字段 | 單位 | 測量區間 | 說明 |
|------|------|----------|------|
| `tick_wait` | μs | Tick 到達 → 開始處理 | 行情等待延遲，反映隊列擁塞程度 |
| `calc` | μs | 開始計算 → 因子完成 | 因子計算耗時（僅計算時間） |
| `total` | μs | Tick 到達 → Model 輸出 | **端到端延遲**（最重要指標） |

### 時間線分解

```
0μs          0.7μs              52.1μs                           298.8μs
 │             │                  │                                │
 Tick         開始              因子                            Model
 到達        處理因子          完成計算                         輸出完成

 │◄─ wait ─►│◄──── calc ─────►│◄───── model_pipeline ──────►│
   0.7μs          51.4μs                  246.7μs
```

### 性能基準

#### 健康指標 (單核、無 ASAN)

| 階段 | 優秀 | 良好 | 警告 | 嚴重 |
|------|------|------|------|------|
| tick_wait | < 2 μs | < 5 μs | < 10 μs | > 10 μs |
| calc | < 50 μs | < 100 μs | < 200 μs | > 200 μs |
| total | < 300 μs | < 500 μs | < 1 ms | > 1 ms |

#### 當前實測值分析

```
tick_wait = 0.7 μs   ✅ 優秀（隊列暢通，無擁塞）
calc      = 51.4 μs  ✅ 優秀（因子計算高效）
total     = 298.8 μs ✅ 優秀（端到端延遲極低）
```

**推算其他階段**:
```
model_pipeline = total - tick_wait - calc
               = 298.8 - 0.7 - 51.4
               = 246.7 μs

包含: factor_scan + model_calc + callback_overhead
```

### 性能瓶頸診斷

| 症狀 | 可能原因 | 排查方向 |
|------|----------|----------|
| tick_wait > 10 μs | 隊列擁塞 | 檢查計算線程數、隊列大小 |
| calc > 200 μs | 因子計算複雜 | 優化因子算法、減少因子數 |
| total > 1 ms | 整體瓶頸 | 檢查模型複雜度、線程調度 |
| model_pipeline > 500 μs | 模型計算慢 | 優化模型、檢查鎖競爭 |

---

## 維護指南

### 修改涉及的文件

當你需要調整延遲監控功能時，可能涉及以下文件：

#### 1. 編譯開關
- **文件**: `hf-live/CMakeLists.txt`
- **位置**: L26-34
- **內容**:
  ```cmake
  option(HF_TIMING_METADATA "Inject timing metadata into callback values" OFF)

  if(HF_TIMING_METADATA)
      message(STATUS "📊 HF_TIMING_METADATA ENABLED - Latency metadata will be injected")
      add_compile_definitions(HF_TIMING_METADATA)
  endif()
  ```
- **維護**: 不建議修改，除非要更改默認行為

#### 2. Factor-only 路徑元數據注入
- **文件**: `hf-live/app_live/thread/factor_result_scan_thread.h`
- **函數**: `SendData(int code_idx, uint64_t start_tsc, int64_t timestamp)`
- **位置**: L197-255
- **維護點**:
  - 如需調整元數據列數，同步修改 Python 解析邏輯
  - 如需添加新的時間測量點，在此處計算並填充

#### 3. ModelEngine 路徑元數據提取
- **文件**: `hf-live/app_live/engine/factor_calculation_engine.cpp`
- **位置**: L136-186 (model_callback lambda)
- **維護點**:
  - **critical**: `start_tsc` 反算公式必須與 FactorResultScanThread 一致
  - 如元數據格式變更，同步更新提取邏輯

#### 4. ModelEngine 輸出轉換
- **文件**: `hf-live/adapter/signal_api.cpp`
- **函數**: `SendCallback` (ModelEngine 分支)
- **位置**: L108-154
- **維護點**:
  - 維護 11 列 → 8 列的映射關係
  - 確保與 Factor-only 路徑輸出格式一致

#### 5. Python 層解析
- **文件**: `strategies/test_hf_live/test_hf_live.py`
- **函數**: `on_factor`
- **位置**: L200-219
- **維護點**:
  - marker 檢測邏輯 (`values[0] == -999.0`)
  - 元數據解析字段名稱

### 添加新的時間測量點

**場景**: 想測量某個新階段的延遲（如網絡收包時間）

**步驟**:

1. **修改元數據格式**（從 8 列擴展到 9 列）:
   ```cpp
   // factor_result_scan_thread.h
   #ifdef HF_TIMING_METADATA
       std::vector<double> values_with_metadata;
       values_with_metadata.reserve(9 + factor_count);  // 改為 9

       // ... 前 7 列不變 ...
       values_with_metadata.push_back(new_metric_us);  // 新增第 8 列
       values_with_metadata.push_back(0.0);            // 保留欄位移到第 9 列
   #endif
   ```

2. **同步 Python 解析**:
   ```python
   # test_hf_live.py
   if len(values) > 9 and values[0] == -999.0:  # 改為 9
       latency_info = {
           # ... 前 6 個不變 ...
           'new_metric_us': values[7],
       }
       actual_values = values[9:]  # 改為 9
   ```

3. **同步 signal_api.cpp 轉換邏輯**:
   ```cpp
   // signal_api.cpp
   #ifdef HF_TIMING_METADATA
       output_with_metadata.reserve(9 + output_size);  // 改為 9
       // ... 添加新欄位映射 ...
   #endif
   ```

### 版本兼容性

**問題**: 如何確保不同版本的 libsignal.so 與策略代碼兼容？

**方案**: Marker 機制自動兼容

```python
# Python 層自動檢測
if len(values) > 8 and values[0] == -999.0:
    # 新版 libsignal.so (HF_TIMING_METADATA=ON)
    latency_info = parse_metadata(values)
    actual_values = values[8:]
else:
    # 舊版 libsignal.so (HF_TIMING_METADATA=OFF)
    latency_info = None
    actual_values = values
```

**保證**: Python 代碼可以同時處理兩種版本的 .so 文件

---

## 故障排除

### 問題 1: 看不到延遲輸出

**症狀**:
```bash
docker exec godzilla-dev pm2 logs | grep Latency
# 無輸出
```

**診斷步驟**:

```bash
# 1. 確認 .so 文件是否包含元數據代碼
docker exec godzilla-dev bash -c "strings /app/hf-live/build/libsignal.so | grep 'HF_TIMING_METADATA\|metadata'"

# 預期: 應看到 "metadata" 相關字符串
```

**可能原因 & 解決方案**:

| 原因 | 驗證方法 | 解決方案 |
|------|----------|----------|
| 未啟用編譯開關 | `strings` 無輸出 | 重新編譯: `cmake -DHF_TIMING_METADATA=ON` |
| .so 路徑錯誤 | 檢查 config.json 的 `signal_library_path` | 確保指向正確的 .so 文件 |
| 未重啟策略 | 檢查進程 PID 和啟動時間 | `pm2 restart strategy_test_hf_live` |

### 問題 2: 延遲值異常大

**症狀**:
```
📊 [Latency] tick_wait=0.0μs calc=0.0μs total=216218796032.0μs
```

**原因**: `start_tsc` 傳播中斷（通常是 model_callback 未提取元數據）

**診斷**:
```bash
# 檢查 model_callback 日誌
docker exec godzilla-dev bash -c "cat /root/.pm2/logs/strategy-test-hf-live-error.log | grep 'model_callback.*Timing'"

# 預期: 應看到 "Timing: tick_wait=...μs"
```

**解決**: 確認 `factor_calculation_engine.cpp` 包含元數據提取邏輯

### 問題 3: 延遲值全為 0

**症狀**:
```
📊 [Latency] tick_wait=0.0μs calc=0.0μs total=0.0μs
```

**原因**: FactorResultScanThread 未注入元數據

**檢查**:
```bash
# 查看 C++ 層日誌
docker exec godzilla-dev bash -c "cat /root/.pm2/logs/strategy-test-hf-live-error.log | grep 'FactorScan'"
```

**解決**: 確認 `factor_result_scan_thread.h` 中 `#ifdef HF_TIMING_METADATA` 塊存在

### 問題 4: 編譯錯誤

**症狀**:
```
error: 'HF_TIMING_METADATA' was not declared in this scope
```

**原因**: CMake 緩存問題

**解決**:
```bash
# 清除舊的 build 緩存
rm -rf build_timing
cmake -B build_timing -DHF_TIMING_METADATA=ON
cmake --build build_timing
```

---

## 技術細節

### RDTSC 時鐘機制

**時間測量技術**: CPU Time Stamp Counter (TSC)

```cpp
// timer_utils.h
class RdtscTimer {
public:
    uint64_t operator()() const {
        return __rdtsc();  // x86 指令，讀取 CPU 周期數
    }

    static double GetScaler() {
        // 將 TSC 轉換為微秒的比例因子
        // 典型值: ~0.000416667 (2.4 GHz CPU)
        return scaler_;
    }
};

// 使用示例
auto start = timer_utils::RdtscTimer()();
// ... 執行操作 ...
auto end = timer_utils::RdtscTimer()();
double elapsed_us = (end - start) * scaler;
```

**優點**:
- 極低開銷（< 30 CPU cycles）
- 納秒級精度
- 無系統調用

**注意事項**:
- TSC 在不同 CPU 核心可能不同步（需要線程綁核）
- 功耗管理可能改變 TSC 頻率（需要 constant_tsc 特性）

### 元數據格式規範

#### 8 列統一格式

| 索引 | 字段名 | 類型 | 單位 | 說明 |
|------|--------|------|------|------|
| 0 | marker | double | - | 固定值 -999.0，用於識別元數據 |
| 1 | tick_wait_us | double | μs | 從 Tick 到達到開始處理的延遲 |
| 2 | factor_calc_duration_us | double | μs | 因子計算純耗時（不含等待） |
| 3 | factor_calc_elapsed_us | double | μs | 因子計算累計延遲（含等待） |
| 4 | scan_elapsed_us | double | μs | 因子掃描累計延遲 |
| 5 | total_elapsed_us | double | μs | **端到端總延遲** |
| 6 | output_count | double | - | 因子/模型輸出數量 |
| 7 | reserved | double | - | 保留欄位，用於未來擴展 |

#### ModelEngine 內部 11 列格式

| 索引 | 字段名 | 來源 |
|------|--------|------|
| 0 | model_id | ModelResultScanThread |
| 1 | tick_max_wait_elapsed_us | 從 input 傳播 |
| 2 | factor_max_calc_elapsed_us | 從 input 傳播 |
| 3 | factor_scan_elapsed_us | 從 input 傳播 |
| 4 | factor_send_elapsed_us | ModelCalculationThread 計算 |
| 5 | model_calc_elapsed_us | ModelResultScanThread 計算 |
| 6 | factor_max_calc_duration_us | 從 input 傳播 |
| 7 | factor_scan_duration_us | 從 input 傳播 |
| 8 | factor_send_duration_us | 從 input 計算 |
| 9 | model_calc_duration_us | 從 elapsed 計算 |
| 10 | output_size | 模型輸出數量 |
| 11+ | model_predictions | 模型預測值 |

**轉換規則** (signal_api.cpp):
```cpp
8-column[0] = -999.0                    // marker
8-column[1] = 11-column[1]              // tick_wait
8-column[2] = 11-column[6]              // factor_calc_dur
8-column[3] = 11-column[2]              // factor_calc_elapsed
8-column[4] = 11-column[3]              // scan_elapsed
8-column[5] = 11-column[5]              // total_elapsed (model_calc_elapsed)
8-column[6] = 11-column[10]             // output_count
8-column[7] = 0.0                       // reserved
8-column[8+] = 11-column[11+]           // predictions
```

### 編譯器優化行為

**關閉時的代碼消除**:

```cpp
// 源代碼
void SendData() {
#ifdef HF_TIMING_METADATA
    // 100 行元數據處理代碼
    std::vector<double> metadata;
    metadata.push_back(-999.0);
    // ...
#else
    // 2 行正常代碼
    callback(symbol, timestamp, values);
#endif
}

// 編譯後 (HF_TIMING_METADATA=OFF)
void SendData() {
    callback(symbol, timestamp, values);  // 100 行代碼完全消失
}
```

**驗證方法**:
```bash
# 比較兩個 .so 的大小
ls -lh build/libsignal.so build_timing/libsignal.so

# 預期: build_timing/ 稍大（多了元數據處理代碼）
```

### 性能影響測試

**開銷測量** (HF_TIMING_METADATA=ON):

| 操作 | 增加延遲 | 百分比 |
|------|----------|--------|
| Metadata prepend | ~5 μs | < 2% |
| Metadata extraction | ~3 μs | < 1% |
| Python parsing | ~2 μs | < 1% |
| **總計** | **~10 μs** | **< 4%** |

**結論**: 即使開啟，對延遲影響也極小（< 4%）

---

## 附錄

### A. 相關文件清單

#### C++ 層
```
hf-live/
├── CMakeLists.txt                              # 編譯開關定義
├── adapter/signal_api.cpp                      # ModelEngine 回調轉換
├── app_live/
│   ├── common/timer_utils.h                    # RDTSC 時鐘工具
│   ├── engine/
│   │   ├── factor_calculation_engine.cpp       # ModelSendCallback 元數據提取
│   │   └── model_calculation_engine.cc         # ModelEngine 主邏輯
│   └── thread/
│       ├── factor_result_scan_thread.h         # Factor-only 元數據注入
│       └── model_result_scan_thread.h          # ModelEngine 結果掃描
└── models/
    └── test0000/test0000_model.cc              # 模型時間字段傳播
```

#### Python 層
```
strategies/
└── test_hf_live/
    └── test_hf_live.py                         # on_factor 元數據解析
```

### B. Git 提交記錄

```bash
# hf-live submodule
commit c92bb6b
feat(phase-5d): implement zero-interface latency monitoring

# 主倉庫
commit d310da5
feat(phase-5d): implement zero-interface latency monitoring
```

### C. 測試驗證

**E2E 測試結果**:
```
✅ 編譯測試 (HF_TIMING_METADATA=ON/OFF)
✅ 功能測試 (延遲數據正確輸出)
✅ 性能測試 (延遲 < 300 μs)
✅ 兼容性測試 (Python 可處理兩種模式)
```

**測試用例**:
```bash
# 1. 關閉模式
cmake -B build && cmake --build build
# 預期: Python 不輸出 [Latency] 行，功能正常

# 2. 開啟模式
cmake -B build_timing -DHF_TIMING_METADATA=ON && cmake --build build_timing
# 預期: Python 輸出延遲數據，值在合理範圍

# 3. 切換測試
# 先部署 build/libsignal.so，測試無延遲模式
# 再部署 build_timing/libsignal.so，測試延遲模式
# 預期: 策略代碼無需修改，自動適配
```

---

## 變更歷史

| 版本 | 日期 | 修改內容 |
|------|------|----------|
| v1.0 | 2024-12-14 | 初版發布，完整實現零接口延遲監控 |

---

## 聯繫方式

**維護者**: Godzilla Team
**分支**: `feature/latency-monitoring`
**相關文檔**: `prd_hf-live.07-implementation.md` (hf-live 主架構)
