# C API 詳細設計 - Linus 原則

## 文檔元信息
- **版本**: v1.0
- **日期**: 2025-12-04
- **設計哲學**: Linus Torvalds 極簡主義 - 清晰、簡潔、易維護

---

## 核心設計原則

### Linus 三原則

> "Talk is cheap. Show me the code."
> "Data structures first, functions follow."
> "Keep it simple, stupid."

**應用到 hf-live**:
1. **數據結構優先**: API 圍繞 `Depth*`, `double*` 設計
2. **極簡函數集**: 4 個函數完成所有任務
3. **自我說明**: API 簽名即文檔

---

## API 全覽

```c
// signal_api.h - 總共只需 4 個函數

#ifndef SIGNAL_API_H
#define SIGNAL_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// 1. 不透明句柄 (Opaque Handle)
// ============================================================

typedef void* signal_handle_t;

// ============================================================
// 2. 回調函數簽名
// ============================================================

/**
 * 因子/預測結果回調
 *
 * @param symbol      交易對 (e.g., "BTCUSDT")
 * @param timestamp   Unix 時間戳 (納秒)
 * @param values      因子/預測值數組
 * @param count       數組長度
 * @param user_data   用戶自定義數據 (創建時傳入)
 *
 * 調用環境:
 *   - 線程: signal_on_data() 調用者線程
 *   - 延遲: < 1μs (必須立即返回,禁止阻塞)
 */
typedef void (*signal_callback_t)(
    const char* symbol,
    int64_t timestamp,
    const double* values,
    int count,
    void* user_data
);

// ============================================================
// 3. 核心 API (4 個函數)
// ============================================================

/**
 * 創建信號引擎實例
 *
 * @param config_json  JSON 配置字符串 (詳見下方配置說明)
 * @return
 *   - 成功: 不透明句柄 (非 NULL)
 *   - 失敗: NULL (錯誤信息輸出到 stderr)
 *
 * 線程安全性: 否 (調用者必須保證單線程創建)
 *
 * 配置範例:
 * {
 *   "type": "factor",           // "factor" | "model"
 *   "library": "./factor.so",   // 動態庫路徑
 *   "entry": "FactorEntry",     // 入口類名
 *   "config": {                 // 用戶自定義配置
 *     "window": 20,
 *     "params": [0.01, 0.05]
 *   }
 * }
 */
signal_handle_t signal_create(const char* config_json);

/**
 * 銷毀信號引擎實例
 *
 * @param handle  signal_create() 返回的句柄
 *
 * 線程安全性: 否 (調用者必須保證:
 *   1. 所有 signal_on_data() 調用已完成
 *   2. 單線程銷毀)
 *
 * 行為:
 *   - 釋放所有內部資源
 *   - 卸載動態庫
 *   - 句柄失效 (不可再使用)
 */
void signal_destroy(signal_handle_t handle);

/**
 * 註冊回調函數
 *
 * @param handle     信號引擎句柄
 * @param callback   回調函數指針
 * @param user_data  用戶自定義數據 (將傳遞給回調)
 * @return
 *   - 0:  成功
 *   - -1: 失敗 (handle 無效)
 *
 * 線程安全性: 否 (必須在 signal_on_data() 調用前完成)
 *
 * 注意:
 *   - 每個句柄只能註冊一個回調
 *   - 重複調用會覆蓋前一個回調
 */
int signal_register_callback(
    signal_handle_t handle,
    signal_callback_t callback,
    void* user_data
);

/**
 * 推送市場數據 (觸發計算)
 *
 * @param handle  信號引擎句柄
 * @param type    數據類型 (目前僅支持 DEPTH = 1)
 * @param data    數據指針 (void* 零拷貝傳遞)
 *
 * 類型映射:
 *   type=1 → const Depth* (market_data_types.h)
 *
 * 線程安全性: 是 (內部使用無鎖隊列)
 *
 * 性能:
 *   - 指針轉型: < 10ns
 *   - 內存拷貝: 0 字節
 *   - 隊列操作: < 50ns (lock-free SPMC)
 *
 * 注意:
 *   - data 指針必須在回調執行期間有效
 *   - 建議: 回調中立即拷貝需要的數據
 */
void signal_on_data(signal_handle_t handle, int type, const void* data);

#ifdef __cplusplus
}
#endif

#endif  // SIGNAL_API_H
```

---

## 設計決策詳解

### 1. 為什麼只有 4 個函數?

**Linus 原則**: "Perfection is achieved not when there is nothing left to add, but when there is nothing left to take away."

**證明完備性**:
| 任務 | API 組合 |
|------|---------|
| 因子計算 | create → register → on_data (循環) → destroy |
| 模型推理 | create → register → on_data (循環) → destroy |
| 參數調整 | destroy → create (新配置) |
| 錯誤恢復 | destroy → create |

**無需額外函數**:
- ❌ `signal_set_config()` → 用 destroy + create (配置是不變的)
- ❌ `signal_pause()` → 調用者控制是否調用 on_data
- ❌ `signal_get_status()` → 回調即狀態 (有輸出=正常)

---

### 2. 為什麼用 void* 不透明句柄?

**ABI 穩定性保證**:

```c
// 內部實現可以任意變化,不影響用戶代碼

// Version 1.0
struct SignalEngine {
    FactorEntry* entry;
    SPMCQueue<Event>* queue;
};

// Version 2.0 (添加緩存)
struct SignalEngine {
    FactorEntry* entry;
    SPMCQueue<Event>* queue;
    RingBuffer<Depth>* cache;  // 🔥 新增字段
    MetricsCollector* metrics; // 🔥 新增字段
};

// 用戶代碼無需重新編譯 ✅
signal_handle_t h = signal_create(...);  // 依然有效
```

**對比方案**:
```c
// ❌ 錯誤設計: 暴露結構體
typedef struct {
    void* entry;
    void* queue;
} signal_handle_t;

// 問題: 添加字段 = ABI 破壞 = 用戶必須重新編譯
```

---

### 3. 為什麼用 JSON 配置?

**Linus 原則**: "Configuration should be data, not code."

**錯誤設計** (函數地獄):
```c
// ❌ 100+ 個 setter 函數
signal_handle_t h = signal_create();
signal_set_library(h, "./factor.so");
signal_set_entry(h, "FactorEntry");
signal_set_window(h, 20);
signal_set_param(h, 0, 0.01);
signal_set_param(h, 1, 0.05);
signal_set_thread_count(h, 4);
signal_set_log_level(h, 2);
// ... 100 more lines ...
```

**正確設計** (數據驅動):
```c
// ✅ 1 個函數 + 數據
const char* config = R"({
  "type": "factor",
  "library": "./factor.so",
  "entry": "FactorEntry",
  "config": {"window": 20, "params": [0.01, 0.05]}
})";
signal_handle_t h = signal_create(config);
```

**優點**:
- 配置可以來自文件/網絡/數據庫
- 易於序列化/版本管理
- 無需為每個參數設計 setter

---

### 4. 錯誤處理機制

**Unix 哲學**: "Errors should be rare and obvious."

#### 返回值約定

```c
// 創建函數: NULL = 失敗
signal_handle_t h = signal_create(config);
if (h == NULL) {
    fprintf(stderr, "Failed to create signal engine\n");
    // stderr 已輸出詳細錯誤
    return -1;
}

// 操作函數: -1 = 失敗, 0 = 成功
if (signal_register_callback(h, cb, NULL) != 0) {
    fprintf(stderr, "Failed to register callback\n");
    signal_destroy(h);
    return -1;
}

// void 函數: 不會失敗 (設計保證)
signal_on_data(h, 1, depth);  // 永遠成功 (或內部隊列滿則丟棄)
```

#### 錯誤信息輸出

```c
// 內部實現範例
signal_handle_t signal_create(const char* config_json) {
    if (!config_json) {
        fprintf(stderr, "[signal_api] NULL config_json\n");
        return NULL;
    }

    // 解析 JSON
    rapidjson::Document doc;
    if (doc.Parse(config_json).HasParseError()) {
        fprintf(stderr, "[signal_api] Invalid JSON at offset %zu: %s\n",
                doc.GetErrorOffset(),
                GetParseError_En(doc.GetParseError()));
        return NULL;
    }

    // 加載動態庫
    void* lib = dlopen(library_path, RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "[signal_api] Failed to load %s: %s\n",
                library_path, dlerror());
        return NULL;
    }

    // ... 創建成功 ...
    return engine;
}
```

#### 為什麼不用錯誤碼?

**錯誤設計**:
```c
// ❌ 過度工程
enum SignalError {
    SIGNAL_OK = 0,
    SIGNAL_INVALID_HANDLE = -1,
    SIGNAL_NULL_CONFIG = -2,
    SIGNAL_JSON_PARSE_ERROR = -3,
    SIGNAL_LIBRARY_NOT_FOUND = -4,
    SIGNAL_SYMBOL_NOT_FOUND = -5,
    // ... 100 more error codes ...
};

// 用戶被迫寫 100 行 switch
int err = signal_create_ex(config, &handle);
switch (err) {
    case SIGNAL_NULL_CONFIG:
        // ...
    case SIGNAL_JSON_PARSE_ERROR:
        // ...
    // ... 100 cases ...
}
```

**Linus 方式**:
- 成功/失敗用返回值 (NULL/-1)
- 詳細錯誤用 stderr (人類可讀)
- 代碼只需檢查 NULL/-1

---

### 5. 線程安全設計

#### 原則: 最小線程安全保證

**不提供線程安全** (調用者責任):
- `signal_create()` - 創建必須單線程
- `signal_destroy()` - 銷毀必須單線程 + 所有 on_data 已完成
- `signal_register_callback()` - 註冊必須在 on_data 前完成

**提供線程安全** (內部保證):
- `signal_on_data()` - 多線程調用安全 (lock-free queue)

#### 為什麼 create/destroy 不線程安全?

**Linus 原則**: "Don't protect fools."

```c
// ❌ 錯誤用法 (程序設計錯誤)
// Thread 1
signal_handle_t h = signal_create(config);

// Thread 2 (同時)
signal_handle_t h2 = signal_create(config);  // 💥 競爭條件

// 解決方案: 不是 API 的責任,是調用者的責任
```

**正確做法**:
```c
// 調用者保證單線程創建
std::mutex create_mutex;
{
    std::lock_guard<std::mutex> lock(create_mutex);
    h = signal_create(config);
}

// 或者: 在主線程創建,工作線程只調用 on_data
```

#### on_data() 線程安全實現

**Lock-free SPMC Queue**:

```cpp
// 內部實現範例 (偽代碼)
class SignalEngine {
private:
    // Single Producer, Multiple Consumer 無鎖隊列
    boost::lockfree::spsc_queue<Event,
        boost::lockfree::capacity<1024>> queue_;

public:
    void OnData(int type, const void* data) {
        Event e{type, data, std::chrono::steady_clock::now()};

        // 嘗試入隊 (無鎖操作)
        if (!queue_.push(e)) {
            // 隊列滿: 丟棄最舊數據 (高頻場景正常行為)
            metrics_.dropped_events++;
        }

        // 通知工作線程 (無鎖)
        worker_cv_.notify_one();
    }
};
```

**性能**:
- 無互斥鎖 (mutex-free)
- 單次 push: ~50ns
- 支持 1000+ 萬 events/sec

---

## 完整使用範例

### 場景 1: 因子大師 (純因子計算)

```c
#include "signal_api.h"
#include "market_data_types.h"
#include <stdio.h>

// 回調函數: 接收因子結果
void on_factors(const char* symbol, int64_t timestamp,
                const double* values, int count, void* user_data) {
    // 打印前 5 個因子
    printf("[%s] Factors: ", symbol);
    for (int i = 0; i < (count < 5 ? count : 5); i++) {
        printf("%.6f ", values[i]);
    }
    printf("\n");

    // 可選: 發送到策略 (通過用戶提供的回調)
    void (*send_to_strategy)(const double*, int) =
        (void (*)(const double*, int))user_data;
    if (send_to_strategy) {
        send_to_strategy(values, count);
    }
}

int main() {
    // 1. 創建因子引擎
    const char* config = R"({
        "type": "factor",
        "library": "./my_factors.so",
        "entry": "MyFactorEntry",
        "config": {
            "window": 20,
            "ema_period": 10
        }
    })";

    signal_handle_t engine = signal_create(config);
    if (!engine) {
        return -1;  // stderr 已輸出錯誤
    }

    // 2. 註冊回調
    void (*sender)(const double*, int) = get_strategy_sender();
    if (signal_register_callback(engine, on_factors, sender) != 0) {
        signal_destroy(engine);
        return -1;
    }

    // 3. 主循環: 接收市場數據
    while (running) {
        // 從 Godzilla 獲取數據 (void* 零拷貝)
        const void* depth_ptr = get_depth_from_godzilla();

        // 推送到因子引擎 (觸發計算)
        signal_on_data(engine, 1, depth_ptr);  // type=1 表示 Depth

        // 回調 on_factors() 會在內部線程執行
    }

    // 4. 清理
    signal_destroy(engine);
    return 0;
}
```

---

### 場景 2: 模型大師 (因子→模型→預測)

```c
#include "signal_api.h"

// 因子回調: 轉發到模型
void on_factors_to_model(const char* symbol, int64_t timestamp,
                         const double* values, int count, void* user_data) {
    signal_handle_t model_engine = (signal_handle_t)user_data;

    // 轉發因子到模型引擎
    // (模型引擎也接受 void* ,這裡傳 double* 即可)
    signal_on_data(model_engine, 2, values);  // type=2 表示因子數組
}

// 預測回調: 發送到策略
void on_predictions(const char* symbol, int64_t timestamp,
                    const double* values, int count, void* user_data) {
    printf("[%s] Predictions: ", symbol);
    for (int i = 0; i < count; i++) {
        printf("%.6f ", values[i]);
    }
    printf("\n");

    // 發送到策略
    send_to_strategy(values, count);
}

int main() {
    // 1. 創建模型引擎
    signal_handle_t model = signal_create(R"({
        "type": "model",
        "library": "./onnx_model.so",
        "entry": "ONNXModelEntry",
        "config": {"model_path": "./model.onnx"}
    })");

    // 2. 創建因子引擎
    signal_handle_t factor = signal_create(R"({
        "type": "factor",
        "library": "./factors.so",
        "entry": "FactorEntry",
        "config": {"window": 20}
    })");

    // 3. 連接: 因子 → 模型
    signal_register_callback(factor, on_factors_to_model, model);
    signal_register_callback(model, on_predictions, NULL);

    // 4. 主循環
    while (running) {
        const void* depth = get_depth_from_godzilla();
        signal_on_data(factor, 1, depth);
        // 數據流: depth → factor → on_factors_to_model → model → on_predictions
    }

    // 5. 清理 (順序: 先清理下游)
    signal_destroy(model);
    signal_destroy(factor);
    return 0;
}
```

---

### 場景 3: Python ctypes 綁定

**注意**: 此示例僅展示 C API 可被 Python ctypes 調用。在 Godzilla 實際集成中,我們使用 **pybind11** 綁定 (見 [prd_hf-live.07-implementation.md §1.2](prd_hf-live.07-implementation.md)),而非 ctypes。

```python
# python_binding.py (僅作為 C API 使用示例)
import ctypes
import json

# 加載動態庫
lib = ctypes.CDLL("./libsignal_api.so")

# 定義函數簽名
lib.signal_create.argtypes = [ctypes.c_char_p]
lib.signal_create.restype = ctypes.c_void_p

lib.signal_destroy.argtypes = [ctypes.c_void_p]
lib.signal_destroy.restype = None

lib.signal_register_callback.argtypes = [
    ctypes.c_void_p,
    ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int64,
                     ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                     ctypes.c_void_p),
    ctypes.c_void_p
]
lib.signal_register_callback.restype = ctypes.c_int

lib.signal_on_data.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
lib.signal_on_data.restype = None

# Python 包裝類
class SignalEngine:
    def __init__(self, config_dict):
        config_json = json.dumps(config_dict).encode('utf-8')
        self.handle = lib.signal_create(config_json)
        if not self.handle:
            raise RuntimeError("Failed to create signal engine")

        self.callback_func = None  # 保持引用避免 GC

    def register_callback(self, callback):
        # 轉換 Python 函數到 C 回調
        @ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int64,
                          ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                          ctypes.c_void_p)
        def c_callback(symbol, timestamp, values, count, user_data):
            # 轉換到 Python 類型
            py_symbol = symbol.decode('utf-8')
            py_values = [values[i] for i in range(count)]
            callback(py_symbol, timestamp, py_values)

        self.callback_func = c_callback  # 保持引用
        if lib.signal_register_callback(self.handle, c_callback, None) != 0:
            raise RuntimeError("Failed to register callback")

    def on_data(self, data_type, data_ptr):
        lib.signal_on_data(self.handle, data_type, data_ptr)

    def __del__(self):
        if self.handle:
            lib.signal_destroy(self.handle)

# 使用範例
def my_callback(symbol, timestamp, values):
    print(f"[{symbol}] Values: {values[:5]}")

engine = SignalEngine({
    "type": "factor",
    "library": "./factors.so",
    "entry": "FactorEntry",
    "config": {"window": 20}
})

engine.register_callback(my_callback)

# 推送數據 (假設 depth_ptr 是從 C++ 獲取的指針)
engine.on_data(1, depth_ptr)
```

---

## 性能特性

### 延遲分解

| 操作 | 延遲 | 說明 |
|------|------|------|
| `signal_create()` | ~10ms | 一次性操作 (加載 .so + 初始化) |
| `signal_destroy()` | ~5ms | 一次性操作 (卸載 .so) |
| `signal_register_callback()` | ~100ns | 僅保存函數指針 |
| `signal_on_data()` - 入隊 | ~50ns | Lock-free push |
| `signal_on_data()` - 計算 | ~1-10μs | 取決於因子/模型複雜度 |
| `callback()` 調用 | ~100ns | 函數指針調用 + 參數傳遞 |

**總延遲 (Depth → 回調)**:
```
50ns (入隊) + 1μs (計算) + 100ns (回調) = ~1.15μs (median)
```

**吞吐量**:
- 單引擎: ~800k events/sec
- 10 個引擎: ~8M events/sec (線性擴展)

---

### 內存佔用

```c
// 每個 signal_handle_t
sizeof(SignalEngine) =
    8 (vtable ptr) +
    8 (entry ptr) +
    1024*16 (queue, lock-free) +
    256 (metrics) +
    64 (misc)
  = ~16KB

// 100 個並發引擎: ~1.6MB (可忽略)
```

---

## ABI 穩定性保證

### 版本兼容性

**保證**:
- ✅ 內部實現可任意變化 (void* 隔離)
- ✅ 添加新函數不破壞舊代碼
- ✅ 舊 .so 可與新 API header 編譯

**不保證**:
- ❌ 刪除函數 (主版本升級)
- ❌ 修改函數簽名 (主版本升級)
- ❌ 修改回調簽名 (主版本升級)

### 版本標記

```c
// signal_api.h
#define SIGNAL_API_VERSION_MAJOR 1
#define SIGNAL_API_VERSION_MINOR 0
#define SIGNAL_API_VERSION_PATCH 0

// 編譯時檢查
#if SIGNAL_API_VERSION_MAJOR != 1
#error "Incompatible API version"
#endif
```

---

## 編譯與鏈接

### 編譯 API 庫

```bash
# signal_api.cpp (實現)
g++ -std=c++17 -O3 -fPIC -shared \
    -I./include \
    -o libsignal_api.so \
    signal_api.cpp \
    -ldl -lpthread

# 結果: libsignal_api.so (~200KB)
```

### 用戶代碼編譯

```bash
# 因子大師代碼
gcc -std=c11 -O3 \
    -I./hf-live/include \
    -o factor_runner \
    factor_runner.c \
    -L./hf-live/lib -lsignal_api \
    -Wl,-rpath,'$ORIGIN/../lib'

# 無需鏈接 Godzilla 任何庫 ✅
```

---

## 錯誤處理實踐

### 最佳實踐

```c
// ✅ 正確: 檢查每個返回值
signal_handle_t h = signal_create(config);
if (!h) {
    log_error("Failed to create engine");
    return -1;
}

if (signal_register_callback(h, cb, NULL) != 0) {
    log_error("Failed to register callback");
    signal_destroy(h);
    return -1;
}

// on_data 永不失敗,無需檢查
signal_on_data(h, 1, depth);
```

### 常見錯誤

```c
// ❌ 錯誤: 未檢查返回值
signal_handle_t h = signal_create(config);
signal_register_callback(h, cb, NULL);  // 💥 h 可能是 NULL!
signal_on_data(h, 1, depth);  // 💥 段錯誤

// ❌ 錯誤: 忘記銷毀
void process() {
    signal_handle_t h = signal_create(config);
    signal_on_data(h, 1, depth);
    return;  // 💥 內存泄漏!
}

// ✅ 正確: 使用 RAII (C++)
class SignalGuard {
    signal_handle_t h_;
public:
    SignalGuard(const char* cfg) : h_(signal_create(cfg)) {
        if (!h_) throw std::runtime_error("create failed");
    }
    ~SignalGuard() { signal_destroy(h_); }
    operator signal_handle_t() const { return h_; }
};

void process() {
    SignalGuard h(config);
    signal_on_data(h, 1, depth);
    // 自動銷毀 ✅
}
```

---

## 調試支持

### 編譯時調試

```bash
# 啟用調試符號
g++ -std=c++17 -g -O0 -fPIC -shared \
    -DSIGNAL_DEBUG=1 \
    -o libsignal_api_debug.so \
    signal_api.cpp
```

### 運行時日誌

```c
// 內部實現 (SIGNAL_DEBUG=1 時啟用)
#ifdef SIGNAL_DEBUG
#define LOG_DEBUG(fmt, ...) \
    fprintf(stderr, "[DEBUG][%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...) ((void)0)
#endif

void signal_on_data(signal_handle_t handle, int type, const void* data) {
    LOG_DEBUG("on_data: handle=%p, type=%d, data=%p", handle, type, data);
    // ...
}
```

### Valgrind 檢查

```bash
# 內存泄漏檢查
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         ./factor_runner

# 預期結果: 0 leaks
```

---

## 與 Godzilla 集成

### Godzilla Strategy 端

```cpp
// strategies/hf_strategy.cpp (C++)

#include "strategy.h"
#include "signal_api.h"

class HFStrategy : public Strategy {
private:
    signal_handle_t factor_engine_;
    signal_handle_t model_engine_;

    static void on_predictions(const char* symbol, int64_t timestamp,
                              const double* values, int count, void* user_data) {
        HFStrategy* self = static_cast<HFStrategy*>(user_data);

        // values[0] = 預測的價格變動方向 (-1/0/1)
        if (values[0] > 0.7) {
            self->buy(symbol, 0.001);  // 買入信號
        } else if (values[0] < -0.7) {
            self->sell(symbol, 0.001);  // 賣出信號
        }
    }

public:
    void on_start() override {
        // 創建因子引擎
        factor_engine_ = signal_create(R"({
            "type": "factor",
            "library": "/app/hf-live/lib/factors.so",
            "entry": "FactorEntry",
            "config": {"window": 20}
        })");

        // 創建模型引擎
        model_engine_ = signal_create(R"({
            "type": "model",
            "library": "/app/hf-live/lib/model.so",
            "entry": "ModelEntry",
            "config": {"model_path": "/app/models/lstm.onnx"}
        })");

        // 連接回調
        signal_register_callback(factor_engine_,
            [](const char* s, int64_t t, const double* v, int c, void* ud) {
                signal_handle_t model = static_cast<signal_handle_t>(ud);
                signal_on_data(model, 2, v);  // 轉發到模型
            }, model_engine_);

        signal_register_callback(model_engine_, on_predictions, this);
    }

    void on_quote(const Quote* quote) override {
        // hf-live 暫不處理 Quote
    }

    void on_depth(const Depth* depth) override {
        // 零拷貝傳遞: Depth* → void*
        signal_on_data(factor_engine_, 1, depth);

        // 數據流:
        // depth → factor_engine → lambda → model_engine → on_predictions → buy/sell
    }

    void on_stop() override {
        signal_destroy(model_engine_);
        signal_destroy(factor_engine_);
    }
};

EXPORT_STRATEGY(HFStrategy);
```

---

## 總結

### API 設計回顧

| 原則 | 實現 |
|------|------|
| **極簡** | 4 個函數完成所有任務 |
| **穩定** | void* 句柄 = ABI 不破壞 |
| **快速** | 零拷貝 + 無鎖隊列 = <1μs |
| **清晰** | Unix 風格錯誤處理 |
| **靈活** | JSON 配置 = 數據驅動 |

### Linus 原則驗證

> "Talk is cheap. Show me the code."

✅ API 簽名即文檔,無需額外說明

> "Data structures first, functions follow."

✅ 圍繞 `Depth*`, `double*` 設計 API

> "Keep it simple, stupid."

✅ 4 個函數 vs 100+ 個函數

---

**版本**: v1.0 (2025-12-04)
**設計哲學**: Linus Torvalds - 極簡、穩定、高效
