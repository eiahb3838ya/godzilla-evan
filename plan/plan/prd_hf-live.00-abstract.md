# 實時因子計算框架 - 核心設計

## 文檔元信息
- **版本**: v0.5-final
- **日期**: 2025-12-03 (更新: 開箱即用設計 - SignalSender + 框架自動處理)
- **項目**: hf-live (submodule, private)
- **範疇**: 核心理念與架構設計

---

## 一、核心定位 (30 秒理解)

### 什麼是 hf-live?

**hf-live** = 獨立的因子計算與模型推理引擎 (.so 動態庫)

```
┌──────────────────────────┐
│  godzilla-evan           │  策略執行層
│  (下單引擎 + 策略邏輯)    │  - 行情接收、訂單管理
└────────┬─────────────────┘
         │ .so (C API, void*)
┌────────┴─────────────────┐
│  hf-live                 │  因子計算層
│  (因子計算 + 模型推理)    │  - 基於 Godzilla 數據結構
└──────────────────────────┘
```

### 核心價值

1. **適配 Godzilla**: 完全基於 Godzilla 數據結構,零轉換成本
2. **完全解耦**: 兩個項目零依賴,僅通過 C ABI 通信
3. **性能極致**: void* 零拷貝設計,<10ns 延遲
4. **可擴展性**: 支持多交易所、多數據源的統一接口

---

## 二、設計哲學

### 2.1 Linus 原則

> "Good programmers worry about data structures and their relationships."

- **模塊化**: 每個項目只做一件事並做到極致
- **ABI 邊界**: 通過清晰的 C 接口實現解耦
- **零拷貝**: 性能優先,必要時犧牲維護性

### 2.2 三大設計原則

#### 原則 1: 單向依賴

```
Godzilla 知道:
  ✅ 有個 .so 提供因子計算
  ✅ 需要調用 signal_on_data()
  ✅ 會收到 on_factor() 回調

Godzilla 不知道:
  ❌ 內部如何計算因子
  ❌ 使用了哪些因子庫
```

```
HF-Live 知道:
  ✅ 會收到市場數據 (void* + type)
  ✅ 需要回調返回結果
  ✅ 使用 Godzilla 數據結構定義

HF-Live 不知道:
  ❌ 數據來自 Godzilla 還是其他平台
  ❌ 結果會被如何使用
```

#### 原則 2: 數據結構共享 (Bundled Header)

```cpp
// hf-live 直接包含 Godzilla 數據結構定義
#include "market_data_types.h"  // 已複製到 hf-live/include/

extern "C" void signal_on_data(void* handle, int type, const void* data) {
    switch (type) {
        case DEPTH:
            OnDepth(static_cast<const Depth*>(data));
            break;
        case TRADE:
            OnTrade(static_cast<const Trade*>(data));
            break;
    }
}
```

**保證** (詳見 [prd_hf-live.02-data-structure-sharing.md](prd_hf-live.02-data-structure-sharing.md)):
- ✅ Single Source of Truth (Godzilla msg.h 為唯一來源)
- ✅ 編譯時大小確定 + 零拷貝
- ✅ 極低維護成本 (< 1次/年手動同步)
- ✅ 獨立編譯場景零配置 (header 已在倉庫中)

#### 原則 3: 性能優先

```cpp
// 極致優化: 零拷貝設計
extern "C" void signal_on_data(void* handle, int type, const void* data) {
    const Depth* depth = static_cast<const Depth*>(data);  // 僅指針轉型,0ns
    double price = depth->bid_price[0];  // 直接內存訪問
}
```

**保證**: 版本化 header → 結構定義完全一致 → 零拷貝安全

---

## 三、項目結構

### 3.1 hf-live 項目 (獨立倉庫)

```
hf-live/                              # Git Submodule (Private)
├── include/
│   ├── market_data_types.h          # 直接複製自 Godzilla msg.h
│   └── market_data_types.VERSION    # 版本追蹤文件
│
├── adapter/
│   ├── api.h                        # C API 聲明
│   └── adapter.cpp                  # 數據分發邏輯
│
├── _comm/                           # 框架基礎設施 (自動處理複雜度)
│   ├── signal_sender.h              # 🔥 SignalSender (Engine 調用統一發送)
│   ├── signal_sender.cpp
│   └── engine_base.h                # Engine 基類
│
├── app_live/                        # 框架代碼 (對標 ref app_live)
│   ├── engine.h                     # Engine 統一調度
│   ├── engine.cpp                   # 🔥 收集因子並統一發送
│   └── entry.cpp                    # .so 入口
│
├── factors/                         # 因子庫 (對標 ref factors)
│   ├── _template/                   # 因子模板
│   ├── _comm/                       # 因子基礎類
│   └── my_factors/                  # 🔥 因子大師編寫 (專注計算)
│       ├── factor_entry.h
│       └── factor_entry.cpp         # OnDepth/OnTrade 計算邏輯
│
├── models/                          # 🔥 模型庫 (CORE 組件,對標 ref models)
│   ├── _comm/                       # 模型基礎設施
│   │   ├── model_base.h             # 模型基類
│   │   └── model_registry.h         # 模型註冊
│   └── demo/                        # 示例 ONNX 模型
│
├── CMakeLists.txt
└── build/
    └── libsignal.so                 # 🎯 最終產物
```

### 3.2 godzilla-evan 集成點

```
godzilla-evan/                        # Public Repo
├── core/cpp/wingchun/include/kungfu/wingchun/
│   └── market_data_types.h          # 🔥 數據結構定義 (Single Source of Truth)
│
├── hf-live/                         # Submodule (不上傳源碼)
│   ├── include/
│   │   └── market_data_types.h      # 直接包含在 hf-live 倉庫 (版本化快照)
│   └── build/libsignal.so             # 僅包含編譯產物
│
├── core/cpp/wingchun/src/strategy/
│   └── runner.cpp                   # 🔥 添加: signal_on_data()
│
├── core/python/kungfu/wingchun/
│   └── strategy.py                  # 🔥 擴展: on_factor() 默認空實現
│
└── strategies/factor_strategy/
    └── run.py                       # 🔥 實現: on_factor()
```

---

## 四、核心交互流程

### 4.1 完整數據流 (包含模型預測)

```
Binance → Godzilla MD → runner.cpp events_
    ↓ (runner.cpp 轉發)
signal_on_data(type=DEPTH, data=Depth*)  ← C API (void*)
    ↓ (adapter 分發)
FactorCalculationEngine::OnDepth(const Depth*)
    ↓ (因子計算 - 多線程)
FactorResultScanThread → 收集因子
    ↓ (標準流程)
    ├→ Option A: 直接發送因子 (少見)
    │   └→ SignalSender::Send(factors)
    │
    └→ Option B: 發送到模型 (🔥 常規流程)
        ↓
    ModelCalculationEngine::SendFactors(factors)
        ↓ (ONNX 預測 - 多線程)
    ModelResultScanThread → 收集預測值
        ↓
    SignalSender::Send(predictions)  ← 統一發送接口
        ↓ (C API 邊界)
    factor_callback_(predictions)    ← 函數指針
        ↓
    Python on_factor(predictions)     ← 策略大師不知道是預測值
        ↓ (策略邏輯)
context.insert_order()
```

### 4.2 關鍵節點

| 節點 | 位置 | 職責 |
|------|------|------|
| **自動轉發** | runner.cpp | events_ → signal_on_data() |
| **數據分發** | adapter | void* → OnDepth/OnTrade |
| **因子計算** | FactorCalculationEngine | 基於 Godzilla 數據結構 |
| **模型預測** | ModelCalculationEngine | ONNX Runtime 推理 (🔥 常規) |
| **結果發送** | SignalSender | 統一發送 (因子或預測值) |
| **結果回調** | adapter | 調用 factor_callback_ |
| **策略決策** | strategies/run.py | on_factor → insert_order |

---

## 五、C API 設計 (核心接口)

**完整 C API 設計**: 見 [prd_hf-live.06-c-api-detail.md](prd_hf-live.06-c-api-detail.md)

**核心原則**: Linus 極簡主義 - 4 個函數完成所有任務

```c
extern "C" {
    void* signal_create(const char* config_json);
    void signal_register_callback(void* handle, factor_callback_fn cb, void* user_data);
    void signal_on_data(void* handle, int type, const void* data);
    void signal_destroy(void* handle);
}
```

**關鍵特性**:
- ✅ Opaque handle (void*) - ABI 穩定性
- ✅ 零拷貝設計 (<10ns)
- ✅ Unix 風格錯誤處理 (NULL/-1 + stderr)
- ✅ 線程安全 (Lock-free SPMC queue)

### 5.2 數據類型設計 (統一 vs 交易所前綴)

**詳細設計決策**: 見 [prd_hf-live.06-c-api-detail.md §4](prd_hf-live.06-c-api-detail.md)

**核心原則**: 統一類型 + 運行時字段區分

```cpp
enum MarketDataType : int32_t {
    DEPTH = 101,    // 所有交易所共享
    TRADE = 103,
    TICKER = 102,
};

// 運行時通過 exchange_id 字段區分
extern "C" void signal_on_data(void* handle, int type, const void* data) {
    const Depth* d = static_cast<const Depth*>(data);
    if (strcmp(d->exchange_id, "binance") == 0) { /* ... */ }
}
```

**優勢**:
- ✅ 新增交易所無需修改 API
- ✅ 符合 Godzilla 設計哲學 (類型描述性質,字段描述來源)
- ✅ Depth/Trade 結構已包含 `exchange_id` 字段

---

## 六、關鍵技術決策

### 6.1 為什麼用 .so 而非獨立進程?

| 方案 | 延遲 | 隔離性 | 決策 |
|------|------|--------|------|
| .so | ~0ns | 低 | ✅ 選擇 (高頻交易延遲敏感) |
| IPC | ~1-10μs | 高 | ❌ 不可接受 |

### 6.2 為什麼用 C ABI?

- ✅ 跨編譯器穩定 (GCC 4.x ↔ GCC 11.x)
- ✅ 跨語言兼容 (Python ctypes, Rust FFI)
- ❌ C++ ABI 不穩定 (虛函數表、異常處理)

### 6.3 回調機制詳解 (Python → C++ → hf-live → Python)

**核心問題**: hf-live (.so) 如何將因子計算結果回傳給 Python 策略?

#### 完整數據流 (開箱即用設計)

```
┌─────────────────────────────────────────────────────────────┐
│  Python Strategy (strategies/run.py)                        │
│  def on_factor(context, symbol, timestamp, values):         │
│      context.insert_order(...)  # 🎯 策略大師只寫這個       │
└──────────────────┬──────────────────────────────────────────┘
                   ↑ (5) 框架自動調用
         ┌─────────┴─────────┐
         │  Strategy 基類     │  wingchun/strategy.py
         │  _internal_cb()    │  (框架自動處理 ctypes)
         └─────────┬──────────┘
                   ↑ (4) 函數指針回調
         ┌─────────┴─────────┐
         │  SignalSender      │  hf-live/_comm/
         │  ::Send()          │  (框架提供的發送器)
         └─────────┬──────────┘
                   ↑ (3) 一行代碼發送
         ┌─────────┴─────────┐
         │  MyFactorEngine    │  hf-live/app_live/
         │  OnDepth()         │  🎯 因子大師只寫這個
         └─────────┬──────────┘
                   ↑ (2) 數據分發
         ┌─────────┴─────────┐
         │  Adapter           │  adapter/adapter.cpp
         │  signal_on_data()  │  (框架自動分發)
         └─────────┬──────────┘
                   ↑ (1) 市場數據
         ┌─────────┴─────────┐
         │  runner.cpp        │  events_ 事件流
         │  on_depth()        │  (框架自動轉發)
         └────────────────────┘
```

**用戶只需關注兩個點**:
- 🎯 因子大師: `MyFactorEntry::OnDepth()` - 專注因子計算
- 🎯 策略大師: `Strategy::on_factor()` - 專注交易邏輯

**框架自動處理**:
- ✅ .so 加載與初始化
- ✅ 回調函數註冊
- ✅ 數據類型轉換
- ✅ Python/C++ 邊界管理
- ✅ 因子收集與統一發送 (`Engine::OnDepth()` + `SignalSender::Send()`)

#### 關鍵步驟拆解

**步驟 0: 初始化 (框架自動處理)**

```python
# strategies/factor_strategy/run.py
from kungfu.wingchun import Strategy

class MyStrategy(Strategy):
    """策略大師只需繼承 Strategy,框架自動處理 hf-live 加載與回調註冊"""

    def on_factor(self, context, symbol, timestamp, values):
        """
        因子回調 - 與 on_depth 同等地位,框架自動調用

        Args:
            context: 策略上下文 (同 on_depth)
            symbol: str, 標的代碼 (如 "btc_usdt")
            timestamp: int64, 時間戳 (納秒)
            values: List[float], 因子值列表
        """
        if values[0] > 0.5:
            context.insert_order(...)
```

**框架內部實現** (策略大師無需關心):

**C++ 端 (pybind11 綁定)**:
```cpp
// core/cpp/wingchun/src/bindings/strategy_bind.cpp
class PyStrategy : public Strategy {
    void on_factor(Context* context, const char* symbol, int64_t timestamp,
                   const double* values, int count) override {
        py::list py_values;
        for (int i = 0; i < count; ++i) {
            py_values.append(values[i]);
        }
        PYBIND11_OVERRIDE(void, Strategy, on_factor, context,
                         std::string(symbol), timestamp, py_values);
    }
};
```

**Python 端 (Strategy 基類)**:
```python
# core/python/kungfu/wingchun/strategy.py
class Strategy:
    def on_factor(self, context, symbol, timestamp, values):
        """用戶可覆寫的回調 (默認空實現,與 on_depth 同等地位)"""
        pass
```

**詳細實現**: 見 [prd_hf-live.07-implementation.md §1.2](prd_hf-live.07-implementation.md)

**核心流程** (概念層):

```cpp
// 步驟 1: Godzilla runner.cpp 零拷貝轉發
events_ | is(msg::type::Depth) | $([&](event_ptr event) {
    signal_on_data(handle, 101, event->data_address());  // void* 零拷貝
});

// 步驟 2: hf-live adapter 分發到因子計算
extern "C" void signal_on_data(void* handle, int type, const void* data) {
    engine->OnDepth(static_cast<const Depth*>(data));
}
```

**完整實現代碼**: 見 [prd_hf-live.07-implementation.md §1.1](prd_hf-live.07-implementation.md) (包含 dlopen、函數指針、錯誤處理)

**步驟 3: 因子計算與統一發送** (ref 風格架構)

**完整架構與代碼**: 見 [prd_hf-live.07-implementation.md §2](prd_hf-live.07-implementation.md) (實現代碼) 和 [prd_hf-live.03-workflow.md §3.2](prd_hf-live.03-workflow.md) (因子大師工作流)

**核心設計**:
- 因子模塊: 僅計算,不負責發送 (因子大師編寫)
- Engine: 統一收集與發送 (框架代碼)
- SignalSender: 統一發送接口

---

**步驟 4: Python 策略接收因子**

**詳細實現**: 見 [prd_hf-live.07-implementation.md §1.2](prd_hf-live.07-implementation.md) (pybind11 綁定)

**用戶視角** (策略大師):
```python
# strategies/my_strategy/run.py
class MyStrategy(Strategy):
    def on_factor(self, context, symbol, timestamp, values):
        """因子回調 - 與 on_depth 同等地位"""
        if values[0] > 0.5:
            context.insert_order(...)
```

---

### 6.4 數據結構共享策略 (Bundled Header 方案)

**核心問題**: hf-live submodule 如何獲知 Godzilla 數據結構定義?

#### 最終方案: 直接複製 (Bundled Header)

**完整決策理由與方案演進**: 見 [prd_hf-live.02-data-structure-sharing.md](prd_hf-live.02-data-structure-sharing.md)

**核心思想**:
- market_data_types.h 直接複製到 hf-live/include/ (一次性操作)
- 變動頻率 < 1次/年,手動同步成本 < 10分鐘/年
- 場景 A/B 零配置,獨立編譯場景下無需額外設置

**使用示例**:

```cpp
// hf-live/factors/my_factors/factor_entry.cpp
#include "market_data_types.h"  // 直接 include,零配置

class MyFactorEntry {
    void OnDepth(const Depth* depth) {
        factors_[0] = (depth->bid_price[0] - depth->ask_price[0]) / ...;
    }
};
```

**零拷貝保證**:

```cpp
// 編譯時: 兩邊結構大小一致 (使用同一個 header)
sizeof(Depth) = 336 bytes  // godzilla + hf-live 完全一致

// 運行時: void* 零拷貝轉型 (0ns)
const Depth* depth = static_cast<const Depth*>(data);
```

### 6.5 因子大師獨立開發能力

**完整工作流**: 見 [prd_hf-live.03-workflow.md §3](prd_hf-live.03-workflow.md)

**核心問題**: 因子大師是否需要理解 Godzilla 平台實現?

**解答**: ✅ 完全獨立開發,零依賴

**關鍵能力**:
- ✅ 需要知道: `Depth`/`Trade` 數據結構, `OnDepth()` API, `make` 編譯
- ❌ 不需要知道: Godzilla runner.cpp, Strategy 基類, pm2 配置, RxCpp

**獨立開發場景**:
- 場景 A: 在 godzilla-evan/hf-live 內開發 → `make` 零配置
- 場景 B: 獨立 clone hf-live 倉庫 → `make` 零配置 (header 已 bundle)

**協作模型**: 因子大師交付 `.so` → 策略大師 `pm2 restart` 熱更新

---

## 七、版本管理策略

### 7.1 Submodule 方案

**完整 Git Submodule 配置**: 見 [prd_hf-live.04-project-config.md §2](prd_hf-live.04-project-config.md)

**核心策略**:
- hf-live 作為 private submodule 添加到 godzilla-evan
- .gitignore 排除源碼/因子,僅跟蹤 commit hash
- 可選上傳編譯好的 libsignal.so 二進制文件

### 7.2 CI/CD 流程

**完整 CI/CD pipeline 設計**: 見 [prd_hf-live.08-build-deploy.md §2](prd_hf-live.08-build-deploy.md)

**關鍵流程**:
- hf-live 倉庫: 自動構建 → 驗證二進制 → 上傳 artifact
- godzilla-evan 倉庫: Submodule 更新 → 集成測試
- 版本發佈: Git tag → 自動創建 GitHub Release

---

## 八、後續分步文檔規劃

本文檔定義核心精神,具體實現將在以下文檔展開:

1. **✅ prd_hf-live.01-data-mapping.md** (已完成)
   - Godzilla 所有公開市場數據結構羅列
   - 與 ref 數據結構字段級對照
   - 差異點標註

2. **✅ prd_hf-live.02-data-structure-sharing.md** (v4.0-final, 已完成)
   - 🔥 核心決策: 直接複製 market_data_types.h 到 hf-live 倉庫
   - 放棄 symlink 方案 (獨立編譯場景下 symlink 斷裂)
   - Bundled header 方案: 手動同步 (< 1次/年)
   - 版本化管理: market_data_types.VERSION 追蹤依賴

3. **✅ prd_hf-live.03-workflow.md** (v2.0, 已完成)
   - 三大角色工作流: 因子大師、模型大師、策略大師
   - 場景 A (一人大師): godzilla-evan 內開發
   - 場景 B (獨立編譯): hf-live 獨立 clone 編譯
   - Git Submodule 管理策略

4. **✅ prd_hf-live.04-project-config.md** (v2.0, 已完成)
   - Git Submodule 配置詳解
   - CMakeLists.txt 配置 (極簡版: 直接 include hf-live/include/)
   - 依賴管理: 無外部依賴 (header 已 bundle)

5. **✅ prd_hf-live.05-code-reuse-plan-v2.md** (已完成)
   - ref 代碼複用策略
   - 因子框架架構設計
   - 模型推理框架設計

6. **✅ prd_hf-live.06-c-api-detail.md** (v1.0, 已完成)
   - 🔥 Linus 極簡原則: 4 個函數完成所有任務
   - 完整 C API 簽名 (signal_create/destroy/register_callback/on_data)
   - 錯誤處理機制 (Unix 風格: NULL/-1 + stderr)
   - 線程安全設計 (Lock-free SPMC queue)
   - ABI 穩定性保證 (opaque void* handle)
   - 零拷貝性能 (<10ns)
   - Python ctypes 綁定範例

7. **✅ prd_hf-live.07-implementation.md** (v1.1, 已完成)
   - 🔥 配置化 .so 路徑 (signal_library_path)
   - 🔥 pybind 層 on_factor 綁定 (與 on_depth 同等地位)
   - 🔥 ref 代碼完整複製說明 (非引用)
   - runner.cpp 集成點詳細實現
   - adapter/factor engine 完整代碼
   - SignalSender 統一發送機制
   - 因子模塊與 Engine 協作模式

8. **✅ prd_hf-live.08-build-deploy.md** (v1.0, 已完成)
   - 🔥 構建優化 (Release 模式、LTO、CPU 指令集)
   - 🔥 完整 CI/CD pipeline (GitHub Actions workflow)
   - 🔥 灰度發佈與回滾策略
   - 🔥 監控與故障排查手冊 (3 個常見問題)
   - 🔥 版本發佈 checklist

---

## 九、常見問題 (FAQ)

### Q1: adapter 寫在哪個項目?
**A**: hf-live 項目。Godzilla 僅依賴編譯後的 .so 文件。

### Q2: 如何保證雙方數據結構一致?
**A**: 通過版本化 header 快照 (market_data_types.h)
- ✅ 編譯時保證一致性 (相同 header)
- ✅ 極低維護成本 (< 1次/年手動同步)
- ✅ 版本追蹤 (market_data_types.VERSION)
- ✅ 錯誤安全 (編譯失敗 > 運行崩潰)

### Q3: 性能開銷?
**A**: <10ns (零拷貝,僅指針轉型)
- 版本化 header 保證結構定義完全一致
- void* 直接轉型為 const Depth* 無內存拷貝

### Q4: 支持多交易所嗎?
**A**: ✅ 是。通過統一數據結構 + 運行時字段區分
- 所有交易所使用相同的 Depth/Trade 結構
- 通過 `exchange_id` 字段區分交易所來源
- hf-live 可在運行時判斷交易所並執行特定邏輯
- 新增交易所無需修改 enum 或 API 簽名

### Q5: 如何調試?
**A**:
- HF-Live 獨立測試: `./build/test_standalone`
- Godzilla 集成測試: `gdb --args python dev_run.py`

---

## 十、總結

### 核心價值鏈

```
因子大師                策略大師
   ↓                      ↓
開發因子 (Godzilla 接口)   編寫策略 (on_factor)
   ↓                      ↓
編譯 .so              加載 .so
   ↓                      ↓
    ← 數據流 (C ABI) →
   ↓                      ↓
因子計算              交易決策
```

### 關鍵數字

- **代碼複用率**: 95% (從 ref 項目)
- **新增代碼量**: ~150 行 (adapter)
- **性能開銷**: <10ns (零拷貝)
- **維護成本**: < 10分鐘/年 (手動同步 header)

### 下一步行動

1. ✅ 閱讀本文檔 - 理解核心精神
2. ✅ 數據結構映射 - 確認基於 Godzilla 數據結構
3. ✅ 數據結構共享 - Bundled header 方案確定
4. ✅ 設計 C API - Linus 極簡 4 函數設計
5. ⏭️ 實現 adapter - 編寫適配層代碼
6. ⏭️ 集成測試 - 端到端驗證

---

**文檔版本**: v0.6-final
**最後更新**: 2025-12-04
**核心改進**:
- 數據結構共享從 symlink 改為 bundled header (獨立編譯零配置)
- C API 設計完成 (Linus 極簡 4 函數)
- 文檔規劃更新 (06 節已完成)
**下一個文檔**: `prd_hf-live.07-implementation.md`
