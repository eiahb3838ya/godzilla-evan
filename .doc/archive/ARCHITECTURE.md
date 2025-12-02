---
title: System Architecture
updated_at: 2025-12-01
owner: core-dev
lang: en
tokens_estimate: 1500
layer: archive
tags: [architecture, overview, high-level]
purpose: "High-level system architecture overview - detailed docs in modules/"
---

# System Architecture

**技術棧**: C++17 (核心) + Python 3.8+ (策略) + pybind11 (綁定)  
**性能指標**: 納秒級時間精度, <1μs 事件寫入, 1M+ events/sec 吞吐量

**詳細技術文檔**: 見 [`..modules/`](../modules/) 目錄

---

## 三層架構

```
┌─────────────────────────────────────┐
│  Python Strategies (策略層)         │
│  - 用戶交易邏輯                      │
│  - 策略回調 (on_depth, on_order)    │
└──────────────┬──────────────────────┘
               │ pybind11
┌──────────────▼──────────────────────┐
│  Wingchun (交易引擎層)               │
│  - Strategy Runner (策略執行)       │
│  - Broker (訂單路由)                │
│  - Book (持倉追蹤)                  │
│  - Gateway (交易所連接)             │
└──────────────┬──────────────────────┘
               │ Event API
┌──────────────▼──────────────────────┐
│  Yijinjing (事件溯源層)             │
│  - Journal (append-only log)       │
│  - Reader/Writer (zero-copy I/O)   │
│  - Time System (nanosecond)        │
└──────────────┬──────────────────────┘
               │ Memory-mapped files
┌──────────────▼──────────────────────┐
│  OS / Hardware                      │
└─────────────────────────────────────┘
```

---

## Yijinjing (易筋經) - 事件溯源

**核心概念**: Event Sourcing + Memory-Mapped Journal

### 三層儲存設計
```
Journal (API 層)
   ↓ 管理
Page (1-128 MB 記憶體映射檔案)
   ↓ 包含
Frame (48-byte header + data)
```

### Frame 結構
```cpp
struct frame_header {  // 48 bytes
    uint32_t length;         // 總長度
    int64_t gen_time;        // 生成時間 (nanoseconds)
    int64_t trigger_time;    // 觸發時間 (延遲追蹤)
    int32_t msg_type;        // 訊息類型 ID
    uint32_t source;         // 來源 location UID
    uint32_t dest;           // 目標 location UID
};
```

**Zero-Copy 設計**: Frame 是指向 mmap 區域的指標,無資料複製

**詳細文檔**: [`modules/yijinjing.md`](../modules/yijinjing.md)

---

## Wingchun (詠春) - 交易引擎

**核心概念**: Actor-based Trading Framework

### 四個核心組件

1. **Strategy Runner** - 策略執行環境
   - 管理策略生命週期
   - 路由事件到回調函數
   - 提供 Context API

2. **Broker** - 訂單路由與管理
   - 訂單狀態機 (Pending → Submitted → Filled)
   - 路由到正確的 Gateway
   - 訂單追蹤與驗證

3. **Book** - 持倉與帳務
   - 實時持倉追蹤
   - PnL 計算
   - 資金管理

4. **Gateway** - 交易所介面
   - MarketData: WebSocket 訂閱
   - Trader: REST API 下單
   - 抽象介面 (可擴展)

### 事件流
```
MD Gateway → Journal → Strategy → Journal → TD Gateway
     ↓                    ↓                      ↓
   Depth              on_depth()              Order
```

**詳細文檔**: 
- [`modules/wingchun.md`](../modules/wingchun.md)
- [`modules/event_flow.md`](../modules/event_flow.md)

---

## Python/C++ 綁定

**技術**: pybind11 (自動型別轉換 + GIL 管理)

### 綁定層次
```python
# Python 策略
from kungfu.wingchun import Strategy, Side, OrderStatus

class MyStrategy(Strategy):
    def on_depth(self, context, depth):  # ← C++ Depth 物件
        context.insert_order(...)         # ← 呼叫 C++ Context API
```

**關鍵特性**:
- **Type Conversion**: C++ `Order` ↔ Python `Order` (自動)
- **GIL 釋放**: I/O 操作時釋放 GIL
- **Memory Safety**: 物件生命週期由 C++ 管理

**詳細文檔**: [`modules/python_bindings.md`](../modules/python_bindings.md)

---

## Binance Extension

**實作**: REST API (下單/查詢) + WebSocket (market data)

### 市場支援
- **Spot** (現貨): `/api/v3/` endpoints
- **Futures** (合約): `/fapi/v1/` endpoints  
- **Toggle**: 執行時可切換 (`enable_spot`, `enable_futures`)

### 配置
- **Testnet/Mainnet**: 編譯時決定 (`TESTNET` flag)
- **API Keys**: 從資料庫載入,不寫入程式碼

**詳細文檔**: 
- [`modules/binance_extension.md`](../modules/binance_extension.md)
- [`adr/004-binance-market-toggle.md`](../adr/004-binance-market-toggle.md)

---

## 資料結構 (msg.h)

**Location**: `core/cpp/wingchun/include/kungfu/wingchun/msg.h`

### 核心結構

| 結構體 | 行號 | 用途 | 關鍵不變量 |
|--------|------|------|-----------|
| **Order** | 666-730 | 訂單狀態機 | `volume_traded ≤ volume` |
| **Depth** | 242-302 | 市場深度 (10 檔) | `bid_price[0] > bid_price[1]` |
| **Position** | 1000-1071 | 持倉追蹤 | `long_tot = long_yd + long_td` |
| **Asset** | 947-998 | 資金狀態 | `avail ≤ total` |

**詳細文檔**: 
- [`contracts/order_object_contract.md`](../contracts/order_object_contract.md)
- [`contracts/depth_object_contract.md`](../contracts/depth_object_contract.md)
- [`CODE_INDEX.md`](../CODE_INDEX.md)

---

## 配置系統

### 配置檔位置
```
~/.config/kungfu/app/runtime/config/
├── md/
│   └── binance/config.json          # Market Data 配置
└── td/
    └── binance/<account>.json       # Trading 配置 (API keys)
```

### 危險配置項 (絕不提交到 Git)
- `access_key` - API 金鑰
- `secret_key` - API 密鑰  
- `passphrase` - API 密碼短語

**詳細文檔**: [`config/CONFIG_REFERENCE.md`](../config/CONFIG_REFERENCE.md)

---

## 效能特性

### Yijinjing
- **Event Write**: <1μs (memory-mapped)
- **Throughput**: 1M+ events/sec
- **Zero-Copy**: Frame 直接指向 mmap 區域

### Wingchun
- **Callback Latency**: ~50-200μs (Python → C++ → Python)
- **Order Routing**: <100μs
- **Single-threaded**: 策略回調循序執行 (<1ms/callback)

### Binance Extension
- **REST API**: ~10-50ms (依網路)
- **WebSocket**: ~1-5ms 延遲
- **Reconnect**: 自動重連 (exponential backoff)

---

## 開發工作流

### 策略開發
1. 繼承 `Strategy` 類別
2. 實作回調: `pre_start()`, `on_depth()`, `on_order()`
3. 使用 Context API: `add_account()`, `subscribe()`, `insert_order()`
4. 測試: 先測試網,再實盤

### 新增交易所
1. 繼承 `MarketData` + `Trader`
2. 實作介面: `subscribe()`, `insert_order()`, `cancel_order()`
3. 註冊到 `EXTENSION_REGISTRY_MD/TD`
4. 創建配置契約文檔

**詳細文檔**: 
- [`modules/strategy_framework.md`](../modules/strategy_framework.md)
- [`modules/wingchun.md`](../modules/wingchun.md#新增交易所)

---

## 常見陷阱

1. **Depth 索引**: `bid_price[0]` 是**最佳買價**(最高),不是最差
2. **Order 狀態**: `ex_order_id` 在 `status=Submitted` 後才有值
3. **Symbol 格式**: 必須是 `btc_usdt` (小寫+底線),不是 `BTCUSDT`
4. **Account 命名**: PM2 用 `gz_user1`,資料庫是 `binance_gz_user1`
5. **Journal 是 Append-only**: 無法刪除事件,只能replay

**詳細除錯**: [`operations/debugging_guide.md`](../operations/debugging_guide.md)

---

## 檔案統計

**C++ 核心** (~13K 行):
- `yijinjing/`: 7.3K 行 (event sourcing)
- `wingchun/`: 5.8K 行 (trading framework)

**Extensions** (~2K 行):
- `binance/`: ~2K 行 (REST + WebSocket)

**Python 層** (~3K 行):
- `command/`: CLI 工具
- `wingchun/`: Strategy + bindings

**關鍵檔案**:
- `msg.h`: 1,085 行 (資料結構定義)
- `runner.cpp`: ~200 行 (策略執行引擎)
- `pybind_wingchun.cpp`: ~800 行 (Python 綁定)

---

## 延伸閱讀

### 核心概念
- [`modules/yijinjing.md`](../modules/yijinjing.md) - 事件溯源機制
- [`modules/wingchun.md`](../modules/wingchun.md) - 交易引擎架構
- [`modules/event_flow.md`](../modules/event_flow.md) - 完整事件流

### 開發指南
- [`modules/strategy_framework.md`](../modules/strategy_framework.md) - 策略開發
- [`contracts/strategy_context_api.md`](../contracts/strategy_context_api.md) - API 參考
- [`operations/debugging_guide.md`](../operations/debugging_guide.md) - 除錯指南

### 架構決策
- [`adr/001-docker.md`](../adr/001-docker.md) - 為何用 Docker
- [`adr/004-binance-market-toggle.md`](../adr/004-binance-market-toggle.md) - 市場切換設計

---

**最後更新**: 2025-12-01  
**預估 Token**: ~1500
