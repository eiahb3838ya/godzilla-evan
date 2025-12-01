# 程式碼索引 - 快速定位關鍵程式碼

本文件統一管理所有程式碼錨點,避免在多個文檔中重複維護。

---

## 一、資料結構定義

### msg.h - 核心資料結構

**檔案**: `core/cpp/wingchun/include/kungfu/wingchun/msg.h`

| 結構體 | 行號 | 用途 | 關鍵不變量 | 相關文檔 |
|--------|------|------|-----------|---------|
| **Order** | 666-730 | 訂單狀態機 | `volume_traded ≤ volume` | contracts/order_object_contract.md |
| **Depth** | 242-302 | 市場深度 (10 檔) | `bid_price[0] > bid_price[1]` (降序) | contracts/depth_object_contract.md |
| **Position** | 1000-1071 | 持倉追蹤 | `long_tot = long_yd + long_td` | modules/ledger_system.md |
| **Asset** | 947-998 | 資金狀態 | `avail ≤ total` | modules/ledger_system.md |
| **Trade** | 820-868 | 成交記錄 | `price > 0, volume > 0` | contracts/order_object_contract.md |
| **Quote** | 304-362 | 報價數據 | `bid_price < ask_price` | contracts/depth_object_contract.md |

**陷阱提示**:
- `Depth.bid_price[0]` 是**最佳買價**(最高),不是最差
- `Order.ex_order_id` 在 `status=Submitted` 後才有值
- `Position` 的 `yesterday` 和 `today` 欄位在 T+1 市場特別重要

---

## 二、策略執行引擎

### runner.cpp - 策略生命週期與事件路由

**檔案**: `core/cpp/wingchun/src/strategy/runner.cpp`

| 功能模組 | 行號 | 說明 | 相關文檔 |
|---------|------|------|---------|
| **生命週期管理** | 55-194 | `pre_start` → 事件循環 → `pre_stop` | modules/strategy_framework.md |
| **Depth 事件路由** | 66-76 | 市場數據推送到策略回調 | modules/event_flow.md |
| **Order 事件路由** | 124-141 | 依 `strategy_id` 過濾訂單事件 | modules/order_lifecycle_flow.md |
| **Trade 事件路由** | 143-158 | 成交回報推送 | modules/trading_flow.md |
| **Account 初始化** | 215-237 | 帳戶註冊與驗證 | modules/strategy_framework.md#pre_start |

**事件處理時序**:
```
1. on_start() → 呼叫策略的 pre_start()
2. 進入事件循環
   ├─ on_depth() → 策略的 on_depth()
   ├─ on_order() → 策略的 on_order() (若 strategy_id 匹配)
   └─ on_trade() → 策略的 on_trade()
3. on_stop() → 呼叫策略的 pre_stop()
```

---

### context.cpp - 策略 Context API 實作

**檔案**: `core/cpp/wingchun/src/strategy/context.cpp`

| API 類別 | 行號 | 主要方法 | 相關文檔 |
|---------|------|---------|---------|
| **帳戶管理** | 85-120 | `add_account()`, `list_accounts()` | contracts/strategy_context_api.md#account |
| **訂閱管理** | 250-315 | `subscribe()`, `unsubscribe()` | contracts/strategy_context_api.md#subscribe |
| **下單操作** | 350-410 | `insert_order()`, `cancel_order()` | contracts/strategy_context_api.md#order |
| **狀態管理** | 473-520 | `set_object()`, `get_object()` | contracts/strategy_context_api.md#state |
| **日誌系統** | 540-580 | `log().info()`, `log().error()` | modules/strategy_framework.md#logging |

**重要**: 所有 Context API 都是**執行緒安全的**,但策略回調本身是**單執行緒**執行。

---

## 三、Python 綁定層

### pybind_wingchun.cpp - Python/C++ 綁定

**檔案**: `core/cpp/wingchun/pybind/pybind_wingchun.cpp`

| 綁定類型 | 行號 | 對應 Python 類/枚舉 | 相關文檔 |
|---------|------|-------------------|---------|
| **枚舉綁定** | 264-319 | `Side`, `OrderStatus`, `InstrumentType`, `Exchange` | modules/python_bindings.md#enums |
| **Order 綁定** | 516-547 | `Order` 類 (唯讀屬性) | contracts/order_object_contract.md |
| **Depth 綁定** | 548-580 | `Depth` 類 (陣列屬性) | contracts/depth_object_contract.md |
| **Context API 綁定** | 719-743 | `Strategy.context` 的所有方法 | contracts/strategy_context_api.md |
| **Position 綁定** | 620-658 | `Position` 類 | modules/ledger_system.md |

**Python 範例**:
```python
# 這些類別是由 pybind11 自動生成
from kungfu.wingchun import Side, OrderStatus, InstrumentType

# Order 物件是唯讀的 (在 C++ 端管理)
def on_order(self, context, order):
    assert isinstance(order.status, OrderStatus)
    print(f"Order {order.order_id} status: {order.status}")
```

---

## 四、交易所擴展

### Binance Extension - 交易所介面實作

**檔案**: `core/extensions/binance/`

| 模組 | 檔案 | 關鍵行號 | 說明 | 相關文檔 |
|------|------|---------|------|---------|
| **配置結構** | `include/common.h` | 18-71 | `BinanceCommonConfig` 定義 | contracts/binance_config_contract.md |
| **Testnet/Mainnet** | `include/common.h` | 硬編碼 | 編譯時決定,無法執行時切換 | archive/TESTNET.md |
| **MarketData 實作** | `src/marketdata_binance.cpp` | 全檔案 | WebSocket 訂閱與解析 | modules/binance_extension.md#marketdata |
| **Trader 實作** | `src/trader_binance.cpp` | 全檔案 | REST API 下單與查詢 | modules/binance_extension.md#trader |
| **市場切換邏輯** | `src/trader_binance.cpp` | 98-150 (參考) | `enable_spot` / `enable_futures` 實作 | adr/004-binance-market-toggle.md |

**新增交易所檢查清單**:
1. 繼承 `MarketData` 和 `Trader` 抽象類別
2. 實作 `subscribe()`, `insert_order()`, `cancel_order()` 等介面
3. 註冊到 `EXTENSION_REGISTRY_MD` 和 `EXTENSION_REGISTRY_TD`
4. 創建配置契約文檔 (參考 binance_config_contract.md)

---

## 五、配置系統

### 配置檔位置

| 配置類型 | 路徑 | 格式 | 範例 | 相關文檔 |
|---------|------|------|------|---------|
| **MD 配置** | `~/.config/kungfu/app/runtime/config/md/<source>/config.json` | JSON | `{"url": "ws://...", "timeout": 5000}` | config/CONFIG_REFERENCE.md |
| **TD 配置** | `~/.config/kungfu/app/runtime/config/td/<source>/<account>.json` | JSON | `{"access_key": "...", "secret_key": "..."}` | config/CONFIG_REFERENCE.md#part-2-security-guidelines |
| **策略配置** | `strategies/<name>/config.json` | JSON | `{"threshold": 100, "volume": 1.0}` | modules/strategy_framework.md#config |

**危險配置項** (絕不提交到 Git):
- `access_key` - API 金鑰
- `secret_key` - API 密鑰
- `passphrase` - API 密碼短語

**檢查方法**:
```bash
# 確認 .gitignore 包含配置目錄
grep "config/td" .gitignore

# 檢查是否誤提交金鑰
git log -S "access_key" --all
```

---

## 六、常用工具腳本

### 診斷與驗證腳本

**路徑**: `.doc/operations/scripts/`

| 腳本 | 用途 | 執行方式 |
|------|------|---------|
| `diagnostic.sh` | 完整環境診斷 | `bash .doc/operations/scripts/diagnostic.sh` |
| `verify-commands.sh` | 驗證所有 CLI 指令可用性 | `bash .doc/operations/scripts/verify-commands.sh` |
| `setup-docker-dns.sh` | 修復 Docker DNS 問題 | `bash .doc/operations/scripts/setup-docker-dns.sh` |

**Python 工具**:
```bash
# 驗證文檔中的程式碼引用 (file:line) 是否正確
python3 .doc/operations/scripts/verify_code_refs.py

# 檢查 markdown 連結完整性
python3 .doc/operations/scripts/check_links.py

# 估算文檔 token 數量
python3 .doc/operations/scripts/estimate_tokens.py
```

---

## 七、快速查詢表

### 找不到某個功能的實作?

| 我想找... | 查這裡 |
|----------|--------|
| **Order 的欄位定義** | msg.h:666-730 |
| **策略如何接收 Depth** | runner.cpp:66-76 |
| **Context API 如何下單** | context.cpp:350-410 |
| **Python 如何存取 Order** | pybind_wingchun.cpp:516-547 |
| **Binance 配置格式** | binance/include/common.h:18-71 |
| **配置檔存放位置** | config/CONFIG_REFERENCE.md |

### 修改程式碼後要更新哪些文檔?

| 修改了... | 需更新文檔 |
|----------|-----------|
| **msg.h 資料結構** | contracts/*_object_contract.md + CODE_INDEX.md (本文件) |
| **context.cpp API** | contracts/strategy_context_api.md |
| **runner.cpp 生命週期** | modules/strategy_framework.md + modules/strategy_lifecycle_flow.md |
| **pybind 綁定** | modules/python_bindings.md |
| **binance 配置** | contracts/binance_config_contract.md + config/CONFIG_REFERENCE.md |

---

## 八、維護指南

### 更新本文件的時機

1. **新增/修改資料結構** → 更新「資料結構定義」區塊
2. **重構策略引擎** → 更新「策略執行引擎」區塊
3. **新增 Python API** → 更新「Python 綁定層」區塊
4. **新增交易所** → 在「交易所擴展」新增一行
5. **變更配置路徑** → 更新「配置系統」區塊

### 驗證程式碼引用正確性

```bash
# 自動驗證所有 file:line 引用
python3 .doc/operations/scripts/verify_code_refs.py

# 手動檢查某個檔案
head -n 730 core/cpp/wingchun/include/kungfu/wingchun/msg.h | tail -n 65
```

### 同步更新相關文檔

每次更新 CODE_INDEX.md 時,檢查以下文檔是否需同步:
- [ ] contracts/*_object_contract.md (資料結構變更)
- [ ] modules/python_bindings.md (綁定變更)
- [ ] NAVIGATION.md (新增交叉引用,更新快速入口)
