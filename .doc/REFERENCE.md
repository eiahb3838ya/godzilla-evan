# .doc 文檔系統參考

本文件提供 `.doc/` 目錄的快速概覽和核心摘要,詳細導航見 [NAVIGATION.md](NAVIGATION.md)。

---

## 目錄結構

```
.doc/
├── NAVIGATION.md     # 導航系統 (任務導向、關鍵字索引、依賴圖)
├── CODE_INDEX.md     # 程式碼錨點索引 (檔案行號統一管理)
├── REFERENCE.md      # 本文件 - 快速概覽
│
├── modules/          # 核心模組說明 (9 個文檔)
│   ├── yijinjing.md           # 事件溯源機制
│   ├── wingchun.md            # 交易引擎架構
│   ├── strategy_framework.md # 策略開發框架
│   ├── binance_extension.md  # Binance 實作
│   ├── ledger_system.md       # 帳務系統
│   ├── python_bindings.md     # Python/C++ 綁定
│   ├── event_flow.md          # 事件流程圖
│   ├── order_lifecycle_flow.md   # 訂單狀態機
│   ├── strategy_lifecycle_flow.md # 策略生命週期
│   └── trading_flow.md        # 交易完整流程
│
├── contracts/        # API 契約 (4 個文檔)
│   ├── strategy_context_api.md    # Context API 完整參考
│   ├── order_object_contract.md   # Order 物件契約
│   ├── depth_object_contract.md   # Depth 物件契約
│   └── binance_config_contract.md # Binance 配置契約
│
├── operations/       # 操作指南 (4 個文檔)
│   ├── QUICK_START.md         # 快速啟動指令集
│   ├── pm2_startup_guide.md   # PM2 完整操作指南
│   ├── cli_operations_guide.md # CLI 工具詳解
│   ├── debugging_guide.md     # 除錯診斷流程
│   └── DEBUGGING.md           # 除錯完整手冊 (詳細版)
│
├── config/           # 配置說明 (4 個文檔)
│   ├── config_usage_map.md         # 配置檔使用地圖
│   ├── dangerous_keys.md           # 密鑰安全指南
│   ├── account_naming_convention.md # 帳戶命名規範
│   └── symbol_naming_convention.md  # 交易對命名規範
│
├── adr/              # 架構決策記錄 (4 個文檔)
│   ├── 001-docker.md              # Docker 容器化決策
│   ├── 002-wsl2.md                # WSL2 開發環境
│   ├── 003-dns.md                 # DNS 問題解決
│   └── 004-binance-market-toggle.md # Binance 市場切換
│
└── archive/          # 大文檔存檔 (6 個文檔)
    ├── TESTNET.md       # 測試網設定完整指南
    ├── INSTALL.md       # 安裝指南
    ├── HACKING.md       # 開發指南
    ├── DESIGN.md        # 設計文檔
    ├── ORIGIN.md        # 專案起源
    └── LOG_LOCATIONS.md # 日誌位置完整清單
```

---

## 核心模組摘要

### Yijinjing (易筋經) - 事件溯源
- **職責**: 事件記錄、訊息傳遞、時間旅行除錯
- **延遲**: ~50-200μs
- **關鍵概念**: Journal (append-only log)、Reader/Writer、nano time
- **詳細**: [modules/yijinjing.md](modules/yijinjing.md)

### Wingchun (詠春) - 交易引擎
- **職責**: 策略執行、訂單路由、持倉追蹤、帳務管理
- **架構**: Strategy Runner + Broker + Book + Gateway
- **回調時序**: `pre_start()` → `on_depth()` / `on_order()` / `on_trade()` → `pre_stop()`
- **詳細**: [modules/wingchun.md](modules/wingchun.md)

### Binance Extension - 交易所連接器
- **職責**: REST API + WebSocket 實作
- **支援市場**: Spot (現貨) + Futures (合約)
- **配置**: Testnet/Mainnet 編譯時決定,`enable_spot` / `enable_futures` 執行時切換
- **詳細**: [modules/binance_extension.md](modules/binance_extension.md)

---

## API 契約摘要

### Context API
**完整參考**: [contracts/strategy_context_api.md](contracts/strategy_context_api.md)

```python
# 帳戶管理
context.add_account(source, account)
context.list_accounts()

# 訂閱市場數據
context.subscribe(source, symbols, instrument_type, exchange)

# 下單操作
context.insert_order(symbol, side, price, volume, ...)
context.cancel_order(order_id)

# 狀態管理
context.set_object(key, value)
context.get_object(key)
```

### Order 物件
**完整參考**: [contracts/order_object_contract.md](contracts/order_object_contract.md)

- **關鍵欄位**: `order_id`, `ex_order_id`, `status`, `volume`, `volume_traded`, `avg_price`
- **不變量**: `volume_traded ≤ volume`
- **陷阱**: `ex_order_id` 在 `status=Submitted` 後才有值

### Depth 物件
**完整參考**: [contracts/depth_object_contract.md](contracts/depth_object_contract.md)

- **結構**: `bid_price[10]`, `ask_price[10]`, `bid_volume[10]`, `ask_volume[10]`
- **不變量**: `bid_price[0] > bid_price[1]` (降序), `ask_price[0] < ask_price[1]` (升序)
- **陷阱**: `bid_price[0]` 是**最佳買價**(最高),不是最差

---

## 快速查找

### 按任務類型

| 我想... | 主要文檔 | 補充文檔 |
|--------|---------|---------|
| **開發新策略** | modules/strategy_framework.md | contracts/strategy_context_api.md |
| **除錯 Binance** | modules/binance_extension.md | config/config_usage_map.md, archive/TESTNET.md |
| **理解架構** | modules/yijinjing.md, modules/wingchun.md | modules/event_flow.md |
| **服務部署** | operations/QUICK_START.md | operations/pm2_startup_guide.md |
| **新增交易所** | modules/wingchun.md | modules/binance_extension.md (參考實作) |

**詳細導航**: 見 [NAVIGATION.md](NAVIGATION.md)

---

### 按關鍵字

| 關鍵字 | 文檔 |
|--------|------|
| Order | contracts/order_object_contract.md |
| Depth | contracts/depth_object_contract.md |
| Journal | modules/yijinjing.md |
| Context API | contracts/strategy_context_api.md |
| PM2 | operations/pm2_startup_guide.md, operations/QUICK_START.md |
| 配置 | config/config_usage_map.md |
| Binance | modules/binance_extension.md |

**完整索引**: 見 [NAVIGATION.md#關鍵字快速查找](NAVIGATION.md#二關鍵字快速查找)

---

## 文檔統計

- **總文檔數**: 36 個 .md 文件
- **總大小**: ~1.1MB
- **預估 tokens**: ~576k (全載入)
- **推薦冷啟動**: CLAUDE.md + NAVIGATION.md ≈ 800 tokens
- **一般任務**: 2-3 個文檔 ≈ 15-20k tokens

---

## 維護指南

修改程式碼後,更新相關文檔:

| 修改類型 | 需更新文檔 |
|---------|-----------|
| **資料結構** (msg.h) | contracts/*_object_contract.md + CODE_INDEX.md |
| **API** (context.cpp) | contracts/strategy_context_api.md |
| **生命週期** (runner.cpp) | modules/strategy_framework.md, modules/strategy_lifecycle_flow.md |
| **Python 綁定** (pybind_wingchun.cpp) | modules/python_bindings.md |
| **配置** | config/config_usage_map.md, contracts/binance_config_contract.md |
| **架構決策** | 新增 adr/00X-decision-name.md |

**驗證工具**:
```bash
python3 .doc/operations/scripts/verify_code_refs.py  # 檢查程式碼引用
python3 .doc/operations/scripts/check_links.py       # 檢查連結完整性
python3 .doc/operations/scripts/estimate_tokens.py   # 估算 token 數量
```
