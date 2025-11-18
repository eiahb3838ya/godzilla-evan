---
title: START (對話啟動腳本｜自動學習交易系統語境)
updated_at: 2025-11-17
owner: core-dev
lang: zh-TW
tags: [onboarding, context, ai-assistant, startup]
purpose: "AI 助手快速載入專案語境的標準流程"
---

# 說明

你是一個交易系統開發助手。當使用者在新對話中輸入「遵行 START.md」或「follow .doc/START.md」，請嚴格依以下步驟「自動學習本倉庫的語境層 .doc」，完成後回覆 **READY** 並等待下一個指令。

**專案性質**: 高頻加密貨幣交易系統 (C++ 核心 + Python 策略層)

# 執行政策（全程遵守）

- **語言**: 回覆一律採用繁體中文（zh-TW），除非用戶明確要求英文
- **來源優先序**: `.doc/` > `core/` 源碼 > `strategies/` 策略範例
- **節奏**: 先給結論/摘要，再附路徑與行號，最後補細節；避免貼整段長文
- **只讀模式**: 不要修改任何檔案；閱讀後在你的「對話記憶」中保留要點即可
- **Token 預算**: 遵守 `.doc/50_rag/token_budget.md` 的 auto-load 政策 (~113k tokens)

# 流程（一步步執行並在每步產出對應輸出）

## 1) 初始化（輸出：確認與執行計畫）

- 重述「執行政策」
- 輸出你的閱讀計畫（2-3 行）
- 確認當前工作目錄是否為專案根目錄

**預期輸出**:
```
✓ 執行政策已確認：zh-TW、.doc 優先、只讀模式、token 預算
✓ 閱讀計畫：依序載入系統概覽 → 核心模組 → API 契約 → 操作指南
✓ 工作目錄：/home/huyifan/projects/godzilla-evan
```

---

## 2) 系統概覽（輸出：3 段摘要 + 架構圖）

**必讀文件**:
- `.doc/00_index/DESIGN.md` - 系統設計理念
- `.doc/00_index/ARCHITECTURE.md` - 整體架構
- `.doc/00_index/index.yaml` - 文檔索引

**預期輸出**:
- 1 段系統摘要（100 字內）：專案目標、核心技術棧
- 1 段架構摘要（100 字內）：分層設計（Yijinjing/Wingchun/Strategy）
- 1 段開發流程摘要（80 字內）：如何開發新策略
- ASCII 架構圖（如 DESIGN.md 中所示）

---

## 3) 核心模組（輸出：6 個模組的職責與 API 要點）

**必讀文件**:
- `.doc/10_modules/yijinjing_journal.md` - 事件溯源系統（Journal）
- `.doc/10_modules/strategy_framework.md` - 策略開發框架
- `.doc/10_modules/gateway_architecture.md` - 交易所閘道器
- `.doc/10_modules/python_bindings.md` - pybind11 綁定機制
- `.doc/10_modules/ledger_system.md` - 帳戶/持倉/PnL 追蹤
- `.doc/10_modules/binance_extension.md` - Binance 交易所實作

**預期輸出** (每個模組 2-3 行):
```
1. Yijinjing Journal:
   - 職責: 事件溯源、訊息傳遞、時間旅行
   - 關鍵 API: journal.write(), reader.data<T>()
   - 延遲: ~50-200μs (事件生成 → 策略回調)

2. Strategy Framework:
   - 職責: 策略生命週期管理、回調路由
   - 關鍵回調: pre_start(), on_depth(), on_order(), on_transaction()
   - 限制: 單執行緒、回調必須 <1ms

... (依此類推)
```

---

## 4) 互動流程（輸出：2 個關鍵流程的時序摘要）

**必讀文件**:
- `.doc/20_interactions/strategy_lifecycle_flow.md` - 策略生命週期
- `.doc/20_interactions/order_lifecycle_flow.md` - 訂單生命週期

**預期輸出**:
```
策略生命週期 (12 階段):
  啟動 → on_start() → pre_start() → 事件循環設置 → post_start()
  → 【主執行：on_depth/on_order 回調】
  → 關閉訊號 → pre_stop() → on_exit() → post_stop() → 退出

訂單生命週期 (7 階段):
  insert_order() → Ledger 檢查餘額 → 凍結資金 → TD 閘道器送單
  → 交易所確認 (Submitted) → WebSocket 成交通知 → on_transaction() 回調
  延遲: Testnet ~42ms | Mainnet 同地 ~2ms
```

---

## 5) API 契約（輸出：4 個核心 API 的簽名與不變量）

**必讀文件**:
- `.doc/30_contracts/strategy_context_api.md` - Context 物件 API
- `.doc/30_contracts/order_object_contract.md` - Order 資料結構
- `.doc/30_contracts/depth_object_contract.md` - Depth 資料結構
- `.doc/30_contracts/binance_config_contract.md` - Binance 配置

**預期輸出** (每個契約列出):
- 關鍵方法/欄位 (3-5 個)
- 不變量 (2-3 條)
- 常見陷阱 (1-2 條)

**範例**:
```
Order Object:
  欄位: order_id, status, volume, volume_traded, avg_price
  不變量:
    - volume_traded <= volume
    - volume_left = volume - volume_traded
  陷阱: ex_order_id 在 Submitted 後才有值（Pending 時為空）

Depth Object:
  欄位: bid_price[10], ask_price[10], bid_volume[10], ask_volume[10]
  不變量:
    - bid_price[0] > bid_price[1] (降序)
    - ask_price[0] < ask_price[1] (升序)
  陷阱: bid_price[0] 是最佳買價（不是 worst）
```

---

## 6) 配置管理（輸出：配置檔位置與 3 條風險鍵）

**必讀文件**:
- `.doc/40_config/config_usage_map.md` - 配置檔案地圖
- `.doc/40_config/dangerous_keys.md` - 危險配置項

**預期輸出**:
```
配置檔位置:
  - Binance 閘道器: ~/.config/kungfu/app/config/td/binance/<account>.json
  - 策略配置: 通過 -c 參數傳入 JSON
  - 環境變數: KF_HOME (預設 ~/.config/kungfu/app/runtime)

危險鍵 (絕不可提交):
  1. access_key / secret_key - API 憑證
  2. CLEAR_JOURNAL=1 - 刪除歷史 Journal（僅開發用）
  3. 硬編碼的 testnet/mainnet 切換（需檢查 spot_rest_host）
```

---

## 7) 操作指南（輸出：5 條常用指令）

**必讀文件**:
- `.doc/90_operations/pm2_startup_guide.md` - PM2 流程管理
- `.doc/90_operations/cli_operations_guide.md` - CLI 指令參考
- `.doc/90_operations/DEBUGGING.md` - 除錯指南

**預期輸出**:
```
常用指令:
1. 啟動 Master:    python3 dev_run.py -l info master
2. 啟動 Ledger:    python3 dev_run.py -l info ledger
3. 啟動 MD 閘道器: python3 dev_run.py -l trace md -s binance
4. 啟動 TD 閘道器: python3 dev_run.py -l info td -s binance -a my_account
5. 執行策略:       python3 dev_run.py -l info strategy -n demo -p strategies/demo_spot.py

啟動順序: Master (等 5s) → Ledger (等 5s) → MD → TD → Strategy
優雅關閉: kill -2 <master_pid> (發送 SIGINT 給 master)
```

---

## 8) 開發快速參考（輸出：策略範例骨架）

**必讀文件**:
- `.doc/10_modules/strategy_framework.md#best-practices`
- `strategies/demo_spot.py` (實際程式碼)

**預期輸出**:
```python
# 策略開發模板
class MyStrategy(Strategy):
    def pre_start(self, context):
        """初始化：訂閱市場數據、添加帳戶"""
        context.add_account("binance", "my_account")
        context.subscribe("binance", ["btcusdt"], InstrumentType.Spot, Exchange.BINANCE)
        context.set_object("position", 0.0)  # 狀態管理

    def on_depth(self, context, depth):
        """市場數據回調：每秒 ~10-20 次"""
        if depth.bid_price[0] > self.threshold:
            context.insert_order(symbol="btcusdt", side=Side.Buy, ...)

    def on_order(self, context, order):
        """訂單狀態變化回調"""
        if order.status == OrderStatus.Filled:
            context.log().info(f"成交於 {order.avg_price}")

    def pre_stop(self, context):
        """關閉前：取消所有掛單"""
        for order_id in self.open_orders:
            context.cancel_order(...)
```

---

## 9) RAG 與 Token 預算（輸出：自動載入清單）

**必讀文件**:
- `.doc/50_rag/token_budget.md` - Token 預算政策
- `.doc/50_rag/chunking_strategy.md` - 文檔分塊策略

**預期輸出**:
```
Auto-Load 文檔 (~113k tokens):
  ✓ 00_index/ (10k) - 導航與設計
  ✓ 10_modules/ (40k) - 所有模組卡
  ✓ 30_contracts/ (50k) - 所有 API 契約
  ✓ 40_config/ (8k) - 配置文檔
  ✓ 85_memory/ (5k) - 會話狀態

Selective Load (依任務載入):
  ○ 20_interactions/ - 互動流程（開發新功能時載入）
  ○ 90_operations/ - 操作指南（部署/除錯時載入）
  ○ 95_adr/ - 架構決策（架構變更時載入）

剩餘預算: ~80k tokens (對話 + 程式碼片段)
```

---

## 10) 程式碼錨點記憶（輸出：20 個關鍵程式碼位置）

記住以下關鍵程式碼位置（格式：`file:line`）：

**策略執行**:
- `core/cpp/wingchun/src/strategy/runner.cpp:55-194` - 策略生命週期
- `core/cpp/wingchun/src/strategy/runner.cpp:66-76` - Depth 事件訂閱
- `core/cpp/wingchun/src/strategy/runner.cpp:124-141` - Order 事件路由

**訂單處理**:
- `core/cpp/wingchun/include/kungfu/wingchun/msg.h:666-730` - Order 結構定義
- `core/cpp/wingchun/include/kungfu/wingchun/msg.h:242-302` - Depth 結構定義

**交易所閘道器**:
- `core/extensions/binance/include/common.h:18-71` - Binance 配置結構
- `core/cpp/wingchun/pybind/pybind_wingchun.cpp:516-547` - Order 物件綁定
- `core/cpp/wingchun/pybind/pybind_wingchun.cpp:719-743` - Context API 綁定

**帳本系統**:
- `core/cpp/wingchun/include/kungfu/wingchun/service/ledger.h:22-109` - Ledger 服務
- `core/cpp/wingchun/include/kungfu/wingchun/book/book.h:22-49` - Book 介面
- `core/cpp/wingchun/include/kungfu/wingchun/msg.h:947-998` - Asset 結構
- `core/cpp/wingchun/include/kungfu/wingchun/msg.h:1000-1071` - Position 結構

**Journal 系統**:
- `core/cpp/yijinjing/include/kungfu/yijinjing/journal/journal.h:1-150` - Journal API
- `core/cpp/yijinjing/include/kungfu/yijinjing/journal/frame.h:1-200` - 事件幀結構

**Python 綁定**:
- `core/cpp/wingchun/pybind/pybind_wingchun.cpp:264-319` - 枚舉綁定 (Side, OrderStatus)
- `core/cpp/wingchun/pybind/pybind_wingchun.cpp:35-75` - Trampoline 類別 (虛擬方法)

**CLI 入口**:
- `core/python/kungfu/command/master.py:12-19` - Master 指令
- `core/python/kungfu/command/strategy.py:19-110` - Strategy 指令

**配置管理**:
- `core/extensions/binance/include/common.h:43-71` - 市場切換配置

**預期輸出**: 將上述 20 個錨點記入對話記憶，並確認已儲存

---

## 11) 完成（輸出：READY + 兩段總結 + 下一步詢問）

回覆 `READY` 並輸出：

**總結 1 - 我已學會的核心**:
- ✓ 系統架構：Yijinjing (Journal) → Wingchun (策略/閘道器) → Python (策略層)
- ✓ 開發流程：pre_start() → on_depth()/on_order() 回調 → insert_order() → pre_stop()
- ✓ 關鍵契約：Order 狀態機、Depth 陣列索引、Context API 30+ 方法
- ✓ 操作指令：PM2 啟動順序、CLI 參數、配置檔位置

**總結 2 - 如何檢索與引用**:
- ✓ 文檔優先：`.doc/` 契約/模組 > 原始碼
- ✓ 程式碼引用：使用 `file:line` 格式（如 `runner.cpp:55-194`）
- ✓ Token 預算：自動載入 ~113k，剩餘 ~80k 供對話使用
- ✓ 錨點記憶：20 個關鍵程式碼位置已記入

**詢問使用者下一步**:
```
🎯 語境載入完成，請選擇下一步行動：

A. 開發新策略（例如：「幫我實作網格交易策略」）
B. 除錯現有問題（例如：「為什麼我的訂單一直被拒絕？」）
C. 系統部署（例如：「如何部署到生產環境？」）
D. 架構變更（例如：「如何新增 OKX 交易所支援？」）
E. 閱讀特定文檔（例如：「詳細解釋 Ledger 系統如何計算 PnL」）

或直接描述你的需求。
```

---

# 限流與例外

- **單頁過長者**: 先列重點與程式碼位置，再視需要補充
- **檔案缺失或不可讀**: 回報檔名並跳過，不中斷整體流程
- **Token 超限**: 優先載入 30_contracts/ 和 10_modules/，延後載入 20_interactions/

---

# 快速檢核（內部完成）

執行完畢後，確認：
- [ ] 我已遵守語言/引用/來源優先序
- [ ] 我已輸出每一步要求內容
- [ ] 我已建立 20 個程式碼錨點記憶
- [ ] 我已準備好回答交易系統開發問題
- [ ] 我已理解 C++/Python 分層架構與 pybind11 綁定

---

# 特殊模式

## Memory Mode (記憶導向開發)

若使用者提供 `.doc/85_memory/*.memory.md` 檔案，表示要沿該記憶方向持續開發：

1. **讀取 Memory 檔**:
   - 輸出 3-5 行摘要（背景/目標/現況）
   - 列出待辦/研究假設/已知風險（各 3 項內）

2. **建立工作計畫** (2-4 步):
   - 範例：補充策略邏輯 → 新增單元測試 → 更新文檔 → 驗證延遲

3. **映射關聯**:
   - 找出相關的程式碼錨點（`file:line`）
   - 找出相關的契約文檔（`.doc/30_contracts/`）

4. **詢問執行**:
   - 選項 1: 立刻按計畫第 1 步執行
   - 選項 2: 調整計畫（請指出需增刪的步驟）

---

# 附錄：文檔層級說明

```
.doc/
├── 00_index/          導航入口 (DESIGN, ARCHITECTURE, index.yaml)
├── 10_modules/        模組卡 (yijinjing, strategy, ledger, gateway, bindings)
├── 20_interactions/   互動流程 (strategy_lifecycle, order_lifecycle)
├── 30_contracts/      API 契約 (context_api, order, depth, config)
├── 40_config/         配置管理 (usage_map, dangerous_keys)
├── 50_rag/            RAG 策略 (token_budget, chunking, retrieval_eval)
├── 85_memory/         會話記憶 (持續開發的上下文)
├── 90_operations/     操作指南 (pm2, cli, debugging)
└── 95_adr/            架構決策記錄 (docker, wsl2, market-toggle)
```

**Token 分配**:
- Auto-Load: ~113k (index + modules + contracts + config + memory)
- Selective: ~37k (interactions + operations + ADR)
- 剩餘對話: ~50k (程式碼片段 + 多輪對話)

---

**版本**: 2025-11-17
**維護者**: core-dev
**適用場景**: 新對話啟動、AI 助手上下文載入、開發者快速上手
