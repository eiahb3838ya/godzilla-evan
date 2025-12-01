---
title: UPDATE (程式碼改動後的文檔同步｜只針對本輪改動)
updated_at: 2025-11-17
owner: core-dev
lang: zh-TW
tags: [documentation, update, sync, maintenance]
purpose: "自動偵測程式碼改動並同步更新相關文檔"
---

# 目的

當使用者在完成「某一輪功能/程式改動」後，在新對話輸入「遵行 UPDATE.md」或「follow .doc/UPDATE.md」，你應以「本輪改動」為唯一事實來源，先總結改動，再據此更新 `.doc/`。不處理其他場景。

**適用場景**:
- ✅ 新增交易策略功能
- ✅ 修改 API 介面
- ✅ 新增交易所支援
- ✅ 修改配置結構
- ✅ 修復 Bug 並更新文檔
- ❌ 純文檔修正（直接編輯即可）
- ❌ 歷史改動回溯（需另行處理）

---

# 執行政策（全程遵守）

- **語言**: 回覆一律使用繁體中文（zh-TW）
- **範圍**: 僅修改 `.doc/`，不變動程式碼（除非使用者另行授權）
- **真實來源**: 以「本輪改動集」為準（來自本次對話的實作/描述或提供的 diff）；不得擴散到未涉及的元件
- **引用格式**: 優先使用 `file:line` 格式，其次相對路徑；避免絕對路徑
- **來源優先序**: 程式碼實作 > `.doc/` 現有文檔 > 使用者描述
- **Token 預算**: 遵守 `.doc/50_rag/token_budget.md` 的限制

---

# 更新流程（逐步執行並輸出所需內容）

## 1) 初始化（輸出：執行計畫）

**步驟**:
1. 重述執行政策（簡述）
2. 明確「本輪改動」來源：
   - 若使用者已提供改動摘要/檔案清單/commit 訊息，直接採用
   - 否則請求使用者提供最小資訊：
     - 功能名稱/目的
     - 涉及檔案或模組
     - 主要變更內容
   - （可選）若有 git 訪問，使用 `git diff --name-only` 或 `git log -1 --stat` 輔助，但以使用者描述為準

3. 輸出更新計畫：
   - 批次 0: 彙整改動 → 映射需更新的文檔
   - 批次 1: 更新契約層（30_contracts/）
   - 批次 2: 更新模組層（10_modules/）
   - 批次 3: 更新互動層（20_interactions/）
   - 批次 4: 更新配置層（40_config/）
   - 批次 5: 更新操作層（90_operations/）
   - 批次 6: 更新索引（00_index/）

**預期輸出**:
```
✓ 執行政策：zh-TW、僅更新 .doc/、以本輪改動為準
✓ 改動來源：[commit hash / 使用者描述 / git diff]
✓ 更新計畫：6 批次，預估影響 X 個文檔
```

---

## 2) 改動彙整（輸出：改動總結與影響面）

**步驟**:
1. 產出「本輪改動總結」（100-200 字）：
   - 功能名稱
   - 目的/動機
   - 涉及模組/檔案
   - 核心 API/資料結構變更
   - 配置項變更（如有）
   - 是否新增程式碼錨點

2. 影響映射：依改動對應到需更新的文檔區塊

**預期輸出**:
```
【改動總結】
功能: 新增 OKX 交易所支援
目的: 支援多交易所交易，降低單一交易所風險
涉及檔案:
  - core/extensions/okx/ (新增)
  - core/python/extensions/__init__.py (修改)
  - .doc/10_modules/gateway_architecture.md (需更新)

核心變更:
  - 新增 MarketDataOKX 類別 (okx/marketdata_okx.cpp)
  - 新增 TraderOKX 類別 (okx/trader_okx.cpp)
  - 新增 OKX 配置結構 (okx/include/common.h)
  - 註冊 OKX 到 EXTENSION_REGISTRY_MD/TD

配置項:
  - 新增 ~/.config/kungfu/app/config/md/okx/config.json
  - 新增 ~/.config/kungfu/app/config/td/okx/<account>.json

【影響映射】
需更新文檔:
  ✓ 10_modules/gateway_architecture.md - 新增 OKX 閘道器章節
  ✓ 30_contracts/ - 新增 okx_config_contract.md
  ✓ 40_config/config_usage_map.md - 新增 OKX 配置路徑
  ✓ 90_operations/cli_operations_guide.md - 新增 OKX 啟動範例
  ✓ 00_index/index.yaml - 新增文檔項目
```

---

## 3) 任務清單（輸出：TO_CREATE 與 TO_UPDATE）

**步驟**:
1. 列出 TO_CREATE（需新建的文檔）
2. 列出 TO_UPDATE（需更新的現有文檔）
3. 列出 TO_SKIP（不需更新的文檔，附原因）

**預期輸出**:
```
【TO_CREATE】
1. .doc/30_contracts/okx_config_contract.md
   理由: 新增 OKX 配置結構，需完整契約規範

2. .doc/10_modules/okx_extension.md
   理由: 新增 OKX 交易所實作，需模組卡說明

【TO_UPDATE】
1. .doc/10_modules/gateway_architecture.md
   變更點: 新增「OKX 閘道器」章節，說明 REST/WebSocket 實作

2. .doc/40_config/config_usage_map.md
   變更點: 新增 OKX 配置檔路徑與範例

3. .doc/90_operations/cli_operations_guide.md
   變更點: 新增 md/td -s okx 啟動範例

4. .doc/00_index/index.yaml
   變更點: 新增 okx_config_contract.md 和 okx_extension.md 條目

【TO_SKIP】
1. .doc/20_interactions/order_lifecycle_flow.md
   理由: 訂單流程與交易所無關，無需更新

2. .doc/10_modules/yijinjing_journal.md
   理由: Journal 系統未改動
```

---

## 4) 批次更新（輸出：每批完成小結）

### 批次 0: 檢查程式碼錨點

**步驟**:
- 檢查本輪改動是否引入新的關鍵程式碼位置
- 若有，記錄錨點位置（`file:line` 格式）

**預期輸出**:
```
【程式碼錨點】
新增錨點:
  - core/extensions/okx/marketdata_okx.cpp:45-89 - WebSocket 連接邏輯
  - core/extensions/okx/trader_okx.cpp:120-180 - 下單 API
  - core/extensions/okx/include/common.h:18-65 - OKX 配置結構
```

### 批次 1: 更新契約層（30_contracts/）

**步驟**:
- 新建或更新 API 契約文檔
- 確保包含：結構定義、不變量、使用範例、安全警告
- 更新 `updated_at` 和 `code_refs`

**預期輸出**:
```
【批次 1 完成】
CREATED:
  ✓ 30_contracts/okx_config_contract.md (3,200 tokens)
    - OKX 配置結構（API key, secret, passphrase）
    - 安全警告（testnet vs mainnet）
    - 使用範例

UPDATED:
  ✓ 30_contracts/binance_config_contract.md
    - 新增「多交易所對比」章節
```

### 批次 2: 更新模組層（10_modules/）

**步驟**:
- 新建或更新模組卡
- 確保包含：概覽、架構、關鍵 API、最佳實踐、陷阱
- 更新相關連結

**預期輸出**:
```
【批次 2 完成】
CREATED:
  ✓ 10_modules/okx_extension.md (4,500 tokens)
    - OKX REST/WebSocket 實作
    - 與 Binance 實作的差異
    - 錯誤處理策略

UPDATED:
  ✓ 10_modules/gateway_architecture.md
    - 新增「支援的交易所」章節（Binance, OKX）
    - 更新架構圖
```

### 批次 3: 更新互動層（20_interactions/）

**步驟**:
- 更新互動流程圖（如需）
- 新增時序圖（如需）
- 更新程式碼引用

**預期輸出**:
```
【批次 3 完成】
UPDATED:
  ✓ 20_interactions/order_lifecycle_flow.md
    - 新增 OKX 訂單流程差異說明（passphrase 驗證）

SKIPPED:
  - 20_interactions/strategy_lifecycle_flow.md (策略流程未變)
```

### 批次 4: 更新配置層（40_config/）

**步驟**:
- 更新配置使用地圖
- 更新危險配置項清單（如有）
- 更新未使用配置項清單（如有）

**預期輸出**:
```
【批次 4 完成】
UPDATED:
  ✓ 40_config/config_usage_map.md
    - 新增 OKX 配置檔路徑
    - 新增 OKX 配置範例

  ✓ 40_config/dangerous_keys.md
    - 新增 passphrase 到危險鍵清單
```

### 批次 5: 更新操作層（90_operations/）

**步驟**:
- 更新 CLI 操作指南
- 更新 PM2 啟動指南
- 更新除錯指南（如需）

**預期輸出**:
```
【批次 5 完成】
UPDATED:
  ✓ 90_operations/cli_operations_guide.md
    - md -s 新增 okx 選項
    - td -s 新增 okx 選項
    - 新增 OKX 啟動範例

  ✓ 90_operations/pm2_startup_guide.md
    - 新增 md_okx.json 範例
    - 新增 td_okx.json 範例
```

### 批次 6: 更新索引（00_index/）

**步驟**:
- 更新 `index.yaml` 新增文檔項目
- 更新 `modules.yaml` 新增模組（如需）
- 更新 `CHANGELOG.md`（如需）

**預期輸出**:
```
【批次 6 完成】
UPDATED:
  ✓ 00_index/index.yaml
    - 新增 okx_config_contract.md 項目
    - 新增 okx_extension.md 項目
    - 更新 token 估算

  ✓ 00_index/modules.yaml
    - 新增 OKX 模組引用

  ✓ 00_index/CHANGELOG.md
    - 新增「2025-11-17: 新增 OKX 交易所支援」
```

---

## 5) 結案輸出（輸出：本輪改動的文檔同步結果）

**步驟**:
1. 彙總 CREATED、UPDATED、SKIPPED
2. 檢查 Token 預算是否超限
3. 提出下一步建議（可選）

**預期輸出**:
```
═══════════════════════════════════════════════
        文檔同步完成｜本輪改動：OKX 交易所支援
═══════════════════════════════════════════════

【CREATED】(2 個新檔案)
  ✓ .doc/30_contracts/okx_config_contract.md
    說明: OKX 配置結構契約（API key, secret, passphrase）

  ✓ .doc/10_modules/okx_extension.md
    說明: OKX 交易所閘道器實作（REST + WebSocket）

【UPDATED】(8 個檔案)
  ✓ .doc/10_modules/gateway_architecture.md
    說明: 新增 OKX 章節與多交易所架構圖

  ✓ .doc/40_config/config_usage_map.md
    說明: 新增 OKX 配置路徑與範例

  ✓ .doc/40_config/dangerous_keys.md
    說明: 新增 passphrase 危險鍵警告

  ✓ .doc/90_operations/cli_operations_guide.md
    說明: 新增 OKX 啟動範例

  ✓ .doc/90_operations/pm2_startup_guide.md
    說明: 新增 OKX PM2 配置範例

  ✓ .doc/20_interactions/order_lifecycle_flow.md
    說明: 新增 OKX 訂單流程差異

  ✓ .doc/00_index/index.yaml
    說明: 新增文檔索引項目

  ✓ .doc/00_index/CHANGELOG.md
    說明: 記錄本次更新

【SKIPPED】(3 個檔案)
  - .doc/10_modules/yijinjing_journal.md
    理由: Journal 系統未改動

  - .doc/10_modules/ledger_system.md
    理由: Ledger 系統未改動

  - .doc/20_interactions/strategy_lifecycle_flow.md
    理由: 策略生命週期未改動

【Token 預算檢查】
  新增文檔: ~7,700 tokens
  總文檔量: ~67,700 tokens (在 150k 預算內 ✓)

【NEXT】下一步建議
  1. 測試 OKX 連接: 執行 `python3 dev_run.py -l trace md -s okx`
  2. 驗證配置: 確認 ~/.config/kungfu/app/config/td/okx/ 配置正確
  3. 執行驗證腳本: `python3 .doc/90_operations/scripts/verify_code_refs.py`
  4. 更新 RAG 評測: 新增 OKX 相關測試查詢到 retrieval_eval.yaml
  5. 考慮新增 ADR: 記錄「為何選擇 OKX」的架構決策

═══════════════════════════════════════════════
```

---

# 文檔分類映射表（改動 → 需更新文檔）

## 策略層改動
**涉及**: `strategies/*.py`, `core/cpp/wingchun/strategy/`

**需更新**:
- ✓ `10_modules/strategy_framework.md` - 新增 API、回調、最佳實踐
- ✓ `30_contracts/strategy_context_api.md` - 更新方法簽名
- ✓ `20_interactions/strategy_lifecycle_flow.md` - 更新生命週期
- ○ `50_rag/retrieval_eval.yaml` - 新增測試查詢

## 交易所閘道器改動
**涉及**: `core/extensions/*/`, `core/cpp/wingchun/broker/`

**需更新**:
- ✓ `10_modules/gateway_architecture.md` - 新增閘道器章節
- ✓ `10_modules/<exchange>_extension.md` - 新增模組卡（如 okx_extension.md）
- ✓ `30_contracts/<exchange>_config_contract.md` - 新增配置契約
- ✓ `40_config/config_usage_map.md` - 新增配置路徑
- ✓ `90_operations/cli_operations_guide.md` - 新增啟動範例
- ○ `20_interactions/order_lifecycle_flow.md` - 如有特殊流程

## Ledger/Book 改動
**涉及**: `core/cpp/wingchun/service/ledger.*`, `core/cpp/wingchun/book/`

**需更新**:
- ✓ `10_modules/ledger_system.md` - 更新帳本邏輯
- ✓ `30_contracts/` - 更新 Position/Asset 契約（如有）
- ○ `20_interactions/` - 更新 PnL 計算流程（如有）

## Journal 系統改動
**涉及**: `core/cpp/yijinjing/`

**需更新**:
- ✓ `10_modules/yijinjing_journal.md` - 更新 Journal API
- ○ `20_interactions/event_flow.md` - 更新事件流程

## 配置結構改動
**涉及**: `config/*.json`, `*.h` (配置結構)

**需更新**:
- ✓ `40_config/config_usage_map.md` - 新增/修改配置項
- ✓ `40_config/dangerous_keys.md` - 新增危險鍵（如有）
- ✓ `30_contracts/<module>_config_contract.md` - 更新配置契約

## CLI 指令改動
**涉及**: `core/python/kungfu/command/`

**需更新**:
- ✓ `90_operations/cli_operations_guide.md` - 更新指令參考
- ○ `90_operations/pm2_startup_guide.md` - 更新 PM2 配置範例

## 資料結構改動
**涉及**: `core/cpp/wingchun/include/kungfu/wingchun/msg.h`

**需更新**:
- ✓ `30_contracts/<object>_contract.md` - 更新物件契約（Order, Depth, Trade 等）
- ✓ `10_modules/python_bindings.md` - 更新 pybind11 綁定（如有新欄位）

---

# 特殊情境處理

## 情境 1: Bug 修復（無 API 變更）

**策略**: 僅更新相關章節的「已知問題」或「陷阱」

**範例**:
```
改動: 修復 Order.volume_left 計算錯誤
文檔更新:
  ✓ 30_contracts/order_object_contract.md
    - 移除「已知問題」中的 volume_left 計算錯誤
    - 更新「不變量」章節確認正確公式
```

## 情境 2: 性能優化（無 API 變更）

**策略**: 更新「性能特性」章節

**範例**:
```
改動: Journal 寫入優化，延遲從 200μs 降至 50μs
文檔更新:
  ✓ 10_modules/yijinjing_journal.md
    - 更新「性能特性」表格
  ✓ 20_interactions/order_lifecycle_flow.md
    - 更新「延遲分解」表格
```

## 情境 3: 棄用 API

**策略**: 標記為 DEPRECATED，建議替代方案

**範例**:
```
改動: context.old_method() 棄用，改用 context.new_method()
文檔更新:
  ✓ 30_contracts/strategy_context_api.md
    - 標記 old_method() 為 DEPRECATED
    - 新增 new_method() 文檔
    - 新增「遷移指南」章節
```

## 情境 4: 重構（內部實作改變）

**策略**: 評估是否影響外部 API，若否則最小化更新

**範例**:
```
改動: Ledger 內部重構，外部 API 不變
文檔更新:
  ✓ 00_index/CHANGELOG.md
    - 記錄重構事項
  SKIPPED:
    - 10_modules/ledger_system.md (外部 API 未變)
```

---

# 文檔更新檢查清單

更新每個文檔時，確認：

- [ ] **Front-matter 完整**
  - [ ] `title` 正確
  - [ ] `updated_at` 更新為當前日期
  - [ ] `owner` 適當
  - [ ] `tags` 包含相關標籤
  - [ ] `code_refs` 更新程式碼引用

- [ ] **內容一致性**
  - [ ] 程式碼範例與實際程式碼一致
  - [ ] `file:line` 引用正確（執行 verify_code_refs.py 驗證）
  - [ ] 跨文檔連結有效（執行 check_links.py 驗證）
  - [ ] 術語使用一致

- [ ] **Token 預算**
  - [ ] 單檔案不超過 8,000 tokens（軟性限制）
  - [ ] 總文檔量在 150k auto-load 預算內
  - [ ] 執行 estimate_tokens.py 更新估算

- [ ] **交叉引用**
  - [ ] "Related Documentation" 章節完整
  - [ ] 相關模組/契約互相連結
  - [ ] 操作指南連結到契約

---

# 限制與例外

- **不改動範圍**:
  - ❌ 程式碼檔案（除非使用者明確授權）
  - ❌ 測試檔案
  - ❌ 構建配置

- **延後處理**:
  - ○ 複雜圖表（可先描述，後續補圖）
  - ○ 詳細範例（可先給骨架，後續補完整）

- **例外情況**:
  - 若改動涉及超過 15 個文檔，詢問使用者是否分批更新
  - 若 Token 預算接近上限，詢問使用者是否拆分文檔或標記為 opt-in

---

# 快速檢核（內部完成）

更新完成前，確認：
- [ ] 我已遵守語言/引用/來源優先序
- [ ] 我已輸出每批次的更新小結
- [ ] 我已檢查 `file:line` 引用的正確性
- [ ] 我已更新所有相關文檔的 `updated_at`
- [ ] 我已提供 CREATED/UPDATED/SKIPPED 總結
- [ ] 我已建議下一步行動（測試、驗證等）

---

**版本**: 2025-11-17
**維護者**: core-dev
**適用場景**: 程式碼改動後文檔同步、功能新增文檔更新、API 變更文檔維護
