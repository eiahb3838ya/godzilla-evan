---
title: UPDATE（功能改動後的文檔同步｜只針對本輪改動）
updated_at: 2025-11-13
owner: core-dev
lang: zh-TW
---

# 目的
當使用者在完成「某一輪功能/程式改動」後，在新對話輸入「遵行 UPDATE.md」，你應以「本輪改動」為唯一事實來源，先總結改動，再據此更新 `.doc/`。不處理其他場景。

# 執行政策（全程遵守）
- 語言：回覆一律使用繁體中文（zh‑TW）。
- 範圍：僅修改 `.doc/`，不變動 `context_store/` 舊檔；不修改程式碼（除非使用者另行授權）。
- 真實來源：以「本輪改動集」為準（來自本次對話的實作/描述或提供的 diff）；不得擴散到未涉及的元件。

# 附帶檔案處理（/plan 的 PRD 與 /test 的 TDD）
若使用者在呼叫本文件時，附帶檔案位於下列路徑，請執行對應複製與命名（必要時建立目錄）：

- PRD：任何位於 `/plan/**` 的檔案 → 複製到 `.doc/30_contracts/prd/`
  - 命名規則（prd）：
    - 以原檔名（去副檔名）轉為 snake_case 作為 `<name>`；
    - 若未帶 `_prd` 後綴，則改名為 `<name>_prd.<ext>`；
    - 保留原始副檔名 `<ext>`（通常為 `.md`）。
  - Front‑matter：若缺少 `title/updated_at/owner/lang`，請最小補齊（owner 建議 `prd-owners`，lang 為 zh‑TW）。

- TDD：任何位於 `/test/**` 的檔案 → 複製到 `.doc/30_contracts/tdd/`
  - 命名規則（tdd）：
    - 以原檔名（去副檔名）轉為 snake_case 作為 `<name>`；
    - 若未帶 `_tdd` 後綴，則改名為 `<name>_tdd.<ext>`；
    - 保留原始副檔名 `<ext>`（通常為 `.md`）。
  - Front‑matter：若缺少 `title/updated_at/owner/lang`，請最小補齊（owner 建議 `tdd-owners`，lang 為 zh‑TW）。

完成上述複製後，將新檔列入 CREATED 區塊；若覆蓋既有檔名則列入 UPDATED。
- 引用：優先使用 anchors（`# ctx:anchor:`），其次 `src/path:line`；避免絕對路徑。
- 來源優先序：`.doc/` > `src/` > PRD 原文（`.doc/30_contracts/prd/`，只作追溯，不作主依據）。
- 節奏：先產出計畫與差異清單，再分批更新；每批更新後給 1–2 行小結。

# 更新流程（逐步執行並輸出所需內容）
1) 初始化（輸出：執行計畫）
   - 重述執行政策（簡述）。
   - 明確「本輪改動」來源：
     - 若使用者已提供改動摘要/檔案清單/commit 訊息，直接採用；
     - 否則請求使用者提供最小資訊：功能名稱/目的/涉及檔案或模組；
     -（可選）若有 git 訪問，使用 `git diff --name-only`/`git log -1` 補輔助，但以使用者描述為準。
   - 輸出你的更新計畫（含「附帶檔案處理」）：
     - 批次 0：處理附帶 `/plan` PRD 與 `/test` TDD → cp 至 `.doc/30_contracts/prd|tdd` 並按規則命名；
     - 批次 1：彙整改動→映射頁面；
     - 批次 2..n：分批更新→結案。

2) 改動彙整（輸出：改動總結與影響面）
   - 產出「本輪改動總結」：功能名稱、目的、涉及模組/檔案、核心接口/I/O/鍵位變更、是否新增/修改 anchors。
   - 影響映射：依改動對應到需更新的文檔區塊（只挑與改動相關者）：
     - 訓練/排程：10_modules/module_card__trainer.md、20_interactions（trainer 相關）、30_contracts/trainer_contract.md、40_config（涉及鍵位時）。
     - 資料集/前處理：10_modules/module_card__dataset.md / module_card__preprocessor.md、20_interactions（data pipeline）、30_contracts/dataset_*_contract.md / preparation_contract.md。
     - 損失/度量：10_modules/module_card__loss.md / module_card__metrics.md、30_contracts/loss_contract.md / metrics_contract.md。
     - 配置：40_config/config_usage_map.md / dangerous_keys.md / unused_keys.md（如有鍵位新增/更名）。
     - RAG：50_rag/chunking_config.yaml（若 anchors 新增/改名）、retrieval_eval.md（可補一則結果片段）。
     - 圖表：60_diagrams/*（若呼叫鏈/相依有變動）。

3) 任務清單（輸出：TO_CREATE 與 TO_UPDATE）
   - TO_CREATE：列出本輪改動要求新增的文件（相對路徑）與理由（1 行）。
   - TO_UPDATE：列出需更新的既有文件（相對路徑）與變更點（1–2 行）。

4) 批次更新（輸出：每批完成小結）
   - 批次 0（若有附帶檔案）：
     - 依「附帶檔案處理」規則，將 `/plan/**` → `.doc/contract/prd/`、`/test/**` → `.doc/contract/tdd/`，並套用命名/front‑matter 規則；
     - 輸出：CREATED/UPDATED 的檔案清單 + 一行變更摘要。
   - 批次 1（必要）：更新受影響頁面的內容要點（介面/I/O/鍵位/時序/互動），同步 front‑matter 的 `updated_at` 與（若需要）`owner`。
   - 批次 2（如需）：補 anchors 引用（在相關頁面加入「關鍵 anchors」小節或將原本 `path:line` 轉為 anchors）。
   - 批次 3（如需）：修正不一致（例如移除絕對路徑、去占位語句）與補結果片段（RAG/評測）。
   - 每批更新後，輸出「已更新檔案清單 + 一行變更摘要」。

5) 結案輸出（輸出：本輪改動的文檔同步結果）
   - CREATED：列出新建檔案（相對路徑）與 1 行說明。
   - UPDATED：列出已更新檔案與 1 行說明。
   - SKIPPED/DEFERRED（如有）：列出原因（例如 PRD 原文僅追溯、不修改）。
   - NEXT（可選）：基於「本輪改動」提出下一步（如：補 anchors 到程式碼、擴充 QA 範例）。

# 分區準則（更新時參考）
- 00_index：保留簡潔導航；可在 `index.md` 補當期「What's new」摘要（≤5 行）。
- 20_interactions：每張圖底部增加 3–6 個關鍵 anchors/路徑；避免保留自動生成腳本描述。
- 30_contracts：專注不變量與 I/O；PRD 原文放 `.doc/30_contracts/prd/`；在 `code_refs` 中優先 anchors。
- 10_modules：聚焦 Public API/依賴/陷阱；路徑採相對 `src/` 或 anchors。
- 40_config：統一路徑與命名（避免絕對路徑）；`unused_keys.md` 附日期與責任人；`dangerous_keys.md` 保留高風險鍵名與理由。
- 50_rag：chunking_config.yaml anchors hints 需與程式碼錨點一致；`retrieval_eval.md` 可補 1 則真實結果片段（若可）。
- 60_diagrams：不保留可執行「來源腳本」；必要時提供外部工具倉庫連結。
- 80/85：front‑matter 齊備；長文允許待後補潤飾，先確保標題/owner/lang/updated_at 正確。

# 輸出模板（更新完成時使用｜專屬本輪改動）
- CREATED
  - <path> — <一句話說明>
- UPDATED
  - <path> — <一句話說明>
- SKIPPED/DEFERRED（如有）
  - <path> — <原因>
- NEXT（可選）
  - <下一步建議>

# 限制與例外
- 不改動 `context_store/` 舊版；`.doc/` 以外的變更需另行授權。
- PRD 原文僅歸檔追溯，不直接改寫其內容。
