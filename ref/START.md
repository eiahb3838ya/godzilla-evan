---
title: START（對話啟動腳本｜自動學習 .doc 語境）
updated_at: 2025-11-13
owner: core-dev
lang: zh-TW
---

# 說明
你是一個閱讀/回答代理。當使用者在新對話中輸入「遵行 START.md」，請嚴格依以下步驟「自動學習本倉庫的語境層 .doc」，完成後回覆 READY 並等待下一個指令。

特殊規則（memory 模式）：
- 若本次指令附帶一個位於 `.doc/85_memory/**` 的 `*.memory.md` 檔案，表示本輪對話將「沿該 memory 方向持續開發與研究」。
- 在完成基礎學習後，進入「記憶導向」流程：針對該 memory 檔進行摘要、目標提煉、工作計畫輸出，並詢問是否立刻展開。

# 執行政策（全程遵守）
- 語言：回覆一律採用繁體中文（zh‑TW）。
- 來源優先序：`.doc/` > `src/` > PRD 原文（`.doc/30_contracts/prd/`）。
- 節奏：先給結論/摘要，再附路徑與錨點，最後補細節；避免貼整段長文。
- 不要修改任何檔案；閱讀後在你的「對話記憶」中保留要點即可。

# 流程（一步步執行並在每步產出對應輸出）
1) 初始化（輸出：確認與執行計畫）
   - 重述「執行政策」。
   - 輸出你的閱讀計畫（1–2 行）。

2) 入口導航（輸出：1 段專案摘要 + 2 條導覽路徑）
   - 讀 `.doc/00_index/overview.md`、`.doc/00_index/index.md`。
   - 產出：專案一段摘要；推薦閱讀順序的兩條路徑（互動圖優先、契約優先）。

3) 互動圖（輸出：呼叫鏈 5 行摘要 + 3 個風險/疑慮）
   - 讀 `.doc/20_interactions/` 中關鍵頁：
     - `map__exp_cfg_trainer_model_loss.md`
     - `map__trainer_scheduler_curriculum.md`
     - `map__data_pipeline_dataset_preprocessor.md`
     - `map__optimization_scheduler_param_groups.md`
     - `map__dataset_modes_matrix.md`
     - `map__preparation_timeline.md`

4) 契約（輸出：逐頁「不變量/輸入輸出/例外」要點清單）
   - 讀 `.doc/30_contracts/`：
     - `dataset_feather_contract.md`、`dataset_lazy_contract.md`、`dataset_batch_contract.md`
     - `trainer_contract.md`、`loss_contract.md`、`metrics_contract.md`
     - `preparation_contract.md`
   - 每頁列：不變量（形狀/鍵位/裝置責任）、I/O、例外或限制。

5) 模組卡（輸出：核心 API 與陷阱 3–5 條）
   - 讀 `.doc/10_modules/`：
     - `module_card__trainer.md`、`module_card__dataset.md`、`module_card__preprocessor.md`
     - `module_card__loss.md`、`module_card__metrics.md`、`module_card__toolbox.md`

6) 配置治理（輸出：Top-10 鍵位與 3 條風險鍵）
   - 讀 `.doc/40_config/config_usage_map.md`、`dangerous_keys.md`、`unused_keys.md`。

7) RAG 與檢索評測（輸出：索引策略 3 點 + anchors 使用說明）
   - 讀 `.doc/50_rag/chunking_config.yaml`、`retrieval_eval.md`、`rag_pipeline.md`。
   - 不執行任何腳本；僅記錄「.doc 優先、anchors 已啟用」與評測要點。

8) 錨點對照（輸出：錨點→頁面/職責清單）
   - 將下列 anchors 記入你的對話記憶，並提供對應說明：
     - 訓練/排程：`trainer_oos_eval_modes`、`lr_scheduler_adjust_warmup`、`tabm_outputs_reduce`
     - 資料集：`dataset_feather_index_cols`、`dataset_lazy_index_cols`、`dataset_lazy_return_types`
     - 損失/度量：`ranking_wrapper_contract`、`multi_metric_collection_api`
     - 構建入口（exp）：`exp_build_model`、`exp_build_criterion`、`exp_build_optimizer`、`exp_build_trainer`
     - 設定入口（cfg）：`config_get_module`、`config_load_entry`、`config_parse_filter`
     - 準備（bins）：`preparation_bins_cache_dir`、`preparation_compute_bins`、`preparation_bins_pipeline`

9) 研究/記憶（輸出：如不影響效能，掃描標題與 front‑matter）
   - 略讀 `.doc/80_research/*`、`.doc/85_memory/*` 的標題與 front‑matter（title/tags/owner）。
   - 僅在後續任務需要時再深入。

10) 完成（輸出：READY + 兩段總結 + 下一步詢問）
   - 回覆 `READY`。
   - 總結兩段：
     1) 我已學會的核心：呼叫鏈、契約不變量、anchors 對照。
     2) 如何檢索與引用：.doc 優先、anchors、`src/path:line`。
   - 詢問使用者下一步目標（例如「接著閱讀某 memory.md」或「回答某設計問題」）。

11) 記憶導向（僅當附帶 `.doc/85_memory/**.memory.md` 檔案時啟用）
   - 讀取該 memory 檔：
     - 輸出 3–5 行摘要（背景/目標/現況）。
     - 列出該 memory 的「明確待辦/研究假設/已知風險」（各 3 項內）。
     - 建立「本輪工作計畫」（2–4 步）：例如補充實驗設計、驗證指標、契約/互動圖需增補之處、RAG 樣例或 TDD 雛形（僅文檔層）。
     - 映射關聯 anchors 或 `src/path:line` 以便後續提問與檢索。
   - 問句收束：
     - 提出兩個選項：
       1) 立刻按計畫第 1 步執行（請確認）。
       2) 調整計畫（請指出需增刪的步驟）。

# 限流與例外
- 單頁過長者：先列重點與 anchors，再視需要補充。
- 若檔案缺失或不可讀：回報檔名並跳過，不中斷整體流程。
- PRD 原文：僅做追溯參考，不作為主依據。

# 快速檢核（你需在內部完成）
- [ ] 我已遵守語言/引用/來源優先序。
- [ ] 我已輸出每一步要求內容。
- [ ] 我已建立 anchors 對照記憶。
