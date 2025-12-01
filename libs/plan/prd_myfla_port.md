# PRD：myfla 模塊移植指南（libs/fla → libs/myfla）

本文件定義「如何在給定一個 `libs/fla` 模塊時，復刻出邏輯與數學完全相同的 `libs/myfla` 模塊」的標準流程。所有具體模塊的細節請參考對應的專屬 PRD（例如 RWKV7 的 `plan/prd_rwkv7_attn.plan.md`）；本指南僅提供通用、最不易遺漏的操作步驟。

---

## 1. 基礎原則

1. **接口一致**：myfla 與 fla 的公開 API、模塊結構、命名必須一致，以便策略層與 config 不需修改即可切換。
2. **純 PyTorch**：所有計算需以 PyTorch 實作，嚴禁依賴 Triton、CUDA 專用 kernel 或 torch.compile。
3. **逐條核對**：任何省略或「自行簡化」都必須在專屬 PRD 中明示；若無註記，一律視為必須完全一致。
4. **測試先行**：每個模塊必須附帶 `tests/myfla/test_<module>.py`，並以 `PYTHONPATH=src python3.8 ...` 為標準執行方式。

---

## 2. 標準作業流程（SOP）

### Step 1：規格盤點與依賴表
- 逐行閱讀目標 `libs/fla` 檔案，整理：
  - `__init__` 參數與預設值
  - forward 的輸入/輸出、cache/state 結構
  - 依賴的子模塊（LayerNorm、LoRA、delta-rule 等）
- 產生「功能核對表」：將核心邏輯（chunk 模式、state 更新、mask、cu_seqlens）與可延後的加速功能（Triton kernel、torch.compile）分開列出。

### Step 2：建立資料流參考（即便無法運行官方 fla）
- 依據原始碼與文檔推導各層的張量關係，至少涵蓋：
  - shape/dtype、broadcast 規則、`cu_seqlens` 切片方式。
  - cache/past_key_values 的欄位命名與生命周期。
  - 每個分支（mask=True/False、`use_cache` on/off、`layer_idx` 變化）的期望行為。
- 針對上述推導，設計可重複的測試輸入（固定 random seed），並將「預期行為」寫成 pytest/unittest 內部的 assert：例如「輸入全零時，state 必須維持零」、「mask 採左側 padding 時，對應時間步必須為零」等。
- 若日後取得官方 fla 環境，可再補充真實 fixture；在此之前必須維持這份「推導 + 測試樣本」作為 pseudo-fixture，且在模塊 PRD 中明記。

### Step 3：替換依賴並撰寫純 PyTorch 版本
- 依核對表逐一實作子模塊（LoRA、token_shift、delta-rule、gate correction、short conv……）。
- 若官方使用 fused kernel：以 for-loop、`torch.einsum` 或 `torch.autograd.Function` 取代，並以 `warnings.warn` 標註效能/精度差異。
- 所有模塊必須保留與 fla 相同的參數名稱與 default 值。

### Step 4：主模塊實作與單元測試
- 在 `libs/myfla/layers/<module>.py` 完成類別實作，forward 回傳值必須與原版一致。
- 撰寫 `tests/myfla/test_<module>.py`，以 fixture 或手工構造資料比對：
  - 張量結果（允許極小誤差）
  - cache/past_key_values 結構
  - mask / `cu_seqlens` 等所有分支
- 任何差異都需在測試中以 assert 或註解明確說明（例如「純 PyTorch 版不支援 fused_recurrent 模式」）。

### Step 5：整合冒煙
- 透過 `FLAEncoderFactory` 或更高層的模型載入 myfla 模塊，執行最小端到端資料流（範例：`tests/myfla/test_fla_encoder_strategy_integration.py`）。
- 驗證多層 cache 串接、config 切換、factory 註冊等高層功能皆可使用 myfla 版本。

### Step 6：文件與差異紀錄
- 在專屬 PRD（例如 `plan/prd_rwkv7_attn.plan.md`）紀錄：
  - 參考的 fla 檔案、依賴列表、測試命令
  - fixture 路徑與使用方式
  - 與官方行為的任何已知差異（性能、精度、未實作功能）
- 本文件只敘述通用流程，任何模塊具體細節請在對應 PRD 中追蹤。

---

## 3. 專案結構建議

- `libs/myfla/`
  - `layers/`：對應 fla 的 layer 定義。
  - `modules/`：共用模塊（GroupNorm、LoRA、token_shift、short conv 等）。
  - `ops/`：自訂算子或遞迴（例如 delta-rule）。
  - `utils/`：雜項工具。
- `tests/myfla/`
  - `test_<module>.py`：單元測試。
  - `test_*integration*.py`：編碼器或模型的冒煙測試。
  - `data/`（可選）：存放 fixture。

---

## 4. 文件鉤子

- 實際模塊的 PRD 需掛載於本 SOP。例如：
  - `plan/prd_rwkv7_attn.plan.md`：RWKV7Attention 的核對表與 TDD。
  - `plan/prd_gated_deltanet.plan.md`：GatedDeltaNet 的復刻計畫。
- 策略/工廠層的整體設計請參閱 `plan/prd_fla_import.md`，確保上層僅依賴 myfla。

---

## 5. 現況摘要（截至 2025-11-21）

- myfla 骨架與核心子模塊（LoRA、token_shift、delta-rule、gate correction）已完成並透過 `tests/myfla/*` 覆蓋。
- RWKV7Attention 依照上述流程實作，詳細內容見 `plan/prd_rwkv7_attn.plan.md`。
- GatedDeltaNet 等其他模塊尚待按此 SOP 推進，完成人員需自行建立對應 PRD 與測試。
- 尚未產出 GPU/Triton fixture；待環境允許時須補齊以驗證 myfla 與官方輸出的誤差範圍。

---

## 6. 尚待決議

1. myfla 是否需要完整覆蓋 `libs/fla` 所有模塊，或僅針對當前任務使用到的子集。
2. 可接受的效能門檻為何？純 PyTorch 如顯著慢於官方實作，是否需導入局部 CUDA kernel。
3. 若未來環境允許安裝 Triton，myfla 與 fla 是否並存或擇一。

--- 
