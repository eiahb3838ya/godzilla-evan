# PRD：純 PyTorch 版 GatedLinearAttention（GLA）復刻計畫

## 0. 緒論與定位
- **目標模塊**：`libs/fla/layers/gla.py` 內的 `GatedLinearAttention`（GLA）層。
- **SOP 錨點**：依 `plan/fla/prd_myfla_port.md` 的流程推進，將 GLA 逐步移植到 `libs/myfla`。
- **目前進度**：Step 1（規格盤點與依賴表）。

## 1. Step 1 — 規格盤點與依賴表
### 1.1 主要參數與結構
- `mode`: `'chunk' | 'fused_recurrent' | 'fused_chunk'` — 決定使用 `chunk_gla`、`fused_recurrent_gla` 或 `fused_chunk_gla` ops。
- 投影維度：`hidden_size`、`expand_k`、`expand_v` 決定 `key_dim、value_dim`，並依 `num_heads/num_kv_heads` 拆分。
- `use_short_conv`：決定是否透過 `ShortConvolution` 對 q/k/v 做 causal depthwise conv；cache 結構為 `(conv_state_q, conv_state_k, conv_state_v)`。
- `use_output_gate` + `gate_fn` + `fuse_norm`：若 `gate_fn='swish'` 且 `fuse_norm=True`，會走 `FusedRMSNormGated`，否則為 `RMSNorm + gate_fn`。
- 其他鍵位：`gate_logit_normalizer`（log-sigmoid 後的縮放）、`gate_low_rank_dim`（門控低秩投影）、`clamp_min`（輸出最小值）。

### 1.2 Forward 流程（高階摘要）
1. **Mask / Varlen 處理**：若有 `attention_mask`，透過 `get_unpad_data` + `index_first_axis` 轉成 varlen；也可傳入 `cu_seqlens`（供 fused kernel）。
2. **Short Conv（可選）**：對 `q_proj/k_proj/v_proj` 的輸出套 `ShortConvolution`，更新 conv cache。
3. **Feature map**（可選）：`feature_map_fn` 作用於 q/k（例如 `relu`, `elu`, `swish`）。
4. **GLA 核心運算**：依 `mode` 呼叫 `chunk_gla`、`fused_chunk_gla` 或 `fused_recurrent_gla`，計算 `o` / `gk` / state；支持 `cu_seqlens`、`initial_state`、`output_final_state`。
5. **Gate + Norm**：若啟用 gate 且 fuse 模式則 `FusedRMSNormGated(out, gate)`，否則 `RMSNorm(out) * gate_fn(g_proj(...))`。
6. **投影 / 還原**：`o_proj` 將 `[batch, len, value_dim]` 回射到 `hidden_size`，若先前有 unpad 則 `pad_input` 還原。
7. **Cache 更新**：`past_key_values.update(conv_state=..., recurrent_state=..., layer_idx, offset)`。

### 1.3 依賴模塊
- **投影 & conv**：`ShortConvolution`（需 PyTorch 版已就緒）、`nn.Linear` 投影、`ACT2FN`。
- **Norm / Gate**：`FusedRMSNormGated` 與 `RMSNorm`（需完整功能，包含 residual/prenorm 等）。
- **Ops**：`chunk_gla`, `fused_chunk_gla`, `fused_recurrent_gla`（位於 `libs/fla/ops/gla/*`，需 PyTorch 版本）。
- **Utils**：`get_unpad_data`, `index_first_axis`, `pad_input`, `rearrange/repeat`。
- **Cache**：使用 HuggingFace 風格的 `Cache`，需與 myfla 版 stub 或 Legacy cache 對應。

### 1.4 Mask / Cache / Varlen 行為
- `attention_mask`: 只接受 `[B, seq_len]`（0=padding），不允許任意 `[B, L, L]`。
- Varlen：`cu_seqlens` 供 fused kernel；forward 會支援 `cu_seqlens` 或 mask 二選一。
- Cache：`past_key_values[layer_idx]` 記錄 `conv_state`（tuple）與 `recurrent_state`，各模式需能續接。

### 1.5 待建模重點
- 完整 `FusedRMSNormGated` 功能（activation/residual/prenorm）須到位。
- GLA ops（chunk/fused）需 PyTorch 版本，並與 `plan/prd_kda.plan.md` 中的 `chunk_gla_*` API 對齊。
- 測試需覆蓋三種 mode、mask/unpad、short conv cache、`use_output_gate` true/false、`cu_seqlens` 路徑。

## 2. Step 2 — 資料流推導與 pseudo-fixture（已完成）
### 2.1 張量流與 shape
- 輸入 `hidden_states`: `[B, L, hidden_size]`。若 `attention_mask` 存在，利用 `get_unpad_data` 轉為 varlen，輸出 `[1, total_tokens, hidden_size]` 並攜帶 `indices`/`cu_seqlens`。
- `q_proj`: `[B, L, key_dim]` → reshape `[B, L, num_heads, head_k_dim]`。
- `k_proj`: `[B, L, key_dim_per_group]` → reshape `[B, L, num_kv_heads, head_k_dim]` 再 `repeat` or `rearrange` 成 `[B, L, num_heads, head_k_dim]`。
- `v_proj`: `[B, L, value_dim_per_group]` → reshape 至 `[B, L, num_heads, head_v_dim]`。
- `gk_proj`: 同 `k_proj` 維度，用於 gate logits，最後 `logsigmoid(gk)/gate_logit_normalizer`。
- `ShortConvolution` 路徑：每個投影經 `ShortConvolution`（shape `[B, L, dim]`），`cache` 為 `(B or N, dim, kernel_size)`；varlen 模式時 `ShortConvolution` 逐序列處理。
- 模式切換：若 `L <= 64` 強制 `mode='fused_recurrent'`，否則使用初始化時指定的 mode。
- `chunk_gla/fused_chunk_gla/fused_recurrent_gla` 輸入 q/k/v/gk 與 optional `recurrent_state`、`cu_seqlens`，輸出 `(o, new_state)` 其中 `o` shape 為 `[B, L, num_heads, head_v_dim]`。
- Gate + Norm：若 fuse 模式成立，`o` 與 g reshape 為 `[B, L, num_heads, head_v_dim]` 進入 `FusedRMSNormGated`；否則 `RMSNorm(o)` × `gate_fn(g_proj(...))`。
- 最終 `o_proj` 產生 `[B, L, hidden_size]`；若先前 unpad 過，使用 `pad_input` 以 `indices` 還原。

### 2.2 Mask / `cu_seqlens` 流程
- `attention_mask`: 形狀 `[B, L]`，0 表示 padding；GLA 僅允許 padding mask，不允許任意 `[B, L, L]`。
- 先 `get_unpad_data` 取 `indices` 和 `cu_seqlens`（供 chunk/fused kernel）。資料將 reshape 為 `[1, total_tokens, hidden_size]`，便於 varlen 處理。
- `ShortConvolution` 亦接受 `cu_seqlens`，逐序列維護 cache（`conv_state_q/k/v`）。
- `chunk_gla` / `fused_recurrent_gla` 會直接消費 `cu_seqlens`（varlen）或 `[B,L]`（固定長度）資料。
- `pad_input` 使用 `indices` revert 成 `[B,L,hidden_size]`。

### 2.3 Cache lifecycle
- `past_key_values[layer_idx]` 內容：
  - `conv_state`: `(conv_state_q, conv_state_k, conv_state_v)` 或 `None`（未啟用 short conv）。
  - `recurrent_state`: 由 GLA ops 回傳的 state。
  - `layer_idx`, `offset`（GLA 使用 `q_len` 更新 offset）。
- Forward 流程：
  1. 若 `past_key_values` 內已有本層資料，拆出 conv/recurrent state。
  2. 新資料處理完後，若 `use_cache=True` 或 `past_key_values` 不為 None，呼叫 `past_key_values.update(...)` 寫入。

### 2.4 pseudo-fixture（測試輸入與預期行為）
為後續 TDD 設計四組可重複輸入：
1. **Basic chunk 模式**：`B=2,L=8,hidden=16,num_heads=2,use_short_conv=False,use_output_gate=True`，固定 random seed。預期：
   - `o.shape=(2,8,16)`，`past_key_values[0]['recurrent_state']` 不為 None。
   - `attention_mask=None` 時 `pad_input` 不應觸發；`g_norm_swish_gate` 路徑被使用（因 gate_fn='swish' 且 fuse_norm=True）。
2. **Mask + short conv**：`attention_mask=[[0,0,1,1],[0,1,1,1]]`、`use_short_conv=True`。預期：
   - `pad_input` path 啟用，前兩個 padding token 對應位置輸出全 0。
   - `conv_state` tuple 長度 3，元素 shape `[num_layers?, hidden?, conv_size]`（依 varlen 分支而定）。
3. **`cu_seqlens` + feature map**：傳入 varlen （`cu_seqlens=[0,5,9]`），啟用 `feature_map='relu'`。預期：
   - `ShortConvolution` 每段 len 分別為 5/4。
   - `chunk_gla` 使用 `cu_seqlens`，`o` 與 expected chunk 版本一致（可與 chunk 模式 assert allclose）。
4. **Fused recurrent mode**：`L=32`（<64）、`mode='chunk'`，`eval()`。模組應自動改走 `fused_recurrent_gla`，`past_key_values` 續接後第二次 forward 的 `recurrent_state` shape 與第一次一致；輸出 finite。

這些 pseudo-fixture 將寫入 `tests/myfla/test_gla.py` 的 helper，並在 Step 4 TDD 時套用。

## 3. Step 3 — myfla 版本實作計畫
- **目標**：依 Step 1/2 的規格與 pseudo-fixture，完成 GLA 各模組在 `libs/myfla` 的純 PyTorch 實作，並確保與官方 `libs/fla/layers/gla.py` 完整對齊。

### 3.1 Layer 骨架 (`libs/myfla/layers/gla.py`)
1. **建立 `class GatedLinearAttention(nn.Module)`**
   - 對應官方：`libs/fla/layers/gla.py:37-203`
   - 簽名：`__init__(self, mode='chunk', hidden_size=1024, expand_k=0.5, expand_v=1.0, num_heads=4, num_kv_heads=None, feature_map=None, use_short_conv=False, conv_size=4, conv_bias=False, use_output_gate=True, gate_fn='swish', elementwise_affine=True, norm_eps=1e-5, gate_logit_normalizer=16, gate_low_rank_dim=16, clamp_min=None, fuse_norm=True, layer_idx=None, **kwargs)`
   - 子任務：
     - 依官方計算 `key_dim/value_dim/head_k_dim/head_v_dim` 與多頭 reshape 邏輯。
     - 初始化投影層 `q/k/v/gk/o_proj`，與 `gate` 低秩投影（若 `gate_low_rank_dim` 不為 None）。
     - 若 `use_short_conv=True`，建立 `ShortConvolution` 實例並保存 `state_size`。
     - 根據 `gate_fn/fuse_norm` 決定使用 `FusedRMSNormGated` 或 `RMSNorm + gate_fn`。
     - 初始化 feature map 函式（透過 `ACT2FN` 或自訂 `relu/elu/swish` 等）。
2. **Forward 介面實作**
   - 對應官方：`libs/fla/layers/gla.py:205-371`
   - 簽名：`forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, output_attentions=False, cu_seqlens=None, **kwargs)`
   - 子任務：
     - 支援 mask → `get_unpad_data` → varlen ⇒ `pad_input` 整個流程（與 `index_first_axis` 配套）。
     - 由 `past_key_values` 讀取 `conv_state_q/k/v` 與 `recurrent_state`。
     - 根據輸入長度 `seq_len` 及 `self.mode` 決定呼叫 chunk/fused_chunk/fused_recurrent。
     - 前後串接 short conv → feature map → GLA ops → gate/norm → output projection。
     - 結束時若 `use_cache=True` 或 `past_key_values` 非空，呼叫 `past_key_values.update(conv_state=..., recurrent_state=..., layer_idx=self.layer_idx, offset=seq_len)`。
     - 返回 `(output, None, past_key_values)` 與官方一致。

### 3.2 ShortConvolution 與 Norm 模塊（與 Layer 共用）
1. **ShortConvolution (`libs/myfla/modules/convolution.py`)**
   - 對應官方：`libs/fla/modules/convolution.py:888-963`
   - 子任務：
     - 確認 myfla 版已支援 mask 或 `cu_seqlens`（若未實作 varlen，需補 `_process_single_sequence` 內之 per-sequence 迴圈）。
     - Cache 行為需與官方一致：`cache` shape `[num_seq, dim, kernel_size]`，varlen 模式逐序列更新。
     - `state_size` 屬性需提供給 encoder 其他部分。
2. **FusedRMSNormGated / RMSNorm (`libs/myfla/modules/layernorm.py`)**
   - 對應官方：`libs/fla/modules/fused_norm_gate.py`
   - 子任務：
     - 補齊 `activation` 參數（例如 `swish/gelu/sigmoid/identity`）與 `residual`、`prenorm` 選項。
     - `elementwise_affine`、`residual_in_fp32` 等參數需與官方 API 對齊（即便初期以 PyTorch fallback 實作）。
     - 若未支援，需加入 gate logits 正規化（`gate_logit_normalizer`）與 clamp 行為。

### 3.3 GLA Ops PyTorch 版 (`libs/myfla/ops/gla/`)
1. **`chunk_gla`**
   - 對應官方：`libs/fla/ops/gla/chunk.py`
   - 簽名：`chunk_gla(q, k, v, gk, cu_seqlens=None, initial_state=None, output_final_state=False, feature_map_fn=None, clamp_min=None)`
   - 子任務：
     - 實作 WY 分解（類似 KDA/GatedDeltaNet chunk 路徑），支援 `cu_seqlens`、`initial_state`。
     - `feature_map_fn`（如 `relu/elu/swish`）應在 ops 內對 q/k 套用。
     - 回傳 `(outputs, new_state)` 供 main layer 使用。
2. **`fused_chunk_gla`**
   - 對應官方：`libs/fla/ops/gla/fused_chunk.py`
   - 簽名：`fused_chunk_gla(q, k, v, gk, chunk_size, feature_map_fn=None, **kwargs)`
   - 子任務：
     - PyTorch fallback：將輸入分 chunk 後呼叫 `chunk_gla` 或以 for-loop 實作。
     - 支援 `gate_logit_normalizer/clamp_min` 等參數，保持輸出一致。
3. **`fused_recurrent_gla`**
   - 對應官方：`libs/fla/ops/gla/fused_recurrent.py`
   - 簽名：`fused_recurrent_gla(q, k, v, gk, initial_state=None, output_final_state=False, feature_map_fn=None, clamp_min=None)`
   - 子任務：
     - 逐 token 遞推，維護 recurrent state（同官方 shape，例如 `[B,H,head_v_dim,head_v_dim]` 或等效表示）。
     - 允許 `feature_map_fn`、clamp/min、`gate_logit_normalizer` 等參數。

### 3.4 工具 / Feature map 函式
- `libs/myfla/layers/utils.py` 需確認 `get_unpad_data/index_first_axis/pad_input` 與官方一致，特別是 varlen/mask 分支。
- `ACT2FN` 需涵蓋 `swish/silu/relu/gelu/identity`，供 feature map 與 gate_fn 使用。

### 3.5 前處理流程整合（Forward 內部邏輯）
1. **Mask/unpad → varlen**（對應 `gla.py:209-240`）
   - 子任務：透過 `attention_mask` 呼叫 `get_unpad_data` 取得 `indices`、`cu_seqlens`、`max_len`，並用 `index_first_axis` 將資料改為 varlen 格式。
2. **短卷積 + cache 注入**（`gla.py:240-284`）
   - 子任務：若 `past_key_values` 內有該層資料，取出 `conv_state_q/k/v` 與 `recurrent_state`；對 `q_proj/k_proj/v_proj` 的結果呼叫 `ShortConvolution`。
3. **Mode 選擇與 feature map**（`gla.py:284-319`）
   - 子任務：若 `seq_len <= 64` 強制 `mode='fused_recurrent'`；否則依 `self.mode` 選擇 chunk/fused_chunk。
4. **GLA ops 呼叫**（`gla.py:320-335`）
   - 子任務：將 q/k/v/gk（含 feature map、clamp）輸入對應 ops，取得 `(outputs, new_state)`。
5. **Gate + Norm**（`gla.py:338-357`）
   - 子任務：若 `use_output_gate`，決定走 `FusedRMSNormGated` 或 `RMSNorm + gate_fn` 路徑；`gate_fn` 取自 `ACT2FN`。
6. **還原 / 投影 / Cache 更新**（`gla.py:357-371`）
   - 子任務：若之前 unpad 過，呼叫 `pad_input` 還原 shape；最後 `o_proj` 轉為 `[B,L,hidden_size]`；依條件呼叫 `past_key_values.update(...)` 並回傳 `(output, None, past_key_values)`。

### 3.6 測試命令（供 Step 4 使用）
- 單元測試：`PYTHONPATH=src python3.8 tests/myfla/test_gla.py`
- 整合測試：`PYTHONPATH=src python3.8 tests/myfla/test_fla_encoder_strategy_integration.py`
- 冒煙（可選）：`PYTHONPATH=src python3.8 src/cfg/cfg_hf/cfg_setE_gla.py`

> 完成上述所有子任務後，即可進入 Step 4 進行 TDD 與整合測試，確保 myfla GLA 行為與官方版本完全對應。

---
**後續更新節點**：完成 Step 2 後補充資料流推導與測試樣本；完成 Step 3-5 後依序填入實作細節與驗收結果。

## 4. Step 4 — 單元測試與驗證策略（待執行）
- **測試檔案**：`tests/myfla/test_gla.py`
- **場景覆蓋**：
  1. Basic chunk 模式（無 mask/short conv）驗證輸出 shape、`recurrent_state` 不為 None。
  2. Mask + 短卷積：左側 padding 需於 pad_input 還原後保持 0；cache tuple 應為三個 tensor。
  3. Varlen (`cu_seqlens`) + `feature_map='relu'`：varlen 與標準模式結果應 allclose。
  4. Fused recurrent 模式（`L<=64` 或 `mode='fused_recurrent'`）：兩次 forward 後，cache 續接正常。
- **命令**：`PYTHONPATH=src python3.8 tests/myfla/test_gla.py`

## 5. Step 5 — 整合冒煙（待執行）
- 擴充 `FLAEncoderFactory`（`name='gla'`），在 `tests/myfla/test_fla_encoder_strategy_integration.py` 中驗證：
  - 工廠可建立 GLA encoder 並完成前向。
  - 多層 `past_key_values` 串接、`layer_idx` 變化、`use_cache` 續接。
- 端到端冒煙（可選）：若有 SetE 專用 cfg，執行 `PYTHONPATH=src python3.8 src/cfg/cfg_hf/cfg_setE_gla.py`；如無則以最小 dataloader + encoder 執行 smoke test。

## 6. Step 6 — 驗收與差異紀錄（待執行）
- 完成標準：
  1. `libs/myfla/layers/gla.py` 與官方邏輯完全一致；若存在差異（如 Fused kernel 只用 PyTorch fallback），需明確記錄。
  2. Step 4 的 TDD、Step 5 的整合冒煙皆通過，並紀錄執行時間與命令。
  3. 於本章節列出已知差異表：

    | 類別 | 描述 | 影響與 workaround |
    |------|------|-------------------|
    | Fused kernel | 例：PyTorch fallback 取代 Triton | 註明效能差異或警告 |
    | 功能缺口 | 例：FusedRMSNormGated 尚缺 prenorm/residual | 說明後續計畫 |
    | 測試差異 | 例：無官方 fixture；僅依 pseudo-fixture | 指定來源 |

- 若以上全部完成，即可在文末更新狀態為「已驗收」，並回填至 `plan/prd_myfla_port.md` 的總覽。
