# KDA Ops 依賴鏈分析
> 分析日期：2025-11-27  
> 來源：`libs/fla/ops/kda/chunk.py:L1-L357`  
> 原則：禁止任何簡化/MVP，加總依賴僅允許官方逐行復刻。以下狀態已同步 `plan/fla/prd_kda.plan.md`（Stage 2.6）。

## ✅ Stage 1 依賴（全部完成）

| 函式 | 來源 | 狀態 | 用途 |
|------|------|------|------|
| `chunk_local_cumsum` | `fla.ops.utils` | ✅ | gate 的 chunk-wise 前綴和 |
| `chunk_gated_delta_rule_fwd_h` | `fla.ops.common.chunk_delta_h` | ✅ | Gated Delta Rule forward 狀態 |
| `chunk_gated_delta_rule_bwd_dhu` | `fla.ops.common.chunk_delta_h` | ✅ | Gated Delta Rule backward 梯度 |
| `chunk_bwd_dv_local` | `fla.ops.common.chunk_o` | ✅ | chunk dv 梯度損益 |
| `l2norm_fwd` / `l2norm_bwd` | `fla.modules.l2norm` | ✅ | `use_qk_l2norm_in_kernel` 專用 |

## ✅ Stage 2 依賴現況（2025-11-27）

### 2.1 `chunk_intra.py` — Intra-chunk Attention
- **實作狀態**：`chunk_kda_fwd_intra`、`chunk_kda_bwd_intra` 及其 kernel 皆於 2025-11-25 逐行移植完成，採 fp32 累積 + 官方註解。
- **Varlen**：2025-11-27 起改用 `_build_sequence_infos` + chunk offset 寫法（不再逐序列切片），Aqk/Akk 及 dq/dk/db/dg 均依 chunk 內位置寫回；`tests/myfla/test_kda_ops_chunk.py::TestKDAIntraVarlen` forward/backward 重新通過。
- **剩餘事項**：待 Stage 2.6 擴充 gradcheck 及多頭/multi-chunk smoke，並評估 `chunk_inter` 的 varlen 需求。

### 2.2 `wy_fast.py` — WY 表示（Woodbury）
- **實作狀態**：`recompute_w_u_fwd`、`prepare_wy_repr_bwd` 及其 helper 已 1:1 轉寫，含 solve_tril、Woodbury 身份式。
- **Varlen**：同樣採 per-seq 切片，`TestKDAIntraVarlen` 檢查 w/u/qg/kg、dk/dv/dbeta/dg/dA 與切片結果一致。
- **未決議題**：暫無阻塞；僅需 Stage 2.6 gradcheck/多頭 smoke 例行驗證。

### 2.3 `chunk_inter.py` — Inter-chunk Backward
- **實作狀態**：`chunk_kda_bwd_dqkwg` 完整實作 dw 負號、複雜 cumsum-based dg 累積。
- **Varlen**：尚未接入 `cu_seqlens` 分支，預定於 Stage 2.6 在 cache/multi-chunk 測試中補齊。

### 2.4 `gla/chunk.py` — `chunk_gla_fwd_o_gk` / `chunk_gla_bwd_dA`
- **實作狀態**：forward/backward kernel 已以純 PyTorch 實現，保留 chunk 遮罩、fp32 累積與官方註釋；禁止任何簡化版本，稽核完成。
- **Varlen 支援**：`_iter_chunk_spans` 於 2025-11-27 重寫，支援 flatten 與 per-batch `cu_seqlens`；`chunk_gla_fwd_o_gk`/`chunk_gla_bwd_dA` 依序列本地 chunk index 讀取 `h[b, chunk_idx]`，並在 fp32 buffer 完成 mask 後轉回輸入 dtype。
- **測試狀態**：`tests/myfla/test_kda_ops_chunk.py::TestGLAChunk` 所有案例（含 gradcheck、varlen）現為綠燈；`PYTHONPATH=src python3.8 tests/myfla/test_kda_ops_chunk.py` 僅剩 cache 續接測試失敗。

### 2.5 `chunk.py` — ChunkKDA 主入口
- **實作狀態**：`chunk_kda_fwd`、`chunk_kda_bwd`、`ChunkKDAFunction`、`chunk_kda` 均已串聯 Stage 1/2 依賴並支援 `use_qk_l2norm_in_kernel`、`output_final_state`。
- **Varlen/Cache**：API 已接受 `cu_seqlens` 與 cache state；`chunk_gated_delta_rule_fwd_h` 在 varlen 場景可回傳 per-seq final state。
- **回歸**：`TestChunkKDAFunction::test_chunk_kda_cache_continuation`（partial-run + resume）結果不一致；需檢查 ctx 中存放的 qg/kg、`final_state` 取值，以及 `chunk_kda_fwd_intra`/`chunk_kda_bwd_intra` 的 varlen 切片是否破壞狀態。

## 🔬 Stage 2.6 測試矩陣（進行中）

| 測試模組 | 覆蓋內容 | 目前狀態 |
|----------|-----------|----------|
| `TestGLAChunk` | `chunk_gla_fwd_o_gk` / `chunk_gla_bwd_dA` forward + gradcheck（固定長、varlen） | ✅ 通過（varlen chunk 對齊已修復） |
| `TestKDAIntraVarlen` | `chunk_kda_fwd_intra`、`chunk_kda_bwd_intra`、`recompute_w_u_fwd`、`prepare_wy_repr_bwd` varlen vs. slice | ✅ 通過（column offset 依 chunk 起點寫回） |
| `TestChunkKDAFunction` | `ChunkKDAFunction` forward/backward、`use_qk_l2norm_in_kernel` 切換、cache 續接 | ⚠️ cache continuation 失敗（partial-run vs resume） |
| `TestChunkGatedDeltaRuleVarlen` | Stage 1 delta rule varlen baseline | ✅ 通過，作為比對基準 |

**已知失敗案例**：
1. `chunk_kda_cache_continuation`：cache state 無法與 partial-run + resume 對齊（`TestChunkKDAFunction` 單一失敗）。

**下一步（不得簡化）**：
1. **修復 cache 續接**：在 `ChunkKDAFunction`/`chunk_kda` 中保留尚未湊滿 64 token 的 chunk（或 global chunk offset），使 partial-run + resume 能與 full-run 共用相同 `chunk_local_cumsum`/`chunk_kda_fwd_intra` 列索引；同步更新 `KimiDeltaAttention` 的 cache 結構。
2. **擴充矩陣**：新增 multi-head/multi-chunk/varlen smoke、factory smoke、chunk-level gradcheck；維持 chunk_size=64。

## 依賴鏈（更新版）

### chunk_kda_fwd
```
chunk_kda_fwd (L17-L69)
  ├── chunk_local_cumsum(g)                    ✅ Stage 1
  ├── chunk_kda_fwd_intra(q, k, g, beta)      ✅ Stage 2.1（varlen offset 修中）
  │    ├── _fwd_kernel_intra_sub_inter
  │    └── _fwd_kernel_intra_sub_intra
  ├── recompute_w_u_fwd(k, v, beta, Akk, g)   ✅ Stage 2.2
  ├── chunk_gated_delta_rule_fwd_h(kg, w, u)  ✅ Stage 1（varlen ok）
  └── chunk_gla_fwd_o_gk(q, v_new, g, Aqk, h) ✅ Stage 2.4（varlen chunk idx 修中）
```

### chunk_kda_bwd
```
chunk_kda_bwd (L72-L176)
  ├── recompute_w_u_fwd(...)                   ✅ Stage 2.2
  ├── chunk_gated_delta_rule_fwd_h(...)        ✅ Stage 1
  ├── chunk_bwd_dv_local(...)                  ✅ Stage 1
  ├── chunk_gated_delta_rule_bwd_dhu(...)      ✅ Stage 1
  ├── chunk_gla_bwd_dA(...)                    ✅ Stage 2.4（dtype/device 修中）
  ├── chunk_kda_bwd_dqkwg(...)                 ✅ Stage 2.3（varlen branch TODO）
  ├── prepare_wy_repr_bwd(...)                 ✅ Stage 2.2
  └── chunk_kda_bwd_intra(...)                 ✅ Stage 2.1（varlen slice 修中）
```

## 近期成果與稽核
- 2025-11-27：完成 `libs/myfla/ops/gla/chunk.py`、`libs/myfla/ops/kda/{chunk_intra,wy_fast,chunk_inter,chunk.py}`、`tests/myfla/test_kda_ops_chunk.py` 全面稽核，確認無簡化/placeholder/fallback。
- 2025-11-27：`chunk_gated_delta_rule_*`、`chunk_intra`、`wy_fast` 皆支援 `cu_seqlens`，並於測試中對照切片版本。
- 2025-11-27：`ChunkKDAFunction` 支援 `use_qk_l2norm_in_kernel`，cache state 亦可輸入/輸出；目前僅剩續接測試需調整。

## TODO / 阻塞（Stage 2.6）
1. **Varlen 修復**：`chunk_kda_fwd_intra`/`chunk_kda_bwd_intra` column offset、`chunk_gla_fwd_o_gk` chunk 映射、`chunk_gla_bwd_dA` dtype/device、`chunk_kda_bwd_dqkwg` varlen 分支。
2. **Cache 續接 QA**：`chunk_kda_cache_continuation` 需與 partial-run + resume 結果相同，並記錄 state 轉移。
3. **Gradcheck 擴大**：在短序列設定對 `ChunkKDAFunction` 做 gradcheck，驗證 dq/dk/dv/db/dg/dh0。
4. **整合測試**：新增 `tests/myfla/test_fla_encoder_strategy_integration.py` KDA case，並以 factory 驗證設定。
5. **文檔同步**：修復完成後同時更新本檔與 `plan/fla/prd_kda.plan.md`，維持「無簡化、逐行復刻」註記。
