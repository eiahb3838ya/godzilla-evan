# myfla × fla 檔案對照表（2025-11-17）

依 `plan/prd_myfla_port.md` Step 1，列出 `libs/fla/layers` 與 `libs/myfla/layers` 的檔案映射，並將缺失項目分類為「必須復刻」（factory 或 PRD 已規畫使用）與「尚未使用」（暫未排程，但需建立骨架）。

| fla 檔案 | myfla 狀態 | 分類 | 備註 |
| --- | --- | --- | --- |
| `abc.py` (`ABCAttention`) | stub 已建立 | 尚未使用 | 僅作為共用抽象層 |
| `attn.py` (`Attention`) | stub 已建立 | **必須復刻** | Factory 入口將直接引用 |
| `based.py` (`BasedLinearAttention`) | stub | 尚未使用 | - |
| `bitattn.py` (`BitAttention`) | stub | 尚未使用 | - |
| `comba.py` (`Comba`) | stub | 尚未使用 | - |
| `delta_net.py` (`DeltaNet`) | stub | 尚未使用 | - |
| `deltaformer.py` (`DeltaFormerAttention`) | stub | **必須復刻** | FLAEncoderFactory 待擴充目標 |
| `forgetting_attn.py` (`ForgettingAttention`) | stub | 尚未使用 | - |
| `gated_deltanet.py` | ✅ 已實作 | **必須復刻** | myfla 版本正在復刻流程 |
| `gated_deltaproduct.py` (`GatedDeltaProduct`) | stub | 尚未使用 | - |
| `gla.py` (`GatedLinearAttention`) | stub | **必須復刻** | 對應 Level 4 擴充 |
| `gsa.py` (`GatedSlotAttention`) | stub | **必須復刻** | Level 5 目標 |
| `hgrn.py` (`HGRNAttention`) | stub | 尚未使用 | - |
| `hgrn2.py` (`HGRN2Attention`) | stub | 尚未使用 | - |
| `kda.py` (`KimiDeltaAttention`) | stub | 尚未使用 | - |
| `lightnet.py` (`LightNetAttention`) | stub | 尚未使用 | - |
| `linear_attn.py` (`LinearAttention`) | stub | 尚未使用 | - |
| `log_linear_mamba2.py` (`LogLinearMamba2`) | stub | 尚未使用 | - |
| `mamba.py` (`Mamba`) | stub | 尚未使用 | - |
| `mamba2.py` (`Mamba2`) | stub | 尚未使用 | - |
| `mesa_net.py` (`MesaNet`) | stub | 尚未使用 | - |
| `mla.py` (`MultiheadLatentAttention`) | stub | 尚未使用 | - |
| `mom.py` (`MomAttention`) | stub | 尚未使用 | - |
| `multiscale_retention.py` (`MultiScaleRetention`) | stub | 尚未使用 | - |
| `nsa.py` (`NativeSparseAttention`) | stub | 尚未使用 | - |
| `path_attn.py` (`PaTHAttention`) | stub | 尚未使用 | - |
| `rebased.py` (`ReBasedLinearAttention`) | stub | 尚未使用 | - |
| `rodimus.py` (`RodimusAttention`, `SlidingWindowSharedKeyAttention`) | stub | 尚未使用 | - |
| `rwkv6.py` (`LoRA`) | ✅ 已實作 | **必須復刻** | 純 PyTorch 版 LoRA，無額外 shim |
| `rwkv7.py` (`RWKV7Attention`) | ✅ 已實作 | **必須復刻** | Level 1 已使用 |
| `simple_gla.py` (`SimpleGatedLinearAttention`) | stub | 尚未使用 | - |
| `utils.py`（helpers） | ✅ 已實作 | **必須復刻** | `get_unpad_data/index_first_axis/pad_input/unpad_input` PyTorch 版 + `tests/myfla/test_kda_utils.py` |

> 目前 `libs/myfla/layers/` 僅保留與官方同名檔案；任何額外實驗性元件已移除或合併。

## Modules（核心模塊）

| fla 檔案 | myfla 檔案 | 狀態 | 備註 |
| --- | --- | --- | --- |
| `modules/layernorm.py`（GroupNormRef/GroupNorm/RMSNorm/FusedRMSNormGated） | `libs/myfla/modules/layernorm.py` | ✅ 已實作 | 集中提供 layer_norm_ref、group_norm_ref、rms_norm_ref 與對應類別 |
| `modules/l2norm.py` | `libs/myfla/modules/l2norm.py` | ✅ 已實作 | head-wise L2 normalize |
| `modules/token_shift.py` | `libs/myfla/modules/token_shift.py` | ✅ 已實作 | 支援 batch、varlen、cache |
| `modules/convolution.py:ShortConvolution` | `libs/myfla/modules/convolution.py` | ✅ 已實作 | 深度可分離 causal conv，輸出 cache |
| `modules/activations.py` | 無 | 尚未移植 | 目前未直接使用 |
| `modules/feature_map.py`…（其餘 fla modules） | 無 | 尚未移植 | 待後續需求再補 |

## Ops（具體算子）

| fla 檔案 | myfla 檔案 | 狀態 | 備註 |
| --- | --- | --- | --- |
| `ops/generalized_delta_rule/**` | `libs/myfla/ops/generalized_delta_rule/**` | ✅ 已實作 | PyTorch 版 DPLR/ILPR chunk + fused |
| `ops/gated_delta_rule/chunk.py` | `libs/myfla/ops/gated_delta_rule/chunk.py` | ✅ 完美復刻 | **已對齊官方 API**：`chunk_gated_delta_rule` 含 `torch.autograd.Function` 封裝；前/反向完全對應 Triton kernel 邏輯 |
| `ops/gated_delta_rule/fused_recurrent.py` | `libs/myfla/ops/gated_delta_rule/fused_recurrent.py` | ✅ 完美復刻 | **已對齊官方 API**：`fused_recurrent_gated_delta_rule` 含 state 管理/cache 續接，純 PyTorch 實現 |
| `ops/rwkv7/chunk.py` | `libs/myfla/ops/rwkv7/chunk.py` | ✅ 已實作 | 控制 chunk vs fused 路徑 |
| `ops/rwkv7/fused_addcmul.py` | `libs/myfla/ops/rwkv7/fused_addcmul.py` | ✅ 已實作 | PyTorch fallback |
| `ops/rwkv7/fused_k_update.py` | `libs/myfla/ops/rwkv7/fused_k_update.py` | ✅ 已實作 | PyTorch fallback |
| `ops/rwkv7/gate_output_correction.py` | `libs/myfla/ops/rwkv7/gate_output_correction.py` | ✅ 已實作 | PyTorch fallback |
| `ops/utils/index.py` + `__init__.py` | `libs/myfla/ops/utils/index.py` | ✅ 已實作 | **必須復刻**；Stage 1.2 完成 `prepare_lens/*` 系列 + `tests/myfla/test_ops_utils_index.py` |
| `ops/utils/cumsum.py` | `libs/myfla/ops/utils/cumsum.py` | ✅ 已實作 | Stage 1.3：`chunk_local/global_cumsum*` 純 PyTorch 版 + `tests/myfla/test_ops_utils_cumsum.py` |
| `ops/utils/solve_tril.py` | `libs/myfla/ops/utils/solve_tril.py` | ✅ 已實作 | Stage 1.4：`solve_tril` PyTorch 版（chunk + varlen），`tests/myfla/test_ops_utils_solve_tril.py` |
| `ops/utils/op.py` | `libs/myfla/ops/utils/op.py` | ✅ 已實作 | `exp/log/log2/exp2/safe_exp` + 簡易 `make_tensor_descriptor`，供 KDA kernel 導入 |
| `ops/utils/pack.py` | `libs/myfla/ops/utils/pack.py` | ✅ 已實作 | PyTorch fallback（pack/unpack_sequence），提供 KDA varlen pipeline 鉤子 |
| 其餘 fla ops（GSA、GLA、MesaNet…） | 無 | 尚未移植 | 待需求再補 |

## Models（Transformers / Delta 系列）

| fla 模型 | myfla 對應 | 狀態 | 備註 |
| --- | --- | --- | --- |
| `models/rwkv7/*`（config/modeling） | 無 | 尚未移植 | 目前只在 cfg 中直接使用 encoder，不透過 huggingface-style model |
| `models/gated_deltanet/*` | 無 | 尚未移植 | 同上；暫無自動化建模需求 |
| `models/gla/*`, `models/kda/*`, `models/mla/*`, etc. | 無 | 尚未移植 | 若未來要用 huggingface-style wrapper 再另外補 |

> 結論：myfla 目前聚焦於 `layers/modules/ops` 的純 PyTorch 落地；`models/*`（配置/建模層）尚未復刻，就算使用這些編碼器，仍透過 `FLAEncoderFactory`/`FLAFinancialModel` 來組裝。
