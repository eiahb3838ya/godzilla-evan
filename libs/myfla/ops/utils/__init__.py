"""myfla.ops.utils — 純 PyTorch 版 utilities（源自 `libs/fla/ops/utils`)."""

from .cumsum import (
    chunk_global_cumsum,
    chunk_global_cumsum_scalar,
    chunk_global_cumsum_vector,
    chunk_local_cumsum,
    chunk_local_cumsum_scalar,
    chunk_local_cumsum_vector,
)
from .index import (
    get_max_num_splits,
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_cu_seqlens_from_lens,
    prepare_cu_seqlens_from_mask,
    prepare_lens,
    prepare_lens_from_cu_seqlens,
    prepare_lens_from_mask,
    prepare_position_ids,
    prepare_sequence_ids,
    prepare_split_cu_seqlens,
    prepare_token_indices,
)
from .pack import pack_sequence, unpack_sequence
from .solve_tril import solve_tril

__all__ = [
    'chunk_global_cumsum',
    'chunk_global_cumsum_scalar',
    'chunk_global_cumsum_vector',
    'chunk_local_cumsum',
    'chunk_local_cumsum_scalar',
    'chunk_local_cumsum_vector',
    'get_max_num_splits',
    'pack_sequence',
    'prepare_chunk_indices',
    'prepare_chunk_offsets',
    'prepare_cu_seqlens_from_lens',
    'prepare_cu_seqlens_from_mask',
    'prepare_lens',
    'prepare_lens_from_cu_seqlens',
    'prepare_lens_from_mask',
    'prepare_position_ids',
    'prepare_sequence_ids',
    'prepare_split_cu_seqlens',
    'prepare_token_indices',
    'solve_tril',
    'unpack_sequence',
]
