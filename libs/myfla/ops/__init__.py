"""myfla.ops — collection of pure PyTorch operators (ported from libs.fla.ops)."""

from .kda.chunk import chunk_kda  # Stage 2.5 完整復刻 chunk 模式

__all__ = [
    'chunk_kda',
]
