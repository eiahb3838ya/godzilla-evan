"""myfla：libs.fla 的 PyTorch fallback（無 Triton 環境）。"""

from .layers import LoRA, RWKV7Attention

__version__ = "0.4.0-shim"

__all__ = [
    'LoRA',
    'RWKV7Attention',
]
