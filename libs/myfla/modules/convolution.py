from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortConvolution(nn.Module):
    """
    PyTorch 版短卷積：採深度可分離 conv1d，手動處理 causal padding 與 cache。
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = 'silu',
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=0,
        )
        if activation is None:
            self.activation = None
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation {activation}")

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass of ShortConvolution.

        Source: libs/fla/modules/convolution.py:L888-L963

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, D] or [1, total_tokens, D] when cu_seqlens is provided
            residual (torch.Tensor, optional): Residual tensor of shape [B, T, D]
            mask (torch.Tensor, optional): Mask tensor of shape [B, T]. Mutually exclusive with cu_seqlens.
            cache (torch.Tensor, optional): Cache tensor of shape [B, D, W] or [N, D, W] when cu_seqlens is provided
            output_final_state (bool): Whether to output final state for caching
            cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths tensor of shape [N+1] for
                variable-length sequences. When provided, input x should have shape [1, total_tokens, D]
                where total_tokens is the sum of all sequence lengths. Mutually exclusive with mask.

                Example: For 3 sequences with lengths [5, 3, 7], cu_seqlens = [0, 5, 8, 15].

        Returns:
            output (torch.Tensor): Output tensor of shape [B, T, D] or [1, total_tokens, D]
            cache (torch.Tensor | None): Cache tensor of shape [B, D, W] or [N, D, W], or None
        """
        # Compute number of sequences (Source: fla line 919-920)
        B, T, *_ = x.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1

        # Mutual exclusivity check (Source: fla line 921-924)
        if mask is not None and cu_seqlens is not None:
            raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")

        # Varlen mode: process each sequence independently (Source: fla line 919-963)
        if cu_seqlens is not None:
            outputs = []
            new_caches = [] if output_final_state else None

            for i in range(N):
                # Extract sequence boundaries
                start, end = cu_seqlens[i].item(), cu_seqlens[i+1].item()
                seq_len = end - start

                # Handle empty sequences
                if seq_len == 0:
                    if output_final_state:
                        new_caches.append(torch.zeros(
                            self.hidden_size, self.kernel_size,
                            dtype=x.dtype, device=x.device
                        ))
                    continue

                seq = x[0, start:end, :]  # [T_i, D]

                # Extract corresponding cache (if any)
                seq_cache = cache[i:i+1] if cache is not None else None  # [1, D, W]

                # Extract corresponding residual (if any)
                seq_residual = None
                if residual is not None:
                    seq_residual = residual[0, start:end, :].unsqueeze(0)  # [1, T_i, D]

                # Process single sequence
                seq = seq.unsqueeze(0)  # [1, T_i, D]
                seq_out, seq_new_cache = self._process_single_sequence(
                    seq, seq_residual, seq_cache, output_final_state
                )

                outputs.append(seq_out.squeeze(0))  # [T_i, D]
                if output_final_state:
                    new_caches.append(seq_new_cache.squeeze(0))  # [D, W]

            # Merge results
            y = torch.cat(outputs, dim=0).unsqueeze(0)  # [1, total_tokens, D]
            final_cache = torch.stack(new_caches, dim=0) if new_caches else None  # [N, D, W]

            return y, final_cache
        else:
            # Standard mode: direct call to single sequence processing
            return self._process_single_sequence(x, residual, cache, output_final_state, mask)

    def _process_single_sequence(
        self,
        x: torch.Tensor,  # [B, T, D]
        residual: torch.Tensor | None,  # [B, T, D] or None
        cache: torch.Tensor | None,  # [B, D, W] or None
        output_final_state: bool,
        mask: torch.Tensor | None = None,  # [B, T] or None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Process a single sequence (or batch of sequences) with convolution.

        Args:
            x: Input tensor [B, T, D]
            residual: Optional residual tensor [B, T, D]
            cache: Optional cache tensor [B, D, W]
            output_final_state: Whether to output final state
            mask: Optional mask tensor [B, T]

        Returns:
            output: Output tensor [B, T, D]
            cache: Optional cache tensor [B, D, W]
        """
        b, t, d = x.shape
        xx = x.transpose(1, 2)  # [B, D, T]
        if mask is not None:
            xx = xx * mask.transpose(1, 2)
        if cache is None:
            padded = F.pad(xx, (self.kernel_size - 1, 0))
        else:
            padded = torch.cat([cache, xx], dim=-1)
        y = self.conv(padded)
        if self.activation is not None:
            y = self.activation(y)
        if residual is not None:
            y = y + residual.transpose(1, 2)
        y = y[..., -t:]
        out = y.transpose(1, 2)
        new_cache = None
        if output_final_state:
            new_cache = padded[..., -self.kernel_size:].detach()
        return out, new_cache

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size
