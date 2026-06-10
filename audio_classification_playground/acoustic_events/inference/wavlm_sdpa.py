"""Repo-owned SDPA patch for HuggingFace WavLM attention.

WavLM is not migrated to the transformers attention interface (``_supports_sdpa
= False`` even on ``main``) because of its gru-gated relative-position bias, so
attention runs unfused via ``F.multi_head_attention_forward``. That gated bias
is just an additive ``[bsz*heads, L, L]`` tensor, which folds directly into
``F.scaled_dot_product_attention``'s ``attn_mask`` — routing attention through
the fused/mem-efficient kernel (FlashAttention on Ampere+ for fp16/bf16).

This swaps only ``WavLMAttention.torch_multi_head_self_attention``; the gated-bias
computation in ``WavLMAttention.forward`` is untouched, so outputs match the
stock path to floating-point tolerance. It is a class-method monkeypatch applied
at runtime (idempotent, reversible) — version-controlled here so it survives
``uv sync``, unlike editing site-packages. Apply *before* torch.compile/warmup so
the compiled graph captures the SDPA kernel.

``output_attentions=True`` falls back to the original (SDPA can't return weights).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # keep import-safe if transformers internals move
    from transformers.models.wavlm import modeling_wavlm as _wm

    _ORIG_MHSA = _wm.WavLMAttention.torch_multi_head_self_attention
except Exception:  # pragma: no cover
    _wm = None
    _ORIG_MHSA = None

_PATCHED = False


def _sdpa_mhsa(self, hidden_states, attention_mask, gated_position_bias, output_attentions):
    """Drop-in for ``WavLMAttention.torch_multi_head_self_attention`` using SDPA."""
    if output_attentions:
        return _ORIG_MHSA(self, hidden_states, attention_mask, gated_position_bias, output_attentions)

    bsz, tgt_len, _ = hidden_states.size()
    h, hd = self.num_heads, self.head_dim

    # Same projections as the stock path (separate q/k/v weights + biases).
    q = self.q_proj(hidden_states).view(bsz, tgt_len, h, hd).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, tgt_len, h, hd).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, tgt_len, h, hd).transpose(1, 2)

    # gated_position_bias: [bsz*h, L, L] -> [bsz, h, L, L]; flattening order is
    # (batch, head) in the stock code, matching this view exactly.
    attn_bias = gated_position_bias.view(bsz, h, tgt_len, tgt_len).to(q.dtype)
    if attention_mask is not None:
        # attention_mask is [bsz, L] with 1 == keep (stock uses .ne(1) as pad mask).
        key_pad = attention_mask.ne(1)
        pad = torch.zeros(bsz, 1, 1, tgt_len, dtype=q.dtype, device=q.device)
        pad = pad.masked_fill(key_pad[:, None, None, :], float("-inf"))
        attn_bias = attn_bias + pad

    # SDPA scales q by 1/sqrt(head_dim) and adds attn_bias before softmax, exactly
    # like F.multi_head_attention_forward with this attn_mask. dropout=0 at eval.
    attn_out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_bias, dropout_p=self.dropout if self.training else 0.0
    )
    attn_out = attn_out.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
    attn_out = self.out_proj(attn_out)
    return attn_out, None


def apply_wavlm_sdpa_patch() -> bool:
    """Swap WavLM attention to the SDPA implementation. Idempotent."""
    global _PATCHED
    if _wm is None or _ORIG_MHSA is None:
        return False
    if not _PATCHED:
        _wm.WavLMAttention.torch_multi_head_self_attention = _sdpa_mhsa
        _PATCHED = True
    return True


def remove_wavlm_sdpa_patch() -> None:
    """Restore the stock WavLM attention implementation."""
    global _PATCHED
    if _wm is not None and _ORIG_MHSA is not None and _PATCHED:
        _wm.WavLMAttention.torch_multi_head_self_attention = _ORIG_MHSA
        _PATCHED = False


def wavlm_sdpa_patched() -> bool:
    return _PATCHED
