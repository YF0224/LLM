import torch
import torch.nn as nn
from einops import einsum
import math
from jaxtyping import Bool, Float, Int
from typing import Optional
from torch import Tensor

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        nn.init.trunc_normal_(self.embedding, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor)->torch.Tensor:
        return self.embedding[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # g_i，初始化为 1
        self.gain = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        rms = torch.sqrt(
            torch.mean(x * x, dim=-1, keepdim=True) + self.eps
        )
        return x / rms * self.gain

def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # W1, W3: (d_ff, d_model)  => Linear(d_model -> d_ff)
        # W2:     (d_model, d_ff)  => Linear(d_ff   -> d_model)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        x1 = self.w1(x)            # (..., d_ff)
        x3 = self.w3(x)            # (..., d_ff)
        gated = SiLU(x1) * x3      # (..., d_ff)  elementwise
        return self.w2(gated)      # (..., d_model)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires d_k to be even."

        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)

        # k_idx corresponds to 2k in the paper (0,2,4,...)
        k_idx = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)  # (d_k/2,)
        # inv_freq = Theta^{-2k/d}  (common form)
        inv_freq = self.theta ** (-k_idx / d_k)  # (d_k/2,)

        # Precompute angles for positions 0..max_seq_len-1
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)  # (max_seq_len,)
        angles = positions[:, None] * inv_freq[None, :]  # (max_seq_len, d_k/2)

        cos = torch.cos(angles)  # (max_seq_len, d_k/2)
        sin = torch.sin(angles)  # (max_seq_len, d_k/2)

        # buffers (not learnable)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions:
        - (seq_len,)  OR
        - (..., seq_len) matching x's batch dims
        """
        *batch_dims, seq_len, d_k = x.shape
        assert d_k == self.d_k, f"Expected d_k={self.d_k}, got {d_k}"

        # Accept token_positions as (seq_len,) and broadcast across batch dims
        if token_positions.ndim == 1:
            assert token_positions.shape[0] == seq_len, (
                f"token_positions must have length seq_len={seq_len}, got {token_positions.shape}"
            )
            token_positions = token_positions.view((1,) * len(batch_dims) + (seq_len,))
        else:
            # If provided with batch dims, they must match x
            assert token_positions.shape[-1] == seq_len, (
                f"token_positions last dim must be seq_len={seq_len}, got {token_positions.shape}"
            )
            # Allow broadcasting batch dims too (more tolerant than strict equality)
            # e.g. token_positions could be (1, seq_len) with x (B, seq_len)
            # So we don't strictly assert token_positions.shape == (*batch_dims, seq_len)

        cos = self.cos[token_positions]  # (..., seq_len, d_k/2) via broadcasting
        sin = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out
    
def Softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def ScaledDotProductAttention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Optional[Bool[Tensor, " ... queries keys"]] = None,
) -> Float[Tensor, " ... queries d_v"]:
    # scores: (..., queries, keys)
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(Q.shape[-1])

    if mask is not None:
        # mask=True means keep, mask=False means block
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    attn = Softmax(scores, dim=-1)  # normalize over keys
    out = einsum(attn, V, "... q k, ... k dv -> ... q dv")
    return out

class MultiheadSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal mask.
    - Optional RoPE (applied to Q/K only)
    - Projection weights are provided externally and stored as buffers
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_proj_weight: Tensor,  # (d_model, d_in) == (H*Dh, d_in)
        k_proj_weight: Tensor,  # (d_model, d_in)
        v_proj_weight: Tensor,  # (d_model, d_in)
        o_proj_weight: Tensor,  # (d_model, d_model) == (d_model, H*Dh)
        rope: Optional[RotaryPositionalEmbedding] = None,  # RotaryPositionalEmbedding(theta, d_k=Dh, max_seq_len, ...)
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        if rope is not None:
            assert self.d_head % 2 == 0, "RoPE requires per-head dim (d_head) to be even."
        self.rope = rope

        # adapter 传入权重 -> buffer（默认不训练）
        self.register_buffer("q_proj_weight", q_proj_weight)
        self.register_buffer("k_proj_weight", k_proj_weight)
        self.register_buffer("v_proj_weight", v_proj_weight)
        self.register_buffer("o_proj_weight", o_proj_weight)

    def forward(self, in_features: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        """
        in_features: (..., T, d_in)
        token_positions:
          - None -> use 0..T-1
          - (T,) or (..., T)
        returns: (..., T, d_model)
        """
        *prefix, T, d_in = in_features.shape
        H, Dh = self.num_heads, self.d_head

        if token_positions is None:
            token_positions = torch.arange(T, device=in_features.device, dtype=torch.long)

        # ---- 1) Q K V projections (one matmul per Q/K/V for all heads) ----
        Q = in_features @ self.q_proj_weight.t()  # (..., T, d_model)
        K = in_features @ self.k_proj_weight.t()  # (..., T, d_model)
        V = in_features @ self.v_proj_weight.t()  # (..., T, d_model)

        # ---- 2) reshape into heads: (..., H, T, Dh) ----
        Q = Q.view(*prefix, T, H, Dh).transpose(-3, -2)
        K = K.view(*prefix, T, H, Dh).transpose(-3, -2)
        V = V.view(*prefix, T, H, Dh).transpose(-3, -2)

        # ---- 3) optional RoPE on Q/K ----
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # ---- 4) causal masked attention ----
        scores = (Q @ K.transpose(-1, -2)) / math.sqrt(Dh)  # (..., H, T, T)

        causal = torch.tril(torch.ones(T, T, device=in_features.device, dtype=torch.bool))
        mask = causal.view(*([1] * len(prefix)), 1, T, T)  # (..., 1, T, T) -> broadcast to (..., H, T, T)

        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ V  # (..., H, T, Dh)

        # ---- 5) merge heads + output projection ----
        out = out.transpose(-3, -2).contiguous().view(*prefix, T, H * Dh)  # (..., T, d_model)
        out = out @ self.o_proj_weight.t()  # (..., T, d_model)
        return out
    
class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block:
      x = x + Attn(RMSNorm(x))
      x = x + FFN(RMSNorm(x))
    RoPE is handled inside attention via `rope` argument.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: Optional[nn.Module] = None,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)

        # Init dummy weights; adapter will overwrite via .copy_()
        Wq = torch.empty(d_model, d_model, device=device, dtype=dtype)
        Wk = torch.empty(d_model, d_model, device=device, dtype=dtype)
        Wv = torch.empty(d_model, d_model, device=device, dtype=dtype)
        Wo = torch.empty(d_model, d_model, device=device, dtype=dtype)
        nn.init.zeros_(Wq); nn.init.zeros_(Wk); nn.init.zeros_(Wv); nn.init.zeros_(Wo)

        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            q_proj_weight=Wq,
            k_proj_weight=Wk,
            v_proj_weight=Wv,
            o_proj_weight=Wo,
            rope=rope,
        )

        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    """
    Transformer language model (RoPE):
      token_embeddings -> N * TransformerBlock -> ln_final -> lm_head
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # token embeddings (no learned position embeddings when using RoPE)
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # one shared RoPE module (safe to share across layers)
        d_head = d_model // num_heads
        rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=d_head, max_seq_len=context_length, device=device)

        # transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope=rope,
                eps=eps,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # final norm
        self.ln_final = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)

        # lm head: d_model -> vocab_size
        # Linear.W shape: (out=vocab_size, in=d_model)  matches `lm_head.weight` in reference
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        in_indices: (B, T) with T <= context_length
        returns: (B, T, vocab_size) logits
        """
        B, T = in_indices.shape
        assert T <= self.context_length, f"sequence_length {T} exceeds context_length {self.context_length}"

        token_positions = torch.arange(T, device=in_indices.device, dtype=torch.long)

        x = self.token_embeddings(in_indices)  # (B, T, d_model)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.ln_final(x)  # (B, T, d_model)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits
