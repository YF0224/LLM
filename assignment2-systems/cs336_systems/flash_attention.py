from __future__ import annotations

import math
import torch
import triton
import triton.language as tl
import functools

class FlashAttention2PyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        # Q, K, V expected shapes: (..., N, D)
        # Works for (B, H, N, D) or (B, N, D) etc.
        assert Q.shape == K.shape == V.shape, "Q,K,V must have same shape"
        *prefix, N, D = Q.shape
        assert N >= 16 and D >= 16, "tests guarantee powers of 2 and >=16"

        # tile sizes (>=16)
        Bq = 64
        Bk = 64
        scale = 1.0 / math.sqrt(D)

        # Flatten prefix dims so we can loop easily in Python
        B = 1
        for x in prefix:
            B *= x
        Q_ = Q.reshape(B, N, D)
        K_ = K.reshape(B, N, D)
        V_ = V.reshape(B, N, D)

        O_ = torch.empty((B, N, D), device=Q.device, dtype=Q.dtype)
        L_ = torch.empty((B, N), device=Q.device, dtype=torch.float32)

        # Algorithm 1 over query tiles
        for b in range(B):
            Qb = Q_[b]  # (N, D)
            Kb = K_[b]
            Vb = V_[b]

            for qi in range(0, N, Bq):
                q_end = qi + Bq
                Qi = Qb[qi:q_end, :]  # (Bq, D)

                # on-chip accumulators (fp32)
                Oi = torch.zeros((Bq, D), device=Q.device, dtype=torch.float32)
                mi = torch.full((Bq,), float("-inf"), device=Q.device, dtype=torch.float32)
                li = torch.zeros((Bq,), device=Q.device, dtype=torch.float32)

                # loop over key tiles
                for kj in range(0, N, Bk):
                    k_end = kj + Bk
                    Kj = Kb[kj:k_end, :]  # (Bk, D)
                    Vj = Vb[kj:k_end, :]  # (Bk, D)

                    # S = Qi Kj^T / sqrt(D) : (Bq, Bk), compute in fp32
                    S = (Qi.to(torch.float32) @ Kj.to(torch.float32).T) * scale

                    # (a) says you can ignore is_causal here, so we do nothing.

                    # m_new = max(m_old, rowmax(S))
                    rowmax = S.max(dim=1).values
                    m_new = torch.maximum(mi, rowmax)

                    # P_tilde = exp(S - m_new)
                    P_tilde = torch.exp(S - m_new[:, None])

                    # l_new = exp(m_old - m_new) * l_old + rowsum(P_tilde)
                    alpha = torch.exp(mi - m_new)  # (Bq,)
                    l_new = alpha * li + P_tilde.sum(dim=1)

                    # O_new = diag(exp(m_old - m_new)) O_old + P_tilde @ V
                    Oi = Oi * alpha[:, None] + (P_tilde.to(Vj.dtype) @ Vj).to(torch.float32)

                    # commit running stats
                    mi = m_new
                    li = l_new

                # finalize: O = O / l ; L = m + log(l)
                Oi = Oi / li[:, None]
                Li = mi + torch.log(li)

                # write back (cast output to original dtype)
                O_[b, qi:q_end, :] = Oi.to(Q.dtype)
                L_[b, qi:q_end] = Li

        # reshape back
        O = O_.reshape(*prefix, N, D)
        L = L_.reshape(*prefix, N)  # store fp32 logsumexp

        # save for backward later (per spec)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = bool(getattr(ctx, "is_causal", False))

        flash_bwd = _get_compiled_flash_bwd()
        dQ, dK, dV = flash_bwd(Q, K, V, O, dO, L, is_causal)
        return dQ, dK, dV, None
    

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,   # NEW
):

    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Q block ptr (given pattern)
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # O block ptr (same tiling as Q)
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # L block ptr: (N_QUERIES,) vector
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # K / V start at key tile 0; we advance inside the loop
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Load Qi once
    Qi = tl.load(Q_block_ptr).to(tl.float32)  # (Bq, D)

    # On-chip running buffers (fp32)
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)

    # key tile global start index (in the key dimension)
    k_tile_start = 0

    for _ in range(Tk):
        Kj = tl.load(K_block_ptr).to(tl.float32)
        Vj = tl.load(V_block_ptr)

        S = tl.dot(Qi, tl.trans(Kj)) * scale  # (Bq, Bk)

        if is_causal:
            # q indices for this query tile: [q_start, q_start+1, ...]
            q_start = query_tile_index * Q_TILE_SIZE
            q_idx = q_start + tl.arange(0, Q_TILE_SIZE)          # (Bq,)

            # k indices for this key tile
            k_idx = k_tile_start + tl.arange(0, K_TILE_SIZE)     # (Bk,)

            # mask: keep if q >= k (lower triangle)
            causal = q_idx[:, None] >= k_idx[None, :]            # (Bq, Bk)

            # masked-out entries add -1e6
            S = tl.where(causal, S, S + (-1e6))

        rowmax = tl.max(S, axis=1)
        m_new = tl.maximum(mi, rowmax)

        P_tilde = tl.exp(S - m_new[:, None])
        alpha = tl.exp(mi - m_new)
        l_new = alpha * li + tl.sum(P_tilde, axis=1)

        PV = tl.dot(P_tilde.to(Vj.dtype), Vj)
        Oi = Oi * alpha[:, None] + PV.to(tl.float32)

        mi, li = m_new, l_new

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
        k_tile_start += K_TILE_SIZE   # NEW


    # Final normalize
    Oi = Oi / li[:, None]
    Li = mi + tl.log(li)

    # Store
    out_ty = O_block_ptr.type.element_ty
    tl.store(O_block_ptr, Oi.to(out_ty))
    tl.store(L_block_ptr, Li)


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
        # Q,K,V: (..., N, D)
        assert Q.shape == K.shape == V.shape
        assert Q.is_cuda and K.is_cuda and V.is_cuda

        *prefix, N, D = Q.shape
        B = 1
        for x in prefix:
            B *= x

        # Flatten batch-like dims for kernel
        Q_ = Q.reshape(B, N, D).contiguous()
        K_ = K.reshape(B, N, D).contiguous()
        V_ = V.reshape(B, N, D).contiguous()

        O_ = torch.empty((B, N, D), device=Q.device, dtype=Q.dtype)
        L_ = torch.empty((B, N), device=Q.device, dtype=torch.float32)

        # Tile sizes (>=16). You can tune later.
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64

        grid = (triton.cdiv(N, Q_TILE_SIZE), B)
        scale = 1.0 / math.sqrt(D)

        flash_fwd_kernel[grid](
            Q_, K_, V_,
            O_, L_,
            Q_.stride(0), Q_.stride(1), Q_.stride(2),
            K_.stride(0), K_.stride(1), K_.stride(2),
            V_.stride(0), V_.stride(1), V_.stride(2),
            O_.stride(0), O_.stride(1), O_.stride(2),
            L_.stride(0), L_.stride(1),
            N, N,
            scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,   
            num_warps=4,
        )

        O = O_.reshape(*prefix, N, D)
        L = L_.reshape(*prefix, N)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal  # (c) 会用到
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = bool(getattr(ctx, "is_causal", False))

        flash_bwd = _get_compiled_flash_bwd()
        dQ, dK, dV = flash_bwd(Q, K, V, O, dO, L, is_causal)

        # 最后一个输入是 is_causal（bool），它没有梯度 -> 返回 None
        return dQ, dK, dV, None


@functools.lru_cache(None)
def _get_compiled_flash_bwd():
    # torch.compile 需要函数对象稳定；用 cache 保证只编译一次
    @torch.compile  # 也可以写 torch.compile(..., mode="max-autotune") 看你环境
    def _flash_bwd(Q, K, V, O, dO, L, is_causal: bool):
        # Q,K,V,O,dO: (..., N, D)
        # L: (..., N)  (logsumexp)
        *prefix, N, D = Q.shape
        scale = 1.0 / math.sqrt(D)

        # 全部用 fp32 做稳定计算（尤其 softmax 相关）
        Qf  = Q.to(torch.float32)
        Kf  = K.to(torch.float32)
        Vf  = V.to(torch.float32)
        Of  = O.to(torch.float32)
        dOf = dO.to(torch.float32)
        Lf  = L.to(torch.float32)

        # S = QK^T / sqrt(d)
        S = torch.matmul(Qf, Kf.transpose(-1, -2)) * scale  # (..., N, N)

        if is_causal:
            # mask 掉上三角 (q < k)
            # True 表示要 mask 的位置
            mask = torch.triu(torch.ones((N, N), device=S.device, dtype=torch.bool), diagonal=1)
            # 广播到 prefix
            S = S.masked_fill(mask, -1e6)

        # P = exp(S - L)
        P = torch.exp(S - Lf.unsqueeze(-1))  # (..., N, N)

        # D = rowsum(O * dO)  shape (..., N, 1)
        Dvec = (Of * dOf).sum(dim=-1, keepdim=True)

        # dV = P^T dO
        dV = torch.matmul(P.transpose(-1, -2), dOf)  # (..., N, D)

        # dP = dO V^T
        dP = torch.matmul(dOf, Vf.transpose(-1, -2))  # (..., N, N)

        # dS = P * (dP - D)
        dS = P * (dP - Dvec)  # (..., N, N)

        # dQ = dS K / sqrt(d)
        dQ = torch.matmul(dS, Kf) * scale  # (..., N, D)

        # dK = dS^T Q / sqrt(d)
        dK = torch.matmul(dS.transpose(-1, -2), Qf) * scale  # (..., N, D)

        # cast 回输入 dtype（测试通常容忍 fp32/bf16，但保持一致更好）
        return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)

    return _flash_bwd
