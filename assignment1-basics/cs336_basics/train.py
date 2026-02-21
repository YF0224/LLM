# cs336_basics/train
import os
import torch
import math
import numpy as np
from typing import BinaryIO, IO, Optional
from collections.abc import Iterable, Callable
from jaxtyping import Int, Float
import numpy.typing as npt

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)
        
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr * grad / (math.sqrt(t + 1))
                state['t'] = t + 1

        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0:
            raise ValueError(f"Invalid eps: {eps}")
        b1, b2 = betas
        if not (0 <= b1 < 1 and 0 <= b2 < 1):
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Sparse gradients not supported in this AdamW implementation.")

                state = self.state[p]
                # state init
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                t = state["t"] + 1  # t starts at 1
                m = state["m"]
                v = state["v"]

                # m <- b1*m + (1-b1)*g
                m.mul_(b1).add_(g, alpha=1 - b1)
                # v <- b2*v + (1-b2)*g^2
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                # alpha_t = lr * sqrt(1-b2^t) / (1-b1^t)
                bias_correction1 = 1 - (b1 ** t)
                bias_correction2 = 1 - (b2 ** t)
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # θ <- θ - αt * m / (sqrt(v) + eps)
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-alpha_t)

                # decoupled weight decay: θ <- θ - lr * wd * θ
                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)

                state["t"] = t

        return loss

def IrCosineSchedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        lr = max_learning_rate * it / warmup_iters
        return lr
    elif (warmup_iters <= it) and (it <= cosine_cycle_iters):
        cos_inner = math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        lr = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(cos_inner))
        return lr
    else:
        lr = min_learning_rate
    return lr

def GradientClipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6

    total_sq_norm = torch.zeros((), device=next(iter(parameters)).device)

    # 1) compute global L2 norm
    for p in parameters:
        if p.grad is None:
            continue
        total_sq_norm += p.grad.data.pow(2).sum()

    total_norm = torch.sqrt(total_sq_norm)

    # 2) clip if needed
    if total_norm <= max_l2_norm:
        return

    scale = max_l2_norm / (total_norm + eps)

    # 3) scale gradients in-place
    for p in parameters:
        if p.grad is None:
            continue
        p.grad.data.mul_(scale)
        
def GetBatch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(dataset.shape[0])
    if n <= context_length:
        raise ValueError("dataset too small for the given context_length")

    # start in [0, n - (context_length + 1)]
    starts = np.random.randint(0, n - context_length - 1 + 1, size=batch_size)

    x = np.stack([dataset[s : s + context_length] for s in starts], axis=0)
    y = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts], axis=0)

    x_t = torch.as_tensor(x, dtype=torch.long, device=device)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    return x_t, y_t

def CrossEntropyLoss(
    inputs: Float[torch.Tensor, " batch_size vocab_size"],
    targets: Int[torch.Tensor, " batch_size"],
) -> Float[torch.Tensor, ""]:
    # inputs: (B, V), targets: (B,)
    # m: (B, 1)
    m = torch.max(inputs, dim=-1, keepdim=True).values
    # logsumexp: (B,)
    logsumexp = (torch.log(torch.sum(torch.exp(inputs - m), dim=-1)) + m.squeeze(-1))
    # pick correct logits: (B,)
    correct = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = (logsumexp - correct).mean()
    return loss

def SaveCheckpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(ckpt, out)

def LoadCheckpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    ckpt = torch.load(src, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["iteration"])
    