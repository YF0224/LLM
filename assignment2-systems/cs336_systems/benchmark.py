from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch


# ----------------------------
# Table 1 model sizing configs
# ----------------------------
SIZES: Dict[str, Dict[str, int]] = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


@dataclass
class BenchResult:
    mean_ms: float
    std_ms: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=str, default="small", choices=list(SIZES.keys()))
    p.add_argument("--all_sizes", action="store_true",
                   help="If set, benchmark all sizes in Table 1. If not set, benchmark --size only.")
    p.add_argument("--seq_len", type=int, default="128",
                   help="Comma-separated seq lens, e.g. 128,256,512,1024")

    p.add_argument("--vocab_size", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--mode", type=str, default="fwd+bwd", choices=["fwd", "fwd+bwd"])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_compile", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--csv", type=str, default="",
                   help="Optional path to save results as CSV.")
    return p.parse_args()


# ----------------------------
# You MUST adapt this function
# ----------------------------
def build_model(
    vocab_size: int,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    device: torch.device,
    context_length: int,
    rope_theta: float = 10000.0,
) -> torch.nn.Module:
    from cs336_basics.transformer import TransformerLM  # <- 你的实现路径

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=None,   # 这里保持 None，下面 autocast 决定算子精度
    ).to(device)

    return model


def make_random_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_ids: (B, T), targets: (B, T)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    return x, y


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: (B, T, V), targets: (B, T)
    B, T, V = logits.shape
    return torch.nn.functional.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))


@torch.no_grad()
def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str,
    autocast_dtype: torch.dtype | None,
) -> None:
    if mode == "fwd":
        with torch.autocast(device_type=x.device.type, dtype=autocast_dtype, enabled=(autocast_dtype is not None)):
            _ = model(x)
        return

    # fwd+bwd
    assert optimizer is not None
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=x.device.type, dtype=autocast_dtype, enabled=(autocast_dtype is not None)):
        logits = model(x)
        loss = compute_loss(logits, y)
    loss.backward()
    optimizer.step()


def bench(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    x: torch.Tensor,
    y: torch.Tensor,
    warmup: int,
    steps: int,
    mode: str,
    autocast_dtype: torch.dtype | None,
) -> BenchResult:
    device = x.device

    # warmup (NOT timed)
    for _ in range(warmup):
        run_one_step(model, optimizer, x, y, mode=mode, autocast_dtype=autocast_dtype)
        _sync_if_cuda(device)

    times_ms = []
    for _ in range(steps):
        t0 = time.perf_counter()
        run_one_step(model, optimizer, x, y, mode=mode, autocast_dtype=autocast_dtype)
        _sync_if_cuda(device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    t = torch.tensor(times_ms, dtype=torch.float64)
    return BenchResult(mean_ms=float(t.mean().item()), std_ms=float(t.std(unbiased=True).item()))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # 直接 python benchmark.py 时，跑所有 size；如果你传了 --size 且不想全跑，可以自己加 --all_sizes 开关（见我前一条）
    sizes_to_run = list(SIZES.keys())

    rows = []

    for size in sizes_to_run:
        cfg = SIZES[size]
        seq_len = args.seq_len

        model = build_model(
            vocab_size=args.vocab_size,
            d_model=cfg["d_model"],
            d_ff=cfg["d_ff"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            device=device,
            context_length=seq_len,
        )

        if args.use_compile:
            model = torch.compile(model)

        if args.dtype == "fp32":
            autocast_dtype = None
            model = model.to(torch.float32)
        else:
            autocast_dtype = torch.bfloat16
            model = model.to(torch.float32)  # 参数保持 fp32，算子 autocast 到 bf16

        x, y = make_random_batch(args.batch_size, seq_len, args.vocab_size, device=device)

        optimizer = None
        if args.mode == "fwd+bwd":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        _sync_if_cuda(device)

        res = bench(
            model=model,
            optimizer=optimizer,
            x=x,
            y=y,
            warmup=args.warmup,
            steps=args.steps,
            mode=args.mode,
            autocast_dtype=autocast_dtype,
        )

        rows.append((size, res.mean_ms, res.std_ms))
        print(
            f"[size={size:>5} seq_len={seq_len:>5} dtype={args.dtype:>4} mode={args.mode:>7}] "
            f"mean={res.mean_ms:>10.3f} ms   std={res.std_ms:>10.3f} ms"
        )

        del model, optimizer, x, y
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== Summary ===")
    for size, mean_ms, std_ms in rows:
        print(f"{size:>6}: mean={mean_ms:>10.3f} ms   std={std_ms:>10.3f} ms")


if __name__ == "__main__":
    main()
