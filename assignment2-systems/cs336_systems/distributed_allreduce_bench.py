from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


MB = 1024 * 1024


def sizeof_mb(num_bytes: int) -> float:
    return num_bytes / MB


def make_tensor_numel(bytes_size: int, dtype: torch.dtype) -> int:
    # float32 => 4 bytes
    return bytes_size // torch.tensor([], dtype=dtype).element_size()


@dataclass
class OneResult:
    backend: str
    device: str
    world_size: int
    size_bytes: int
    ok: bool
    mean_ms: float
    std_ms: float
    note: str


def setup_process_group(rank: int, world_size: int, backend: str, master_port: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def bench_one(
    rank: int,
    world_size: int,
    backend: str,
    device: torch.device,
    size_bytes: int,
    warmup: int,
    iters: int,
) -> Tuple[bool, List[float], str]:
    """
    Returns:
      ok, times_ms, note
    """
    dtype = torch.float32
    numel = make_tensor_numel(size_bytes, dtype=dtype)

    try:
        x = torch.randn(numel, device=device, dtype=dtype)
    except RuntimeError as e:
        return False, [], f"alloc_failed: {type(e).__name__}: {e}"

    # warmup
    for _ in range(warmup):
        dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return True, times, ""


def gather_object_all_ranks(obj, rank: int, world_size: int):
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered


def worker(rank: int, args: argparse.Namespace):
    backend = args.backend
    world_size = args.world_size

    setup_process_group(rank, world_size, backend, master_port=args.master_port)

    # Pick device
    if backend == "nccl":
        assert torch.cuda.is_available(), "NCCL requires CUDA"
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # Tensor sizes
    sizes = [int(s.strip()) for s in args.sizes_mb.split(",")]
    sizes_bytes = [s * MB for s in sizes]

    results: List[OneResult] = []
    for size_b in sizes_bytes:
        ok, times, note = bench_one(
            rank=rank,
            world_size=world_size,
            backend=backend,
            device=device,
            size_bytes=size_b,
            warmup=args.warmup,
            iters=args.iters,
        )
        if ok:
            t = torch.tensor(times, dtype=torch.float64)
            mean_ms = float(t.mean().item())
            std_ms = float(t.std(unbiased=True).item()) if len(times) > 1 else 0.0
            res = OneResult(
                backend=backend,
                device=str(device),
                world_size=world_size,
                size_bytes=size_b,
                ok=True,
                mean_ms=mean_ms,
                std_ms=std_ms,
                note="",
            )
        else:
            res = OneResult(
                backend=backend,
                device=str(device),
                world_size=world_size,
                size_bytes=size_b,
                ok=False,
                mean_ms=float("nan"),
                std_ms=float("nan"),
                note=note,
            )
        results.append(res)

        # reduce fragmentation between runs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Gather to rank0 and print
    gathered = gather_object_all_ranks(results, rank, world_size)
    if rank == 0:
        # gathered is list[ list[OneResult] ] per rank
        # aggregate per (size_bytes) across ranks
        print(f"\n=== all-reduce benchmark backend={backend} world_size={world_size} ===")
        print(f"(warmup={args.warmup}, iters={args.iters})")
        header = f"{'size(MB)':>8} | {'status':>6} | {'mean_ms(avg ranks)':>18} | {'std_ms(avg ranks)':>18} | note"
        print(header)
        print("-" * len(header))

        for idx, size_b in enumerate(sizes_bytes):
            per_rank = [gathered[r][idx] for r in range(world_size)]
            if all(r.ok for r in per_rank):
                mean = sum(r.mean_ms for r in per_rank) / world_size
                std = sum(r.std_ms for r in per_rank) / world_size
                status = "OK"
                note = ""
            else:
                status = "OOM/ERR"
                mean = float("nan")
                std = float("nan")
                # show first failing note
                bad = next((r for r in per_rank if not r.ok), None)
                note = bad.note if bad is not None else ""
            print(f"{sizeof_mb(size_b):8.0f} | {status:>6} | {mean:18.3f} | {std:18.3f} | {note}")

        # Optional CSV (one row per rank per size)
        if args.csv:
            import csv

            with open(args.csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["backend", "world_size", "rank", "device", "size_mb", "ok", "mean_ms", "std_ms", "note"])
                for r in range(world_size):
                    for res in gathered[r]:
                        w.writerow([
                            res.backend,
                            res.world_size,
                            r,
                            res.device,
                            int(sizeof_mb(res.size_bytes)),
                            int(res.ok),
                            res.mean_ms,
                            res.std_ms,
                            res.note,
                        ])
            print(f"\nSaved CSV: {args.csv}")

    cleanup_process_group()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"])
    p.add_argument("--world_size", type=int, default=2, choices=[2, 4, 6])
    p.add_argument("--sizes_mb", type=str, default="1,10,100,1024",
                   help="Comma-separated tensor sizes in MB (float32). Example: 1,10,100,1024")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--master_port", type=str, default="29500")
    p.add_argument("--csv", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    mp.spawn(worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
