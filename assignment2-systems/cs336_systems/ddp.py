from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.distributed as dist


def _require_dist_initialized() -> Tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized. Launch with torchrun and init_process_group.")
    return dist.get_rank(), dist.get_world_size()


def _broadcast_module_params_from_rank0(module: torch.nn.Module) -> None:
    # Ensure all ranks start from identical parameters
    for p in module.parameters():
        dist.broadcast(p.data, src=0)


# =========================
# 2.3.2 Individual overlap
# =========================

class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.rank, self.world_size = _require_dist_initialized()
        _broadcast_module_params_from_rank0(self.module)

        self._handles: Dict[int, dist.Work] = {}  # key: id(param) -> handle

        # Register grad-ready hook for each param
        for p in self.module.parameters():
            if not p.requires_grad:
                continue

            def _make_hook(param: torch.nn.Parameter):
                # Called after param.grad is accumulated for this backward
                def _hook(_param: torch.Tensor):
                    # param.grad exists now
                    if param.grad is None:
                        return
                    # Launch async all-reduce immediately (overlap with remaining backward)
                    h = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
                    self._handles[id(param)] = h
                return _hook

            # NOTE: signature is fn(param) for post_accumulate_grad_hook
            p.register_post_accumulate_grad_hook(_make_hook(p))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        # Wait for all launched comms, then average grads
        for p in self.module.parameters():
            if not p.requires_grad or p.grad is None:
                continue
            h = self._handles.get(id(p), None)
            if h is not None:
                h.wait()
            p.grad.div_(self.world_size)

        self._handles.clear()


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    return DDPIndividualParameters(module)


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    # Called after loss.backward(), before optimizer.step()
    if hasattr(ddp_model, "finish_gradient_synchronization"):
        ddp_model.finish_gradient_synchronization()
    else:
        raise AttributeError("DDP model missing finish_gradient_synchronization().")


# =========================
# 2.3.3 Bucketed overlap
# =========================

@dataclass
class _Bucket:
    params: List[torch.nn.Parameter]
    offsets: List[int]          # offset in flat buffer (in elements)
    numels: List[int]
    total_numel: int
    flat: Optional[torch.Tensor] = None
    ready: int = 0
    handle: Optional[dist.Work] = None


class DDPBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.rank, self.world_size = _require_dist_initialized()
        _broadcast_module_params_from_rank0(self.module)

        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        if self.bucket_size_bytes <= 0:
            raise ValueError("bucket_size_mb must be > 0")

        # Build buckets in reverse param order (grads become ready in ~reverse)
        params = [p for p in self.module.parameters() if p.requires_grad]
        params_rev = list(reversed(params))

        self._param_to_bucket: Dict[int, Tuple[int, int]] = {}  # id(param) -> (bucket_idx, param_idx_in_bucket)
        self._buckets: List[_Bucket] = []

        cur: List[torch.nn.Parameter] = []
        cur_numel: List[int] = []
        cur_bytes = 0

        def _flush():
            if not cur:
                return
            offsets = []
            off = 0
            for n in cur_numel:
                offsets.append(off)
                off += n
            b = _Bucket(params=list(cur), offsets=offsets, numels=list(cur_numel), total_numel=off)
            self._buckets.append(b)
            cur.clear()
            cur_numel.clear()

        for p in params_rev:
            # bucket buffer stored in fp32 for stability; but we copy from p.grad dtype and cast when needed
            n = p.numel()
            # Use fp32 bucket to be safe; element size 4 bytes
            add_bytes = n * 4
            if cur and (cur_bytes + add_bytes) > self.bucket_size_bytes:
                _flush()
                cur_bytes = 0
            cur.append(p)
            cur_numel.append(n)
            cur_bytes += add_bytes
        _flush()

        # Build mapping
        for bi, b in enumerate(self._buckets):
            for pi, p in enumerate(b.params):
                self._param_to_bucket[id(p)] = (bi, pi)

        # Register hooks
        for p in params:
            def _make_hook(param: torch.nn.Parameter):
                def _hook(_param: torch.Tensor):
                    if param.grad is None:
                        return
                    bi, pi = self._param_to_bucket[id(param)]
                    bucket = self._buckets[bi]
                    # Lazy allocate bucket flat buffer (fp32)
                    if bucket.flat is None or bucket.flat.device != param.grad.device:
                        bucket.flat = torch.empty(bucket.total_numel, device=param.grad.device, dtype=torch.float32)

                    # Copy this grad into flat buffer
                    start = bucket.offsets[pi]
                    end = start + bucket.numels[pi]
                    bucket.flat[start:end].copy_(param.grad.detach().to(torch.float32).reshape(-1))

                    bucket.ready += 1
                    # If bucket complete: launch async all-reduce now
                    if bucket.ready == len(bucket.params) and bucket.handle is None:
                        bucket.handle = dist.all_reduce(bucket.flat, op=dist.ReduceOp.SUM, async_op=True)
                return _hook

            p.register_post_accumulate_grad_hook(_make_hook(p))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def on_train_batch_start(self) -> None:
        # reset per-iteration state
        for b in self._buckets:
            b.ready = 0
            b.handle = None
            # keep b.flat allocated (reuse) for perf, but OK if None

    def finish_gradient_synchronization(self) -> None:
        # Wait all buckets, then write averaged grads back into params
        for b in self._buckets:
            if b.handle is not None:
                b.handle.wait()
            if b.flat is None:
                continue
            # average
            b.flat.div_(self.world_size)
            # unflatten to each param.grad
            for p, start, n in zip(b.params, b.offsets, b.numels):
                if p.grad is None:
                    continue
                chunk = b.flat[start:start + n].view_as(p.grad).to(p.grad.dtype)
                p.grad.copy_(chunk)

        # do not clear flat buffers; reuse
        for b in self._buckets:
            b.handle = None
            b.ready = 0


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    return DDPBucketed(module, bucket_size_mb)


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    if hasattr(ddp_model, "on_train_batch_start"):
        ddp_model.on_train_batch_start()
    # If your implementation doesn't need it, it's fine to no-op.


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    if hasattr(ddp_model, "finish_gradient_synchronization"):
        ddp_model.finish_gradient_synchronization()
    else:
        raise AttributeError("DDP model missing finish_gradient_synchronization().")


# =========================
# Sharded optimizer (ZeRO-1 minimal)
# =========================

class ShardedOptimizer(torch.optim.Optimizer):
    """
    Minimal optimizer-state sharding:
    - Each param has an owner rank = param_index % world_size
    - Only owner keeps optimizer state + applies update
    - After local step, broadcast updated params from owners to all ranks
    This keeps parameters identical across ranks while sharding optimizer state.
    """
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs):
        self.rank, self.world_size = _require_dist_initialized()

        # Flatten param list (support param groups too)
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            # param groups
            self._all_param_groups = params
            all_params: List[torch.nn.Parameter] = []
            for g in params:
                all_params.extend(list(g["params"]))
        else:
            self._all_param_groups = [{"params": list(params)}]
            all_params = list(self._all_param_groups[0]["params"])

        self._all_params = all_params
        self._owners: Dict[int, int] = {id(p): (i % self.world_size) for i, p in enumerate(self._all_params)}

        # Create local param groups for owner params only
        local_groups = []
        for g in self._all_param_groups:
            local_ps = [p for p in g["params"] if self._owners[id(p)] == self.rank]
            if local_ps:
                ng = dict(g)
                ng["params"] = local_ps
                local_groups.append(ng)

        # If a rank owns no params, still create a dummy optimizer over empty list
        self._local_opt = optimizer_cls(local_groups if local_groups else [{"params": []}], **kwargs)

        # torch.optim.Optimizer base init wants param_groups, defaults
        super().__init__(self._local_opt.param_groups, self._local_opt.defaults)

    def zero_grad(self, set_to_none: bool = True):
        self._local_opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self._local_opt.step(closure)

        # Broadcast updated params from owners so all ranks match
        for p in self._all_params:
            owner = self._owners[id(p)]
            dist.broadcast(p.data, src=owner)

        return loss

    def state_dict(self):
        # Only contains local optimizer state (sharded)
        return self._local_opt.state_dict()

    def load_state_dict(self, state_dict):
        self._local_opt.load_state_dict(state_dict)


def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
