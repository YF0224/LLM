from __future__ import annotations

from typing import Type

import torch



def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyFlashAttnAutogradFunctionClass
    from cs336_systems.flash_attention import FlashAttention2PyTorch
    
    return FlashAttention2PyTorch

def get_flashattention_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    from cs336_systems.flash_attention import FlashAttention2Triton
    return FlashAttention2Triton



def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    Overlaps communication with backprop by async all-reducing
    each parameter's grad as soon as it is ready.
    """
    from cs336_systems.ddp import DDPIndividualParameters
    return DDPIndividualParameters(module)


def ddp_individual_parameters_on_after_backward(
    ddp_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Run after backward(), before optimizer.step().
    Must ensure all async all-reduces are completed/queued and grads are averaged.
    """
    
    if not hasattr(ddp_model, "finish_gradient_synchronization"):
        raise AttributeError("ddp_model must implement finish_gradient_synchronization().")
    ddp_model.finish_gradient_synchronization()


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a DDP container that buckets parameter grads and overlaps
    bucket all-reduce with backprop.
    """
    from cs336_systems.ddp import DDPBucketed
    return DDPBucketed(module, bucket_size_mb=bucket_size_mb)


def ddp_bucketed_on_after_backward(
    ddp_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Run after backward(), before optimizer.step().
    Wait for bucket all-reduces and write averaged grads back to params.
    """
    if not hasattr(ddp_model, "finish_gradient_synchronization"):
        raise AttributeError("ddp_model must implement finish_gradient_synchronization().")
    ddp_model.finish_gradient_synchronization()


def ddp_bucketed_on_train_batch_start(
    ddp_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Run at the start of each training step to reset per-step bucket state.
    """
    # Some implementations might not need this; ours does.
    if hasattr(ddp_model, "on_train_batch_start"):
        ddp_model.on_train_batch_start()


def get_sharded_optimizer(
    params,
    optimizer_cls: Type[torch.optim.Optimizer],
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Returns an optimizer with optimizer-state sharding (ZeRO-1 style minimal):
    each rank only keeps state for the params it "owns", applies updates locally,
    then broadcasts updated params to all ranks.
    """
    from cs336_systems.ddp import ShardedOptimizer
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
