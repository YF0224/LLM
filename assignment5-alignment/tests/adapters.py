import os
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have same length")

    # Tokenize prompt and full (prompt+output) separately so we can build a
    # response mask aligned to the *labels* positions (shifted by one).
    full_id_list: list[list[int]] = []
    prompt_len_list: list[int] = []

    for p, o in zip(prompt_strs, output_strs):
        full_ids = tokenizer(p + o, add_special_tokens=True).input_ids
        prompt_ids = tokenizer(p, add_special_tokens=True).input_ids
        full_id_list.append(list(full_ids))
        prompt_len_list.append(len(prompt_ids))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # Many causal LMs (e.g., GPT2) don't define a pad token; fall back to eos.
        pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

    max_len = max(len(ids) for ids in full_id_list)
    batch = len(full_id_list)
    full_padded = torch.full((batch, max_len), pad_id, dtype=torch.long)
    lengths = torch.tensor([len(ids) for ids in full_id_list], dtype=torch.long)
    for i, ids in enumerate(full_id_list):
        full_padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    # Shift for next-token prediction.
    input_ids = full_padded[:, :-1].contiguous()
    labels = full_padded[:, 1:].contiguous()

    # Build response_mask on the labels positions (same shape as labels).
    # If the response begins at token index prompt_len in the *unshifted* sequence,
    # it begins at index prompt_len-1 in the shifted labels.
    seq_len = labels.shape[1]
    response_mask = torch.zeros((batch, seq_len), dtype=torch.float32)
    for i in range(batch):
        full_len = int(lengths[i].item())
        prompt_len = int(prompt_len_list[i])
        start = max(prompt_len - 1, 0)
        end = max(full_len - 1, 0)  # labels length for this unpadded sample
        if end > start:
            response_mask[i, start:end] = 1.0

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollout_responses and repeated_ground_truths must have same length")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if len(rollout_responses) % group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size")

    rewards = []
    format_rewards = []
    answer_rewards = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        out = reward_fn(resp, gt)
        rewards.append(float(out["reward"]))
        format_rewards.append(float(out.get("format_reward", out["reward"])))
        answer_rewards.append(float(out.get("answer_reward", out["reward"])))

    raw_rewards = torch.tensor(rewards, dtype=torch.float32)
    rg = raw_rewards.view(-1, group_size)
    mean = rg.mean(dim=1, keepdim=True)
    adv = rg - mean
    if normalize_by_std:
        # Use population std (unbiased=False) for stability in small groups.
        std = rg.std(dim=1, keepdim=True, unbiased=False)
        adv = adv / (std + advantage_eps)

    normalized_rewards = adv.view(-1)

    metadata: dict[str, float] = {
        "raw_reward_mean": float(raw_rewards.mean().item()),
        "raw_reward_std": float(raw_rewards.std(unbiased=False).item()),
        "format_reward_mean": float(torch.tensor(format_rewards).mean().item()),
        "answer_reward_mean": float(torch.tensor(answer_rewards).mean().item()),
    }
    return normalized_rewards, raw_rewards, metadata


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    logp = torch.log_softmax(logits, dim=-1)
    p = torch.exp(logp)
    return -(p * logp).sum(dim=-1)


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # (B, T, V)
    logp = torch.log_softmax(logits, dim=-1)
    token_log_probs = logp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    out: dict[str, torch.Tensor] = {"log_probs": token_log_probs}
    if return_token_entropy:
        out["token_entropy"] = run_compute_entropy(logits)
    return out


def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    return -(raw_rewards_or_advantages * policy_log_probs)


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    ratio_clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    surr1 = ratio * advantages
    surr2 = ratio_clipped * advantages
    loss = -torch.minimum(surr1, surr2)

    clipped = ratio != ratio_clipped
    meta = {
        "ratio": ratio.detach(),
        "ratio_clipped": ratio_clipped.detach(),
        "clipped": clipped.detach(),
    }
    return loss, meta


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    if loss_type == "no_baseline":
        return run_compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    if loss_type == "reinforce_with_baseline":
        return run_compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    if loss_type == "grpo_clip":
        return run_compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    raise ValueError(f"Unknown loss_type: {loss_type}")


def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    mask_f = mask.to(dtype=tensor.dtype)
    masked = tensor * mask_f
    if dim is None:
        denom = mask_f.sum()
        denom = torch.clamp(denom, min=1.0)
        return masked.sum() / denom
    denom = mask_f.sum(dim=dim)
    denom = torch.clamp(denom, min=1.0)
    return masked.sum(dim=dim) / denom

def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    # Negative log-likelihood on response tokens only.
    per_token_nll = -policy_log_probs
    loss = run_masked_normalize(
        tensor=per_token_nll,
        mask=response_mask,
        dim=None,
        normalize_constant=float(normalize_constant) if normalize_constant is not None else 1.0,
    )
    loss = loss / float(gradient_accumulation_steps)
    loss.backward()
    meta: dict[str, torch.Tensor] = {}
    return loss.detach(), meta

    
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards is required for loss_type='no_baseline'")
        per_token_loss, meta = run_compute_policy_gradient_loss(
            policy_log_probs=policy_log_probs,
            loss_type=loss_type,
            raw_rewards=raw_rewards,
            advantages=torch.zeros_like(raw_rewards),
            old_log_probs=torch.zeros_like(policy_log_probs),
            cliprange=0.0,
        )
    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages is required for loss_type='reinforce_with_baseline'")
        per_token_loss, meta = run_compute_policy_gradient_loss(
            policy_log_probs=policy_log_probs,
            loss_type=loss_type,
            raw_rewards=torch.zeros_like(advantages),
            advantages=advantages,
            old_log_probs=torch.zeros_like(policy_log_probs),
            cliprange=0.0,
        )
    elif loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError("advantages, old_log_probs, cliprange are required for loss_type='grpo_clip'")
        per_token_loss, meta = run_compute_policy_gradient_loss(
            policy_log_probs=policy_log_probs,
            loss_type=loss_type,
            raw_rewards=torch.zeros_like(advantages),
            advantages=advantages,
            old_log_probs=old_log_probs,
            cliprange=float(cliprange),
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Aggregate only over response tokens (length-normalized).
    loss = run_masked_mean(per_token_loss, response_mask, dim=None)
    loss = loss / float(gradient_accumulation_steps)
    loss.backward()
    return loss.detach(), meta


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    mask_f = mask.to(dtype=tensor.dtype)
    masked = tensor * mask_f
    if dim is None:
        return masked.sum() / float(normalize_constant)
    return masked.sum(dim=dim) / float(normalize_constant)


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    dataset_path = Path(dataset_path)
    # Default template: Alpaca-style prompt/response formatting.
    prompt_template_path = Path(__file__).resolve().parents[1] / "cs336_alignment" / "prompts" / "alpaca_sft.prompt"
    template = prompt_template_path.read_text(encoding="utf-8")

    docs: list[str] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            instruction = ex.get("instruction", ex.get("prompt", ex.get("query", "")))
            response = ex.get("response", ex.get("output", ex.get("answer", "")))
            docs.append(template.format(instruction=instruction, response=response))

    if shuffle:
        random.shuffle(docs)

    # Tokenize all documents and concatenate into a single token stream.
    token_stream: list[int] = []
    for doc in docs:
        ids = tokenizer(doc, add_special_tokens=True).input_ids
        token_stream.extend(list(ids))

    # Build packed examples: input_ids are length seq_length, labels are the next token.
    # Drop any remainder that doesn't have a full next-token label.
    n = len(token_stream)
    max_start = n - (seq_length + 1)
    if max_start < 0:
        max_start = -1

    examples: list[dict[str, Tensor]] = []
    for start in range(0, max_start + 1, seq_length):
        chunk = token_stream[start : start + seq_length + 1]
        inp = torch.tensor(chunk[:-1], dtype=torch.long)
        lab = torch.tensor(chunk[1:], dtype=torch.long)
        examples.append({"input_ids": inp, "labels": lab})

    @dataclass
    class _PackedDataset(Dataset):
        data: list[dict[str, Tensor]]

        def __len__(self) -> int:  # type: ignore[override]
            return len(self.data)

        def __getitem__(self, idx: int) -> dict[str, Tensor]:  # type: ignore[override]
            return self.data[idx]

    return _PackedDataset(examples)


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    n = len(dataset)
    indices = list(range(n))
    if shuffle:
        random.shuffle(indices)

    batches: list[dict[str, Tensor]] = []
    for start in range(0, n, batch_size):
        batch_indices = indices[start : start + batch_size]
        input_ids = torch.stack([dataset[i]["input_ids"] for i in batch_indices], dim=0)
        labels = torch.stack([dataset[i]["labels"] for i in batch_indices], dim=0)
        batches.append({"input_ids": input_ids.to(torch.long), "labels": labels.to(torch.long)})
    return batches


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    # Prefer patterns that explicitly mention an option letter.
    text = model_output.strip()
    m = re.search(r"(?i)\b(answer|correct answer)\s*(is|:)?\s*\(?\s*([ABCD])\s*\)?\b", text)
    if m:
        return m.group(3).upper()
    # Fallback: first standalone letter token.
    m2 = re.search(r"\b([ABCD])\b", text)
    if m2:
        return m2.group(1).upper()
    return None


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    # GSM8K numeric answers are typically integers; take the last explicit number in the output.
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", model_output)
    if not nums:
        return None
    return nums[-1]


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    device = next(lm.parameters()).device

    def _sequence_logp(model: torch.nn.Module, response: str, no_grad: bool) -> torch.Tensor:
        toks = run_tokenize_prompt_and_output(
            prompt_strs=[prompt],
            output_strs=[response],
            tokenizer=tokenizer,
        )
        input_ids = toks["input_ids"].to(device)
        labels = toks["labels"].to(device)
        response_mask = toks["response_mask"].to(device)
        if no_grad:
            with torch.no_grad():
                lp = run_get_response_log_probs(
                    model=model,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )["log_probs"]
        else:
            lp = run_get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )["log_probs"]
        # Sum log-probs over response tokens only.
        return (lp * response_mask).sum()

    logp_pi_c = _sequence_logp(lm, response_chosen, no_grad=False)
    logp_pi_r = _sequence_logp(lm, response_rejected, no_grad=False)
    logp_ref_c = _sequence_logp(lm_ref, response_chosen, no_grad=True)
    logp_ref_r = _sequence_logp(lm_ref, response_rejected, no_grad=True)

    delta = (logp_pi_c - logp_pi_r) - (logp_ref_c - logp_ref_r)
    # DPO loss: -log(sigmoid(beta * delta)) = softplus(-beta * delta)
    return torch.nn.functional.softplus(-beta * delta)
