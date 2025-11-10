from __future__ import annotations
from collections import defaultdict
import os
import regex as re
from multiprocessing import Pool, get_context
from typing import BinaryIO

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_COMPILED_PAT = re.compile(GPT2_PAT)

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """this function finds chunk boundaries in a binary file based on a special token."""
    assert isinstance(split_special_token, bytes) # split_special_token must be of type bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = max(1, file_size //desired_num_chunks)
    bounds = [i for i in range(0, file_size, chunk_size)]
    # bounds = [i * chunk_size for i in range(desired_num_chunks + 1)]
    bounds[-1] = file_size  # Ensure the last boundary is the end of the file
    mini = 4096
    for i in range(1, len(bounds) - 1):
        pos = bounds[i]
        file.seek(pos)
        while True:
            buf = file.read(mini)
            if not buf:
                bounds[i] = file_size
                break
            found = buf.find(split_special_token)
            if found != -1:
                bounds[i] = pos + found
                break
            pos += len(buf)
    return bounds

def process_chunk(args: tuple[str, int, int, list[str]]) -> list[list[int]]:
    input_path, start, end, special_tokens = args
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode('utf-8', errors='ignore')
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    documents = re.split(pattern, chunk)
    chunk_ids : list[list[int]] = []
    for doc in documents:
        for match in _COMPILED_PAT.finditer(doc):
            b = match.group(0).encode("utf-8")
            chunk_ids.append(list(b))

    return chunk_ids

def get_pair_frequencies(ids: list[list[int]]) -> tuple[
        defaultdict[tuple[int, int], set], 
        defaultdict[tuple[int, int], int]
        ]:
    pair_to_indices = defaultdict(set)
    counts = defaultdict(int)
    for i, token_ids in enumerate(ids):
        for pair in zip(token_ids, token_ids[1:]):
            pair_to_indices[pair].add(i)
            counts[pair] += 1
    return pair_to_indices, counts

def merge_pair(ids: list[list[int]], old_pair: tuple[int, int], new_token: int) -> None:
    new_tokens = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == old_pair:
            new_tokens.append(new_token)
            i += 2
        else:
            new_tokens.append(ids[i])
            i += 1
    return new_tokens

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 8
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # aim to initialize the vocabulary with all single-byte tokens
    vocab = {i : bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
    
    with open(input_path, 'rb') as f:
        bounds = find_chunk_boundaries(
            f, 
            desired_num_chunks=num_processes, 
            split_special_token="<|endoftext|>".encode("utf-8")
        )
        
    task_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(bounds[:-1], bounds[1:])
    ]
    
    try:
        ctx = get_context("forkserver")
    except ValueError:
        ctx = get_context("spawn")

    with ctx.Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_chunk, task_args) # list[list[list[int]]]
    
    ids :list[list[int]] = []
    for chunk_id in chunk_results:
        for tok_ids in chunk_id:
            ids.append(tok_ids)
    merges: list[tuple[int, int]] = []

    pair_to_indices, counts = get_pair_frequencies(ids)
    
    nums_merges = vocab_size - len(vocab)
    
    for _ in range(nums_merges):
        if not counts:
            break
# Find the most frequent pair with deterministic tie-breaking
        def rank(pair: tuple[int, int]) -> tuple[int, tuple[bytes, bytes]]:
            return counts[pair], (vocab[pair[0]], vocab[pair[1]])

        best_pair = max(counts, key=rank)
        new_token = vocab[best_pair[0]] + vocab[best_pair[1]]
        new_token_id = len(vocab)
        vocab[new_token_id] = new_token
        merges.append(best_pair)

        # 找出包含 best_pair 的句子
        affected_indices = pair_to_indices[best_pair].copy()

        for i in affected_indices:
            token_ids = ids[i]
            if len(token_ids) < 2:
                continue

            # ⚠️ 只合并这一行中出现 best_pair 的地方
            new_token_ids = merge_pair(token_ids, best_pair, new_token_id)

            # 更新 pair_to_indices / counts（只更新受影响的pair）
            old_pairs = list(zip(token_ids, token_ids[1:]))
            new_pairs = list(zip(new_token_ids, new_token_ids[1:]))

            # 移除旧 pair
            for pair in old_pairs:
                if pair in counts:
                    counts[pair] -= 1
                    if counts[pair] <= 0:
                        del counts[pair]
                        pair_to_indices.pop(pair, None)
                    else:
                        pair_to_indices[pair].discard(i)

            # 添加新 pair
            for pair in new_pairs:
                counts[pair] = counts.get(pair, 0) + 1
                pair_to_indices[pair].add(i)

            # 替换成新 token_ids
            ids[i] = new_token_ids

        # ⚠️ 完成一次合并后，移除 best_pair 自身的索引（因为它已经被合并）
        pair_to_indices.pop(best_pair, None)
        counts.pop(best_pair, None)

    # 输出 merge 记录
    merges = [(vocab[a], vocab[b]) for a, b in merges]
    return vocab, merges
