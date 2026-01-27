import regex as re
import os
from typing import BinaryIO
from collections import Counter, defaultdict

PAT=rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
rx = re.compile(PAT)

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks if desired_num_chunks > 0 else file_size

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# 统计 pair 频率（改：按 word_cnt 加权统计，且基于“词->byte indices”的缓存来做）
def get_pair_freq(word_idx: dict[bytes, list[int]], word_cnt: Counter) -> defaultdict[tuple[int, int], int]:
    pair_freq: defaultdict[tuple[int, int], int] = defaultdict(int)
    for w, idxs in word_idx.items():
        cnt = word_cnt[w]
        if cnt <= 0:
            continue
        for a, b in zip(idxs, idxs[1:]):
            pair_freq[(a, b)] += cnt
    return pair_freq

def pretokenizer_chunk(chunk: bytes, special_tokens: list[bytes]) -> Counter:
    """
    Split a chunk by special tokens (not counted), then regex pre-tokenize.
    Return Counter of pre-tokens (bytes).
    """
    if special_tokens:
        # 为了避免短 token 抢先匹配，按长度降序拼接
        pattern = b"(" + b"|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)) + b")"
        parts = re.split(pattern, chunk)
    else:
        parts = [chunk]

    word_cnt = Counter()
    for part in parts:
        if not part:
            continue
        # special token 本身不进入预分词统计（与对照实现一致：训练完再追加到 vocab）
        if special_tokens and part in special_tokens:
            continue
        for m in rx.finditer(part):
            tok = m.group(0)
            if tok:
                word_cnt[tok] += 1
    return word_cnt

def merge(
    counts: dict[tuple[int, int], int],
    indices: list[int],
    pair: tuple[int, int],
    new_index: int,
    cnt: int
):
    new_indices = []
    i, flag = 0, 1  # flag=1：前一指针指向 unmerged；flag=0：前一指针指向 merged(new_index)
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)

            # 只处理“前向”counts更新
            if i > 0:
                counts[(indices[i - 1], indices[i])] -= cnt
                if flag:
                    counts[(indices[i - 1], new_index)] += cnt
                else:
                    counts[(new_index, new_index)] += cnt

            flag = 0
            i += 2
        else:
            new_indices.append(indices[i])

            if i > 0:
                if not flag:
                    counts[(indices[i - 1], indices[i])] -= cnt
                    counts[(new_index, indices[i])] += cnt

            flag = 1
            i += 1

    return new_indices

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 8,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # ===== 初始化（对齐参考：先 256 字节 vocab；special tokens 训练完再加）=====
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    special_tokens_b = [t.encode("utf-8") for t in (special_tokens or [])]
    merge_times = vocab_size - 256 - len(special_tokens_b)
    if merge_times < 0:
        merge_times = 0

    # ===== 读取文件并分 chunk =====
    chunks: list[bytes] = []
    with open(input_path, "rb") as f:
        split_tok = special_tokens_b[0] if special_tokens_b else b"<|endoftext|>"
        boundaries = find_chunk_boundaries(f, max(1, num_processes), split_special_token=split_tok)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start))

    # ===== 预分词计数（对齐参考：得到 word_cnt）=====
    word_cnt = Counter()
    for chunk in chunks:
        word_cnt.update(pretokenizer_chunk(chunk, special_tokens_b))

    # ===== 构建 word -> indices（对齐参考：缓存每个词的 byte indices）=====
    word_idx: dict[bytes, list[int]] = {}
    for w in word_cnt.keys():
        word_idx[w] = list(w)  # bytes -> list[int] (0..255)

    # ===== 初始化 pair counts（按词频加权）=====
    counts = get_pair_freq(word_idx, word_cnt)

    # ===== BPE 训练：每轮选“最高频 + 字典序最大”的 pair =====
    for _ in range(merge_times):
        if not counts:
            break

        max_value = max(counts.values())
        if max_value <= 0:
            break

        # 找到所有 max 的 pair，然后按 (vocab[a], vocab[b]) 字典序降序取第一个
        max_keys = [(p, (vocab[p[0]], vocab[p[1]])) for p, v in counts.items() if v == max_value]
        pair, sep = sorted(max_keys, key=lambda x: x[1], reverse=True)[0]

        # 更新 merges / vocab
        merges.append(sep)
        new_index = len(vocab)
        vocab[new_index] = sep[0] + sep[1]

        # 更新 word_idx 和 counts（增量更新）
        counts[pair] = 0
        for w, idxs in word_idx.items():
            cnt = word_cnt[w]
            if cnt <= 0:
                continue
            word_idx[w] = merge(counts, idxs, pair, new_index, cnt)

    # ===== 训练完再把 special tokens 追加进 vocab（对齐参考）=====
    for t in special_tokens_b:
        vocab[len(vocab)] = t

    return vocab, merges
