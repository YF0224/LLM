import json
import regex as re
from typing import Iterable, Iterator

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.special_tokens: list[str] = list(special_tokens) if special_tokens else []
        
        # 确保 special tokens 在 vocab 里
        vocab_values = set(self.vocab.values())
        for sp in self.special_tokens:
            sp_b = sp.encode("utf-8")
            if sp_b not in vocab_values:
                self.vocab[len(self.vocab)] = sp_b
                vocab_values.add(sp_b)

        # bytes -> id 的反向表
        self.byte_encoder: dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        # 用于把 special tokens 从文本中切出来（避免被 PAT 拆分）
        if self.special_tokens:
            # 长的优先，避免短 token 抢匹配
            st_sorted = sorted(self.special_tokens, key=len, reverse=True)
            self._special_split_re = re.compile("(" + "|".join(re.escape(x) for x in st_sorted) + ")")
        else:
            self._special_split_re = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        # 兼容两种常见格式：
        # A) {"token_str": id, ...}
        # B) {"id_str": "token_str", ...}
        vocab: dict[int, bytes] = {}
        if vocab_data:
            sample_k = next(iter(vocab_data.keys()))
            sample_v = vocab_data[sample_k]
            if isinstance(sample_v, int):
                # A
                vocab = {int(v): k.encode("utf-8") for k, v in vocab_data.items()}
            else:
                # B
                vocab = {int(k): str(v).encode("utf-8") for k, v in vocab_data.items()}

        # merges 文件：每行两个 token（按空格分隔）
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a, b = line.split(" ")
                merges.append((a.encode("utf-8"), b.encode("utf-8")))

        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, token_bytes: bytes) -> list[bytes]:
        # 预分词单元 token_bytes 先拆成单字节 pieces
        pieces: list[bytes] = [bytes([x]) for x in token_bytes]

        # 按 merges “创建顺序”逐条应用（顺序非常重要）
        for a, b in self.merges:
            i = 0
            new_pieces: list[bytes] = []
            while i < len(pieces):
                if i + 1 < len(pieces) and pieces[i] == a and pieces[i + 1] == b:
                    new_pieces.append(a + b)
                    i += 2
                else:
                    new_pieces.append(pieces[i])
                    i += 1
            pieces = new_pieces

        return pieces

    def encode(self, text: str) -> list[int]:
        # Step 1: 处理 special tokens：先 split 出来，保证它们永远不被拆
        if self._special_split_re:
            parts = [p for p in self._special_split_re.split(text) if p != ""]
        else:
            parts = [text]

        # Step 1: pre-tokenize（对非 special 部分用 PAT）
        pretokens: list[str] = []
        for p in parts:
            if p in self.special_tokens:
                pretokens.append(p)
            else:
                pretokens.extend(PAT.findall(p))

        # Step 2: 对每个 pre-token 独立应用 merges，然后映射到 ids
        ids: list[int] = []
        for tok in pretokens:
            if tok in self.special_tokens:
                b = tok.encode("utf-8")
                ids.append(self.byte_encoder[b])
                continue

            tok_b = tok.encode("utf-8")
            merged_pieces = self._apply_merges(tok_b)
            for piece in merged_pieces:
                ids.append(self.byte_encoder[piece])

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # 惰性：逐 chunk 编码并 yield，不把所有内容读进内存
        for chunk in iterable:
            for tid in self.encode(chunk):
                yield tid

    def decode(self, ids: list[int]) -> str:
        # vocab lookup -> bytes concat -> utf-8 decode with replacement
        out = b"".join(self.vocab[i] for i in ids)
        return out.decode("utf-8", errors="replace")
