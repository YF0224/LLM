import json
import numpy as np
from cs336_basics.pretokenization import train_bpe
from cs336_basics.tokenization import Tokenizer

# ===== 第一步：训练 tokenizer（只用训练集文本）=====
print("训练 BPE tokenizer...")
vocab, merges = train_bpe(
    input_path     = "/data/train.txt", # put your training data here
    vocab_size     = 4000,
    special_tokens = ["<|endoftext|>"],
    num_processes  = 4,
)
print(f"词表大小：{len(vocab)}")
print(f"合并次数：{len(merges)}")

# 保存 vocab
vocab_to_save = {k: list(v) for k, v in vocab.items()}
with open("/tmp/cn_vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_to_save, f, ensure_ascii=False)

# 保存 merges
merges_to_save = [[list(a), list(b)] for a, b in merges]
with open("/tmp/cn_merges.json", "w", encoding="utf-8") as f:
    json.dump(merges_to_save, f, ensure_ascii=False)

# ===== 第二步：用 tokenizer 编码数据 =====
print("\n加载 tokenizer...")
tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

print("编码训练集...")
with open("/data/train.txt", "r", encoding="utf-8") as f:
    train_ids = list(tokenizer.encode_iterable(f))

print(f"训练集 token 数：{len(train_ids):,}")

# 切分 90% 训练，10% 验证
split = int(len(train_ids) * 0.9)
train_data = np.array(train_ids[:split], dtype=np.uint16)
val_data   = np.array(train_ids[split:], dtype=np.uint16)

print(f"训练集：{len(train_data):,} tokens")
print(f"验证集：{len(val_data):,} tokens")

np.save("/data/cn_train_bpe.npy", train_data)
np.save("/data/cn_val_bpe.npy",   val_data)
print("数据已保存")

# ===== 第三步：验证编解码正确 =====
print("\n编解码验证：")
test = "从前有一个小姑娘"
ids = tokenizer.encode(test)
print(f"原文：{test}")
print(f"token IDs：{ids}")
print(f"还原：{tokenizer.decode(ids)}")