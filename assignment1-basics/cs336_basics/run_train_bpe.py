"""
BPE Tokenizer 版中文训练脚本
前置条件：已运行 train_tokenizer.py，生成了以下文件：
  /data/cn_vocab.json
  /data/cn_merges.json
  /data/cn_train_bpe.npy
  /data/cn_val_bpe.npy
用法：python run_train_bpe.py
"""
import os
import json
import time
import numpy as np
import torch
from cs336_basics.transformer import TransformerLM
from cs336_basics.tokenization import Tokenizer
from cs336_basics.train import (
    AdamW, IrCosineSchedule, GradientClipping,
    GetBatch, CrossEntropyLoss, SaveCheckpoint, LoadCheckpoint
)

# ========== 超参数配置 ==========
# 模型
VOCAB_SIZE     = 4000       # 和 BPE 词表一致
CONTEXT_LENGTH = 256        # BPE token 比字节更紧凑，可以用更长的上下文
D_MODEL        = 512
NUM_LAYERS     = 6
NUM_HEADS      = 8
D_FF           = 1344       # 8/3 × 512 ≈ 1365，取最近64的倍数
ROPE_THETA     = 10000.0

# 训练
BATCH_SIZE     = 32
TOTAL_STEPS    = 5000
EVAL_EVERY     = 200
SAVE_EVERY     = 1000

# 优化器
LR_MAX         = 3e-4
LR_MIN         = 3e-5
WARMUP_STEPS   = 200
WEIGHT_DECAY   = 0.1
GRAD_CLIP      = 1.0

# 路径
VOCAB_PATH     = "/data/cn_vocab.json"
MERGES_PATH    = "/data/cn_merges.json"
DATA_PATH      = "/data/cn_train_bpe.npy"
VAL_PATH       = "/data/cn_val_bpe.npy"
CKPT_DIR       = "/data/cn_bpe_checkpoints"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 加载 tokenizer ==========
def load_tokenizer():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)
    vocab = {int(k): bytes(v) for k, v in vocab_raw.items()}

    with open(MERGES_PATH, "r", encoding="utf-8") as f:
        merges_raw = json.load(f)
    merges = [(bytes(a), bytes(b)) for a, b in merges_raw]

    return Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

# ========== 评估函数 ==========
@torch.no_grad()
def evaluate(model, val_data, num_batches=20):
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = GetBatch(val_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
        logits = model(x)
        B, T, V = logits.shape
        loss = CrossEntropyLoss(logits.view(B * T, V), y.view(B * T))
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches

# ========== 生成文本 ==========
@torch.no_grad()
def generate(model, tokenizer, prompt="从前", max_new_tokens=150, temperature=0.8):
    model.eval()

    # 用 tokenizer 编码 prompt
    ids = tokenizer.encode(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    eot_id = tokenizer.encode("<|endoftext|>")[0]  # 文档结束符 ID

    for _ in range(max_new_tokens):
        ids_input = ids[:, -CONTEXT_LENGTH:]
        logits = model(ids_input)
        next_logit = logits[0, -1, :] / temperature
        probs = torch.softmax(next_logit, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)

        # 遇到文档结束符停止
        if next_id.item() == eot_id:
            break

    model.train()
    # 用 tokenizer 解码，不会出现乱码
    result = tokenizer.decode(ids[0].tolist())
    return result

# ========== 主训练循环 ==========
def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"使用设备：{DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(0)}")
        print(f"显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = load_tokenizer()
    print(f"词表大小：{len(tokenizer.vocab)}\n")

    # 加载数据
    print("加载数据...")
    train_data = np.load(DATA_PATH, mmap_mode='r')
    val_data   = np.load(VAL_PATH,  mmap_mode='r')
    print(f"训练集：{len(train_data):,} tokens")
    print(f"验证集：{len(val_data):,} tokens\n")

    # 初始化模型
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        rope_theta=ROPE_THETA,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{total_params:,}（约 {total_params/1e6:.2f}M）\n")

    # 初始化优化器
    optimizer = AdamW(
        model.parameters(),
        lr=LR_MAX,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )

    # 从 checkpoint 恢复
    start_step = 0
    ckpt_path = os.path.join(CKPT_DIR, "latest.pt")
    if os.path.exists(ckpt_path):
        start_step = LoadCheckpoint(ckpt_path, model, optimizer)
        print(f"从 checkpoint 恢复，继续从第 {start_step} 步\n")

    # ===== 训练循环 =====
    print("开始训练...")
    print(f"{'Step':>6}  {'Train Loss':>10}  {'Val Loss':>10}  {'LR':>10}  {'Time':>8}")
    print("-" * 55)

    t0 = time.time()
    model.train()

    for step in range(start_step, TOTAL_STEPS):

        # 1. 学习率调度
        lr = IrCosineSchedule(
            it=step,
            max_learning_rate=LR_MAX,
            min_learning_rate=LR_MIN,
            warmup_iters=WARMUP_STEPS,
            cosine_cycle_iters=TOTAL_STEPS,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 2. 取一个 batch
        # 注意：BPE 编码后是 uint16，GetBatch 会自动转成 long tensor
        x, y = GetBatch(train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)

        # 3. 前向传播
        logits = model(x)
        B, T, V = logits.shape
        train_loss = CrossEntropyLoss(logits.view(B * T, V), y.view(B * T))

        # 4. 反向传播
        optimizer.zero_grad()
        train_loss.backward()

        # 5. 梯度裁剪
        GradientClipping(model.parameters(), max_l2_norm=GRAD_CLIP)

        # 6. 更新参数
        optimizer.step()

        # 7. 定期评估
        if (step + 1) % EVAL_EVERY == 0 or step == 0:
            val_loss = evaluate(model, val_data)
            elapsed = time.time() - t0
            print(f"{step+1:>6}  {train_loss.item():>10.4f}  {val_loss:>10.4f}  {lr:>10.2e}  {elapsed:>6.1f}s")
            t0 = time.time()

        # 8. 定期保存 checkpoint
        if (step + 1) % SAVE_EVERY == 0:
            SaveCheckpoint(model, optimizer, step + 1, ckpt_path)
            print(f"  → checkpoint 已保存（step {step+1}）")

    # ===== 训练结束 =====
    print("\n训练完成！")
    SaveCheckpoint(model, optimizer, TOTAL_STEPS, ckpt_path)

    # 生成文本
    print("\n生成文本示例：")
    print("-" * 40)
    for prompt in ["从前", "他说道", "这一天"]:
        result = generate(model, tokenizer, prompt=prompt, max_new_tokens=150)
        print(f"Prompt: '{prompt}'")
        print(f"Output: {result}")
        print()

if __name__ == "__main__":
    main()
