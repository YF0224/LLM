from __future__ import annotations

import os
import time
import json
import argparse
import base64
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Import custom modules (package-relative)
from .pretokenization import train_bpe
from .tokenization import Tokenizer
from .transformer import TransformerLM
from .train import (
    AdamW,
    IrCosineSchedule,
    GradientClipping,
    GetBatch,
    CrossEntropyLoss,
    SaveCheckpoint,
    LoadCheckpoint,
)

# -----------------------------------------------------------------------------
# base64 helpers (bytes <-> safe ascii)
# -----------------------------------------------------------------------------

def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

ROOT = Path(__file__).resolve().parent.parent  # assignment1-basics/


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    """Central configuration for all experiments."""

    # Input data
    ROOT = ROOT
    TRAIN_DATA_PATH = ROOT / "tests" / "fixtures" / "tinystories_sample_5M.txt"
    VAL_DATA_PATH   = ROOT / "tests" / "fixtures" / "tinystories_sample.txt"

    # Output directories (ALL Path)
    OUTPUT_DIR = ROOT / "outputs"
    TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"
    MODEL_DIR = OUTPUT_DIR / "models"
    LOG_DIR = OUTPUT_DIR / "logs"
    GENERATION_DIR = OUTPUT_DIR / "generations"

    # Tokenizer files
    VOCAB_FILE = TOKENIZER_DIR / "vocab.json"
    MERGES_FILE = TOKENIZER_DIR / "merges.txt"

    # Tokenized data
    TRAIN_TOKENS_FILE = TOKENIZER_DIR / "train_tokens.npy"
    VAL_TOKENS_FILE = TOKENIZER_DIR / "val_tokens.npy"

    # Model checkpoint
    CHECKPOINT_FILE = MODEL_DIR / "checkpoint_final.pt"

    # Tokenizer hyperparams
    VOCAB_SIZE = 10_000
    SPECIAL_TOKENS = ["<|endoftext|>"]
    NUM_PROCESSES = 8

    # Model hyperparams (spec)
    CONTEXT_LENGTH = 256
    D_MODEL = 512
    D_FF = 1344
    NUM_LAYERS = 4
    NUM_HEADS = 16
    ROPE_THETA = 10000.0
    EPS = 1e-5

    # Training hyperparams
    MAX_LEARNING_RATE = 3e-4
    MIN_LEARNING_RATE = 3e-5
    BETA1 = 0.9
    BETA2 = 0.999
    ADAM_EPS = 1e-8
    WEIGHT_DECAY = 0.1

    BATCH_SIZE = 128
    TOTAL_TOKENS = 327_680_000
    WARMUP_ITERS = 1000
    MAX_GRAD_NORM = 1.0

    EVAL_INTERVAL = 500
    LOG_INTERVAL = 100
    SAVE_INTERVAL = 1000

    # Generation
    GEN_MAX_TOKENS = 256
    GEN_TEMPERATURE = 0.8
    GEN_TOP_P = 0.95
    GEN_PROMPTS = [
        "Once upon a time",
        "Once upon a time there was a little girl named",
        "The boy was very happy because",
    ]

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOW_RESOURCE_MODE = False

    @classmethod
    def apply_low_resource_mode(cls):
        cls.LOW_RESOURCE_MODE = True
        cls.TOTAL_TOKENS = 40_000_000
        cls.BATCH_SIZE = 32
        cls.EVAL_INTERVAL = 250
        cls.LOG_INTERVAL = 50
        cls.SAVE_INTERVAL = 500

    @classmethod
    def compute_training_steps(cls) -> int:
        tokens_per_step = cls.BATCH_SIZE * cls.CONTEXT_LENGTH
        return int(cls.TOTAL_TOKENS // tokens_per_step)

    @classmethod
    def print_config(cls):
        print("\n" + "=" * 80)
        print("CONFIGURATION SUMMARY")
        print("=" * 80)
        print(f"Device: {cls.DEVICE}")
        print(f"Low Resource Mode: {cls.LOW_RESOURCE_MODE}")
        print("\nData:")
        print(f"  Training data: {cls.TRAIN_DATA_PATH}")
        print(f"  Validation data: {cls.VAL_DATA_PATH}")
        print("\nTokenizer:")
        print(f"  Vocab size target: {cls.VOCAB_SIZE}")
        print(f"  Special tokens: {cls.SPECIAL_TOKENS}")
        print("\nModel:")
        print(f"  Context length: {cls.CONTEXT_LENGTH}")
        print(f"  d_model: {cls.D_MODEL}")
        print(f"  d_ff: {cls.D_FF}")
        print(f"  Layers: {cls.NUM_LAYERS}")
        print(f"  Heads: {cls.NUM_HEADS}")
        print("\nTraining:")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Total tokens: {cls.TOTAL_TOKENS:,}")
        print(f"  Total steps: {cls.compute_training_steps():,}")
        print(f"  Learning rate: {cls.MIN_LEARNING_RATE} -> {cls.MAX_LEARNING_RATE}")
        print(f"  Warmup iterations: {cls.WARMUP_ITERS}")
        print(f"  Weight decay: {cls.WEIGHT_DECAY}")
        print("=" * 80 + "\n")


# =============================================================================
# LOGGING
# =============================================================================

class ExperimentLogger:
    def __init__(self, log_dir: Path, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.jsonl"
        self.summary_file = self.log_dir / f"{experiment_name}_summary.txt"

        self.metrics = []
        self.start_time = time.time()

        with open(self.summary_file, "w", encoding="utf-8") as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, step: int, metrics: dict, print_to_console: bool = True):
        wallclock_time = time.time() - self.start_time
        entry = {"step": step, "wallclock_time": wallclock_time, **metrics}
        self.metrics.append(entry)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        if print_to_console:
            parts = []
            for k, v in metrics.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            print(f"Step {step} | Time {wallclock_time:.1f}s | " + ", ".join(parts))

    def save_summary(self, final_metrics: dict):
        with open(self.summary_file, "a", encoding="utf-8") as f:
            f.write("\nFinal Results:\n")
            f.write("-" * 80 + "\n")
            for k, v in final_metrics.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
            f.write(f"Total time: {time.time() - self.start_time:.2f}s\n")
            f.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


# =============================================================================
# TOKENIZER I/O (base64)
# =============================================================================

def save_tokenizer_base64(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], vocab_path: Path, merges_path: Path):
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    # vocab.json : {"b64(token_bytes)": id}
    vocab_json = {_b64e(tok): int(i) for i, tok in vocab.items()}
    vocab_path.write_text(json.dumps(vocab_json, ensure_ascii=False), encoding="utf-8")

    # merges.txt : "b64(a) b64(b)\n"
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(_b64e(a) + " " + _b64e(b) + "\n")

def load_tokenizer_base64(vocab_path: Path, merges_path: Path, special_tokens: Optional[list[str]] = None) -> Tokenizer:
    vocab_data = json.loads(vocab_path.read_text(encoding="utf-8"))

    # vocab_data: {"b64(bytes)": id}
    vocab: dict[int, bytes] = {int(v): _b64d(k) for k, v in vocab_data.items()}

    merges: list[tuple[bytes, bytes]] = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a_b64, b_b64 = line.split()  # base64 tokens never contain spaces
            merges.append((_b64d(a_b64), _b64d(b_b64)))

    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


# =============================================================================
# TOKENIZER TRAINING + TOKENIZE DATA
# =============================================================================

def train_tokenizer(config: Config):
    print("\n" + "=" * 80)
    print("TRAINING BPE TOKENIZER")
    print("=" * 80)

    config.TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training BPE with vocab_size={config.VOCAB_SIZE}...")
    print(f"Input file: {config.TRAIN_DATA_PATH}")

    vocab, merges = train_bpe(
        input_path=str(config.TRAIN_DATA_PATH),
        vocab_size=int(config.VOCAB_SIZE),
        special_tokens=list(config.SPECIAL_TOKENS),
        num_processes=int(config.NUM_PROCESSES),
    )

    print(f"Trained tokenizer with {len(vocab)} tokens")

    save_tokenizer_base64(vocab, merges, config.VOCAB_FILE, config.MERGES_FILE)
    print(f"Saved vocab to: {config.VOCAB_FILE}")
    print(f"Saved merges to: {config.MERGES_FILE}")

    return vocab, merges


def tokenize_dataset(config: Config, tokenizer: Tokenizer):
    print("\n" + "=" * 80)
    print("TOKENIZING DATASETS")
    print("=" * 80)

    # training data
    print(f"Tokenizing training data: {config.TRAIN_DATA_PATH}")
    train_text = config.TRAIN_DATA_PATH.read_text(encoding="utf-8", errors="replace")
    train_tokens = np.fromiter(tokenizer.encode(train_text), dtype=np.int32)
    if train_tokens.max() < (1 << 16):
        train_tokens = train_tokens.astype(np.uint16)

    config.TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.TRAIN_TOKENS_FILE, train_tokens)
    print(f"Training tokens: {len(train_tokens):,}")
    print(f"Saved to: {config.TRAIN_TOKENS_FILE}")

    # validation data
    print(f"\nTokenizing validation data: {config.VAL_DATA_PATH}")
    val_text = config.VAL_DATA_PATH.read_text(encoding="utf-8", errors="replace")
    val_tokens = np.fromiter(tokenizer.encode(val_text), dtype=np.int32)
    if val_tokens.max() < (1 << 16):
        val_tokens = val_tokens.astype(np.uint16)

    np.save(config.VAL_TOKENS_FILE, val_tokens)
    print(f"Validation tokens: {len(val_tokens):,}")
    print(f"Saved to: {config.VAL_TOKENS_FILE}")

    return train_tokens, val_tokens


# =============================================================================
# MODEL INIT
# =============================================================================

def initialize_model(config: Config, vocab_size: int) -> TransformerLM:
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)

    model = TransformerLM(
        vocab_size=int(vocab_size),
        context_length=int(config.CONTEXT_LENGTH),
        d_model=int(config.D_MODEL),
        num_layers=int(config.NUM_LAYERS),
        num_heads=int(config.NUM_HEADS),
        d_ff=int(config.D_FF),
        rope_theta=float(config.ROPE_THETA),
        eps=float(config.EPS),
        device=config.DEVICE,
        dtype=torch.float32,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


# =============================================================================
# TRAINING
# =============================================================================

@torch.no_grad()
def evaluate_model(model, val_data: np.ndarray, config: Config, num_batches: int = 20) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = GetBatch(val_data, config.BATCH_SIZE, config.CONTEXT_LENGTH, config.DEVICE)
        logits = model(x)
        B, T, V = logits.shape
        loss = CrossEntropyLoss(logits.view(B * T, V), y.view(B * T))
        total_loss += float(loss.item())
    model.train()
    return total_loss / num_batches


def train_model(config: Config):
    # dirs
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    experiment_name = f"transformer_lm_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = ExperimentLogger(config.LOG_DIR, experiment_name)

    # load tokenizer (base64)
    print("\n" + "=" * 80)
    print("LOADING TOKENIZER")
    print("=" * 80)
    tokenizer = load_tokenizer_base64(config.VOCAB_FILE, config.MERGES_FILE, special_tokens=config.SPECIAL_TOKENS)
    vocab_size = len(tokenizer.vocab)
    print(f"Loaded tokenizer. vocab size = {vocab_size}, merges = {len(tokenizer.merges)}")

    # load tokenized data
    print("\n" + "=" * 80)
    print("LOADING TOKENIZED DATA")
    print("=" * 80)
    train_data = np.load(config.TRAIN_TOKENS_FILE)
    val_data = np.load(config.VAL_TOKENS_FILE)
    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")

    # model
    model = initialize_model(config, vocab_size=vocab_size).to(config.DEVICE)
    model.train()

    # optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.MAX_LEARNING_RATE),
        betas=(float(config.BETA1), float(config.BETA2)),
        eps=float(config.ADAM_EPS),
        weight_decay=float(config.WEIGHT_DECAY),
    )

    total_steps = config.compute_training_steps()
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Total steps: {total_steps:,}")
    print(f"Tokens per step: {config.BATCH_SIZE * config.CONTEXT_LENGTH:,}")
    print(f"Total tokens: {total_steps * config.BATCH_SIZE * config.CONTEXT_LENGTH:,}")
    print("=" * 80 + "\n")

    # resume (optional)
    start_step = 1
    if config.CHECKPOINT_FILE.exists():
        try:
            last = LoadCheckpoint(str(config.CHECKPOINT_FILE), model, optimizer)
            start_step = int(last) + 1
            print(f"[ckpt] resumed from step {last}")
        except Exception as e:
            print(f"[ckpt] failed to resume, starting fresh: {e}")

    start_time = time.time()

    for step in range(start_step, total_steps + 1):
        lr = IrCosineSchedule(
            it=step,
            max_learning_rate=float(config.MAX_LEARNING_RATE),
            min_learning_rate=float(config.MIN_LEARNING_RATE),
            warmup_iters=int(config.WARMUP_ITERS),
            cosine_cycle_iters=int(total_steps),
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = GetBatch(train_data, config.BATCH_SIZE, config.CONTEXT_LENGTH, config.DEVICE)
        logits = model(x)
        B, T, V = logits.shape
        loss = CrossEntropyLoss(logits.view(B * T, V), y.view(B * T))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        GradientClipping(model.parameters(), float(config.MAX_GRAD_NORM))
        optimizer.step()

        if step % config.LOG_INTERVAL == 0:
            logger.log(step, {"train_loss": float(loss.item()), "learning_rate": float(lr)})

        if step % config.EVAL_INTERVAL == 0:
            val_loss = evaluate_model(model, val_data, config)
            logger.log(step, {"train_loss": float(loss.item()), "val_loss": float(val_loss), "learning_rate": float(lr)})
            print(f"  -> Validation loss: {val_loss:.4f}")

        if step % config.SAVE_INTERVAL == 0:
            ckpt_path = config.MODEL_DIR / f"checkpoint_step_{step}.pt"
            SaveCheckpoint(model, optimizer, step, str(ckpt_path))
            print(f"  -> Saved checkpoint: {ckpt_path}")

    SaveCheckpoint(model, optimizer, total_steps, str(config.CHECKPOINT_FILE))
    print(f"\n✓ Saved final checkpoint: {config.CHECKPOINT_FILE}")

    final_val_loss = evaluate_model(model, val_data, config, num_batches=50)

    total_time = time.time() - start_time
    logger.save_summary({
        "total_steps": total_steps,
        "final_val_loss": final_val_loss,
        "total_time_seconds": total_time,
        "total_time_minutes": total_time / 60,
        "vocab_size": vocab_size,
    })

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Logs saved to: {logger.log_file}")
    print("=" * 80 + "\n")

    return model, logger


# =============================================================================
# GENERATION
# =============================================================================

def sample_top_p(logits: torch.Tensor, top_p: float = 1.0, temperature: float = 1.0) -> torch.Tensor:
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    cutoff = torch.where(cumulative > top_p)[0]
    if len(cutoff) > 0:
        cutoff_idx = max(1, int(cutoff[0].item()))
    else:
        cutoff_idx = len(sorted_probs)

    nucleus_probs = sorted_probs[:cutoff_idx]
    nucleus_indices = sorted_indices[:cutoff_idx]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()

    sampled = torch.multinomial(nucleus_probs, num_samples=1)
    return nucleus_indices[sampled]


@torch.no_grad()
def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> str:
    model.eval()

    input_ids = tokenizer.encode(prompt) if prompt else []

    endoftext_id = None
    if "<|endoftext|>" in tokenizer.special_tokens:
        eos_b = "<|endoftext|>".encode("utf-8")
        endoftext_id = tokenizer.byte_encoder.get(eos_b)

    generated = list(input_ids)

    for _ in range(max_tokens):
        ctx = generated[-model.context_length:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]

        next_id = int(sample_top_p(logits, top_p=top_p, temperature=temperature).item())
        if endoftext_id is not None and next_id == endoftext_id:
            break
        generated.append(next_id)

    return tokenizer.decode(generated)


def run_generation(config: Config):
    print("\n" + "=" * 80)
    print("TEXT GENERATION")
    print("=" * 80)

    config.GENERATION_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer_base64(config.VOCAB_FILE, config.MERGES_FILE, special_tokens=config.SPECIAL_TOKENS)
    vocab_size = len(tokenizer.vocab)

    model = initialize_model(config, vocab_size=vocab_size)
    checkpoint = torch.load(str(config.CHECKPOINT_FILE), map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model = model.to(config.DEVICE)
    model.eval()

    print(f"Loaded checkpoint from iteration {checkpoint.get('iteration', 'unknown')}")

    output_file = config.GENERATION_DIR / f"generations_{time.strftime('%Y%m%d_%H%M%S')}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TEXT GENERATION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Temperature: {config.GEN_TEMPERATURE}\n")
        f.write(f"Top-p: {config.GEN_TOP_P}\n")
        f.write(f"Max tokens: {config.GEN_MAX_TOKENS}\n")
        f.write("=" * 80 + "\n\n")

        for i, prompt in enumerate(config.GEN_PROMPTS, 1):
            print(f"\nGenerating from prompt {i}/{len(config.GEN_PROMPTS)}:")
            print(f"Prompt: {repr(prompt)}")

            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=int(config.GEN_MAX_TOKENS),
                temperature=float(config.GEN_TEMPERATURE),
                top_p=float(config.GEN_TOP_P),
                device=config.DEVICE,
            )

            print(f"\nGenerated text ({len(tokenizer.encode(generated))} tokens):")
            print("-" * 80)
            print(generated)
            print("-" * 80)

            f.write(f"PROMPT {i}: {prompt}\n")
            f.write("-" * 80 + "\n")
            f.write(generated + "\n\n")
            f.write("=" * 80 + "\n\n")

    print(f"\n✓ Saved all generations to: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CS336 Assignment 1 - Complete Training and Generation")
    parser.add_argument("--mode", type=str, choices=["all", "tokenizer", "train", "generate"], default="all")
    parser.add_argument("--low_resource", action="store_true")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    if args.low_resource or Config.DEVICE in ["cpu", "mps"]:
        Config.apply_low_resource_mode()

    if args.device:
        Config.DEVICE = args.device

    Config.print_config()

    # tokenizer + tokenize
    if args.mode in ["all", "tokenizer"]:
        if not Config.VOCAB_FILE.exists() or not Config.MERGES_FILE.exists():
            train_tokenizer(Config)

        tokenizer = load_tokenizer_base64(Config.VOCAB_FILE, Config.MERGES_FILE, special_tokens=Config.SPECIAL_TOKENS)

        if (not Config.TRAIN_TOKENS_FILE.exists()) or (not Config.VAL_TOKENS_FILE.exists()):
            tokenize_dataset(Config, tokenizer)
        else:
            print(f"Tokenized data already exists at: {Config.TRAIN_TOKENS_FILE} / {Config.VAL_TOKENS_FILE}")

    # train
    if args.mode in ["all", "train"]:
        train_model(Config)

    # generate
    if args.mode in ["all", "generate"]:
        if not Config.CHECKPOINT_FILE.exists():
            print(f"ERROR: No checkpoint found at {Config.CHECKPOINT_FILE}")
            print("Please train the model first with --mode train")
            return
        run_generation(Config)

    print("\n" + "=" * 80)
    print("ALL DONE! 🎉")
    print("=" * 80)
    print(f"Outputs saved to: {Config.OUTPUT_DIR}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
