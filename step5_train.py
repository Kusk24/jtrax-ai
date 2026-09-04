"""Step 5 — train the model. This is the one that takes hours.

Starts from `ckpt_iter_0.pt`: 25M parameters of random weights that score 0.000
on legal moves. Training teaches it chess from nothing.

Karvonen's own run reached 0.940 legal by iteration 20,000, so that is the
target — not the 600,000 he eventually ran to. Most of the learning happens
early; the rest is refinement we do not need.

Runs on a Kaggle T4 (~4-8 hours) or slowly on the Mac. Paths auto-detect
Kaggle, so the same file works in both places.

Checkpoints every --ckpt-every iterations. Kaggle sessions die; an
un-checkpointed 6-hour run is 6 hours of a 30-hour weekly quota gone.

Run:  conda activate jtrax-ai && python step5_train.py --help
"""

import argparse
import json
import math
import pathlib
import pickle
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE / "vendor"))

ON_KAGGLE = pathlib.Path("/kaggle/working").exists()
OUT_DIR = pathlib.Path("/kaggle/working") if ON_KAGGLE else HERE / "runs"

# From the sweep: 0.000 at iter 0, 0.940 by iter 20k, plateau after 40k.
DEFAULT_ITERS = 40_000
BLOCK = 1023  # the checkpoint's block_size; changing it breaks the weights


def pick_precision(device):
    """Mixed precision where the hardware supports it.

    fp32 on a T4 leaves ~3x on the table — its tensor cores are built for fp16.
    bf16 needs Ampere or newer (Kaggle's T4 is Turing, P100 is Pascal), so fp16
    plus a gradient scaler is the realistic path on free Kaggle GPUs.
    """
    if device != "cuda":
        return torch.float32, False
    # Ask the compute capability, NOT torch.cuda.is_bf16_supported(): that
    # returns True on a T4 because it counts *emulated* bf16, which has no
    # tensor-core path and ran ~9x slower than fp16 on Kaggle.
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:  # Ampere and newer: real bf16 tensor cores, no scaler needed
        return torch.bfloat16, False
    if major == 7:  # Turing (T4): fp16 is the fast path
        return torch.float16, True
    return torch.float32, False  # Pascal (P100): fp16 is not a win


def newest_checkpoint(directory):
    """Highest-numbered ckpt_N.pt below a directory, or None.

    Recursive, because Kaggle nests attached datasets a few levels under
    /kaggle/input and the depth depends on how the dataset was created.
    Names without a step number are skipped: ckpt_iter_0.pt is the random
    starting point, and "resuming" from it silently restarts the run.
    """
    root = pathlib.Path(directory)
    if not root.exists():
        return None
    found = []
    for p in root.rglob("ckpt_*.pt"):
        try:
            found.append((int(p.stem.split("_")[1]), p))
        except (IndexError, ValueError):
            continue
    return str(max(found, key=lambda pair: pair[0])[1]) if found else None


def load_tokens(path, stoi):
    """Whole corpus as one uint16 array. ';' already delimits games, and '\\n'
    is not in the 32-char vocabulary, so it is stripped rather than encoded."""
    text = pathlib.Path(path).read_text().replace("\n", "")
    return np.array([stoi[c] for c in text], dtype=np.uint16)


def get_batch(tokens, batch_size, block, device):
    ix = torch.randint(len(tokens) - block - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(tokens[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(tokens[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def lr_at(it, base_lr, warmup, total):
    """Linear warmup then cosine decay to 10% — nanoGPT's schedule."""
    if it < warmup:
        return base_lr * (it + 1) / (warmup + 1)
    if it > total:
        return base_lr * 0.1
    ratio = (it - warmup) / max(total - warmup, 1)
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * ratio)))


@torch.no_grad()
def estimate_loss(model, splits, batch_size, device, dtype, iters=20):
    model.eval()
    out = {}
    for name, toks in splits.items():
        if toks is None or len(toks) < BLOCK + 2:
            continue
        losses = []
        for _ in range(iters):
            x, y = get_batch(toks, batch_size, BLOCK, device)
            with torch.autocast(device_type=device, dtype=dtype,
                                enabled=dtype != torch.float32):
                _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default=str(HERE / "hf_ckpts" / "ckpt_iter_0.pt"),
                    help="checkpoint to start from (iter_0 = random weights)")
    ap.add_argument("--data", default=str(HERE / "data" / "train.txt"))
    ap.add_argument("--val", default=str(HERE / "data" / "heldout.txt"))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--batch-size", type=int, default=24,
                    help="24 fits a 16 GB T4 at block 1023; lower it on OOM")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--resume", default=None,
                    help="checkpoint to resume from; a directory resumes from "
                         "the highest ckpt_N.pt inside it")
    ap.add_argument("--auto-resume", action="store_true",
                    help="resume from the newest checkpoint in --out if one "
                         "exists, otherwise start from --init. Safe to leave "
                         "on: a re-run continues instead of restarting.")
    args = ap.parse_args()

    from nanogpt_model import GPT, GPTConfig

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} · output: {out_dir}")
    if device == "cpu":
        print("  WARNING: CPU training will take days. Use Kaggle's T4.")

    meta = pickle.loads((HERE / "vendor" / "meta.pkl").read_bytes())
    stoi = meta["stoi"]

    print(f"loading {args.data} ...")
    train_tokens = load_tokens(args.data, stoi)
    val_path = pathlib.Path(args.val)
    val_tokens = load_tokens(val_path, stoi) if val_path.exists() else None
    print(f"  train {len(train_tokens):,} tokens"
          + (f" · held out {len(val_tokens):,}" if val_tokens is not None else ""))

    # A Kaggle session can die at any point, so resuming has to be the easy
    # path rather than something you remember to do. --auto-resume picks up the
    # newest checkpoint without being told which one.
    resume_from = args.resume
    if resume_from and pathlib.Path(resume_from).is_dir():
        resume_from = newest_checkpoint(resume_from)
    if not resume_from and args.auto_resume:
        resume_from = newest_checkpoint(out_dir)
        # Every Kaggle session gets a fresh, empty /kaggle/working, so a
        # checkpoint from last session can only be an attached input dataset.
        # Without this the flag looks like it works and quietly restarts at 0.
        if not resume_from and ON_KAGGLE:
            resume_from = newest_checkpoint("/kaggle/input")
        if resume_from:
            print(f"auto-resume: found {pathlib.Path(resume_from).name}")
        else:
            print("auto-resume: no checkpoint yet, starting from scratch")

    src = resume_from or args.init
    print(f"initialising from {pathlib.Path(src).name}")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)
    start_it = 0
    if resume_from:
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_it = ckpt.get("iter_num", 0)
        print(f"  resumed at iteration {start_it:,}")

    dtype, use_scaler = pick_precision(device)
    scaler = torch.amp.GradScaler(device, enabled=use_scaler)
    print(f"precision: {str(dtype).replace('torch.', '')}"
          + (" + grad scaler" if use_scaler else ""))

    tokens_per_iter = args.batch_size * args.grad_accum * BLOCK
    epochs = (args.iters * tokens_per_iter) / max(len(train_tokens), 1)
    print(f"\n{args.iters:,} iterations · batch {args.batch_size}"
          f" x accum {args.grad_accum} · {tokens_per_iter:,} tokens/iter")
    print(f"total {(args.iters - start_it) * tokens_per_iter / 1e9:.1f}B tokens"
          f" · {epochs:.1f} passes over the corpus")
    if epochs > 15:
        print("  NOTE: more than ~15 passes risks memorising rather than "
              "learning. More games, or fewer iterations.")
    print()

    history = []
    t0 = time.time()
    model.train()
    for it in range(start_it, args.iters):
        lr = lr_at(it, args.lr, args.warmup, args.iters)
        for group in opt.param_groups:
            group["lr"] = lr

        # Gradient accumulation: several small batches before one step, so the
        # effective batch is large without needing the memory for it.
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch(train_tokens, args.batch_size, BLOCK, device)
            with torch.autocast(device_type=device, dtype=dtype,
                                enabled=dtype != torch.float32):
                _, loss = model(x, y)
            scaler.scale(loss / args.grad_accum).backward()
        # Unscale before clipping, or the threshold applies to scaled grads.
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if it % args.eval_every == 0 or it == args.iters - 1:
            losses = estimate_loss(model, {"train": train_tokens,
                                           "val": val_tokens},
                                   args.batch_size, device, dtype)
            elapsed = time.time() - t0
            done = max(it - start_it, 1)
            eta = elapsed / done * (args.iters - it) / 60
            msg = " · ".join(f"{k} {v:.4f}" for k, v in losses.items())
            print(f"  iter {it:6,} · lr {lr:.2e} · {msg} · "
                  f"{elapsed / 60:.1f}m elapsed · ~{eta:.0f}m left", flush=True)
            history.append({"iter": it, "lr": lr, **losses})

        if it > start_it and (it % args.ckpt_every == 0 or it == args.iters - 1):
            path = out_dir / f"ckpt_{it}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "model_args": ckpt["model_args"],
                "iter_num": it,
                "config": {"lr": args.lr, "batch_size": args.batch_size,
                           "grad_accum": args.grad_accum, "data": args.data},
            }, path)
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))
            print(f"    saved {path.name}", flush=True)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes.")
    print(f"Checkpoints in {out_dir}")
    print("\nNext: run step3_probe.py --ckpt <final checkpoint> and compare "
          "against results/sweep_checkpoints.json (iter 0 = 0.000 legal).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
