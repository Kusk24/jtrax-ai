# jtrax-ai

Fine-tuning a chess opponent for JTrax. The goal is an engine that **plays like a
kid at the student's level**, instead of Stockfish turned down — which plays
perfectly and then blunders at random, and reads as a computer to a child.

The model is **Maia-2** (CSSLab, University of Toronto). It is rating-conditioned:
one model serves every student level, because you pass the Elo in at inference
time rather than shipping one net per level.

Background and the option comparison live in the vault:
`jtrax-docs/research/fine-tuning-a-model-for-jtrax.md`.

## Setup — run once

Your system Python is 3.13; maia2 supports 3.10–3.12, so it needs its own env.

```bash
cd ~/Desktop/JTrax/jtrax-ai
conda create -n maia2 python=3.12 -y
conda activate maia2
pip install maia2 onnx onnxruntime
```

The `conda activate maia2` line is needed in **every new terminal window** before
running anything here.

## The two steps that come before Kaggle

Both run on this Mac (Apple Silicon, via MPS) and cost zero Kaggle GPU quota.
They exist to answer the two questions that decide whether this project is worth
starting at all.

### Step 1 — baseline

```bash
python step1_baseline.py
```

Downloads the pretrained Maia-2 and scores it on the bundled example set. Prints
a move-match accuracy number and writes it to `results/baseline.json`.

That number is the thing every later run gets compared to. Without it, "did the
fine-tune help?" has no answer.

### Step 2 — the ONNX gate

```bash
python step2_export_onnx.py
```

The whole plan assumes a fine-tuned Maia-2 can be exported to ONNX and run in the
browser next to the existing Stockfish WASM — free serving, no GPU server. But
**maia2's docs never mention ONNX**, so this is unverified. This script tries the
export and reports honestly whether it worked.

- **Export succeeds** → the plan is real, move on to the eval set and Kaggle.
- **Export fails** → stop and rethink serving before spending 30 GPU-hours. The
  fallback is an inference endpoint on the Go backend, which costs the
  free-tier-no-card rule.

Better to learn this in an evening than after a month of training runs.

## Only then: Kaggle

Kaggle is Step 3 onward — data prep and the actual fine-tune. Notes when you get
there:

- **Private notebooks only.** Students are children; their linked Lichess games
  never go in a public Kaggle notebook or dataset. Public Lichess dumps are fine.
- Pick **2×T4** over the P100 — same 1 hour of quota per wall-clock hour, double
  the compute.
- Do **not** download a raw monthly Lichess dump (~30 GB compressed, ~300 GB
  open). Stream-decompress and filter to the target rating band first, then save
  the filtered result as a private Kaggle Dataset so sessions don't re-download.
- Checkpoint every epoch to `/kaggle/working/`. Sessions die, and an
  un-checkpointed 8-hour run is 8 hours of a 30-hour weekly quota gone.
