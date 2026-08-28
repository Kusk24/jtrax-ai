# jtrax-ai

Take a small, weak, public chess model and fine-tune it into a measurably
better one. The point is the fine-tune itself — an improved model that is ours,
rather than only ever calling someone else's.

**Base model:** `lichess_6layers` from [adamkarvonen/chess_llms][hf] — 1.3M
parameters, 6 layers, 128-dim embedding, character-level over PGN text.

```
model_args: {n_layer: 6, n_head: 8, n_embd: 128, block_size: 1023, vocab_size: 32}
params: 1.3M · iter_num: 600,000 · best_val_loss: 0.3415
```

Chosen because it is genuinely weak (so there is room to improve), obscure
(everyone else uses the 8- and 16-layer), reads chess as plain text (so data
prep is a text file, not a tensor encoder), and is small enough to train on a
MacBook.

**Data:** ~10,000 Lichess games, one rating band, public — never students'
games.

**Metrics:** move-match accuracy and legal-move rate, before vs. after.

[hf]: https://huggingface.co/adamkarvonen/chess_llms

## The benchmark: Maia-2

Maia-2 is **not** the model being trained. It is the reference number, measured
once so results have a scale to sit on:

| | move-match |
|---|---|
| Maia-2 (23.3M params, 338M games) | **0.5311** |
| `lichess_6layers`, before fine-tune | not yet measured |
| `lichess_6layers`, after fine-tune | — |

"I got 0.41" means nothing alone. "0.28 → 0.41, where state of the art is
0.531" is a result. `step1_baseline.py` and `step2_export_onnx.py` produced
that reference and the ONNX serving proof; neither needs running again.

## Setup — run once

System Python is 3.13; maia2 needs 3.10–3.12, so the env is required.

```bash
cd ~/Desktop/JTrax/jtrax-ai
conda create -n jtrax-ai python=3.12 -y
conda activate jtrax-ai
pip install maia2 onnx onnxruntime huggingface_hub
```

`conda activate jtrax-ai` is needed in **every new terminal**.

## Steps

| Script | Status | What it does |
|---|---|---|
| `step1_baseline.py` | done — 0.5311 | Maia-2 reference number |
| `step2_export_onnx.py` | done — 93.2 MB, 2.3e-05 drift | proves browser serving works |
| `step3_probe.py` | done — `lichess_6layers` 0.848 legal | play self-play games, measure legality |
| `step3b_sweep.py` | written, not run | probe early training checkpoints to pick a worse base |
| `step4_data.py` | not written | pull games at one rating band, train/held-out split |
| `step5_train.py` | not written | the fine-tune |
| `step6_eval.py` | not written | all three metrics, before vs after |

## The three eval metrics

A model can follow the rules and still play terribly, so one number is not
enough. Measured before and after, on the same frozen set:

| Metric | Question | Needs |
|---|---|---|
| **legal-move rate** | does it follow the rules? | self-play only |
| **move-match accuracy** | does it play what a human at this level played? | held-out human games |
| **average centipawn loss** | how good are the moves? | Stockfish (`brew install stockfish`) |

ACPL is the standard chess measure of play quality: for each move, how much
worse was it than the engine's best, in hundredths of a pawn. Lower is better —
a strong club player is roughly 20–40, a beginner well over 100.

Steps 1–2 are complete and should not need re-running; the downloads are
cached.

## Rules that do not bend

- **Public Lichess data only.** Students are children; their linked games never
  enter this repo, a Kaggle notebook, or a public dataset.
- **Freeze the held-out set before training.** Without it, "did it improve?"
  has no answer.
- Weights, checkpoints and `.onnx` files stay out of git — see `.gitignore`.
  Everything there is re-downloadable in one line.
- `results/baseline.json` **is** tracked. It must not drift silently.
