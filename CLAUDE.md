# jtrax-ai

Fine-tuning a small, weak, public chess model into a measurably better one.
The deliverable is the improved model and the before/after numbers — not a
feature shipped to students.

House rules: `../CLAUDE.md`. Background and the model comparison live in the
vault: `jtrax-docs/research/fine-tuning-a-model-for-jtrax.md` — that note wins
if this file disagrees with it.

## What is being trained

`lichess_6layers` from `adamkarvonen/chess_llms` — 1.3M params, 6 layers,
n_embd 128, **character-level over PGN text** (vocab_size 32, block_size 1023).
Needs nanoGPT's `GPT` class to load; the checkpoint is weights only.

Not Maia. Maia-2 is the benchmark and is finished — see below.

## What is already settled — do not redo

| Fact | Value |
|---|---|
| Maia-2 reference move-match | **0.5311** (`results/baseline.json`) |
| Maia-2 → ONNX export | works, 93.2 MB fp32, drift 2.34e-05 |
| maia2 `from_checkpoint` | **is a resume mechanism, not a fine-tune path** — the released `.pt` has no `training_metadata`, so it fails validation |
| Maia-2 training scale | 337,855,102 games / 18.3B samples |
| Env | conda `maia2`, py3.12 — system py3.13 is too new |

`step1_baseline.py` and `step2_export_onnx.py` produced these. They do not need
running again; downloads are cached.

## Constraints that do not bend

- **Students are children.** Their linked Lichess games never enter this repo,
  a Kaggle notebook, or any public dataset. Public Lichess dumps are fine.
- **Freeze the held-out eval set before training.** Without it there is no
  answer to "did the fine-tune help?"
- Weights, checkpoints, `.onnx` stay out of git — all re-downloadable in one
  line. `results/baseline.json` is tracked on purpose.
- Python, not pnpm. The one JTrax repo that is not JS.

## Kaggle, when it is needed

Not needed yet — a 1.3M model trains on the M2. If it scales up: private
notebooks only, 2×T4 costs the same quota as the P100 for double the compute,
never download a raw monthly Lichess dump (~30 GB zst / ~300 GB open) —
stream-filter first, and checkpoint every epoch to `/kaggle/working/`.
