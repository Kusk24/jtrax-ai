# jtrax-ai

Fine-tuning **Maia-2** into a chess opponent that plays like a child at the
student's level, to replace dialed-down Stockfish in the JTrax Play screen.

House rules: `../CLAUDE.md`. Canonical background, the model comparison, and
the verified ONNX gate results live in the vault:
`jtrax-docs/research/fine-tuning-a-model-for-jtrax.md` — that note wins if this
file disagrees with it.

## Environment

System Python is 3.13; maia2 supports 3.10–3.12, so the conda env is required.

```bash
conda activate maia2      # every new terminal
```

Not pnpm — this is the one JTrax repo that is Python, not JS.

## What is verified

`step1_baseline.py` → 0.5311 move-match on the bundled example set (MPS).
`step2_export_onnx.py` → ONNX export matches PyTorch to 2.34e-05, 93.2 MB fp32.

`results/baseline.json` is tracked on purpose. Every later run is compared
against it, so it must never drift silently.

## Constraints that do not bend

- **Students are children.** Their linked Lichess games never go in a public
  Kaggle notebook or dataset, and never into this repo. Public Lichess dumps
  are fine.
- **Build the eval set before training.** Without a frozen held-out set, "did
  the fine-tune help?" has no answer.
- Weights and `.onnx` files stay out of git — see `.gitignore`.
