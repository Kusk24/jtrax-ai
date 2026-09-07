# jtrax-ai

Training the three chess opponents the Play screen offers. Two of them are
models we trained; the third is Stockfish, already in the app.

House rules: `../CLAUDE.md`. The full story, decisions and measured numbers live
in the vault — `jtrax-docs/features/training-our-own-chess-opponents.md` wins if
this file disagrees with it.

## The three tiers

| Tier | Model | Artefact | Measured |
|---|---|---|---|
| expert | Stockfish | already in the app | dial to taste |
| strong | Maia-2 fine-tuned on 2000–2800 games | `runs_maia/maia2_ft_20000.pt` | 0.5089 → 0.5256 held-out move-match |
| novice | nanoGPT, ours, from random weights | `runs/ckpt_40000.pt` | 0.936 legal · ~520 Elo |

The novice model is a **character-level GPT over PGN text** — 25.7M params,
8 layers, n_embd 512, block_size 1023, vocab_size 32
(`' #+-.0123456789;=BKNOQRabcdefghx'`). It reads `;1.e4 e5 2.Nf3` and predicts
the next character; the rules emerge because tracking the board is the only way
to predict well. Needs `vendor/nanogpt_model.py` to load.

## What is already settled — do not redo

| Fact | Value |
|---|---|
| Env | conda **`jtrax-ai`**, py3.12 — system py3.13 is too new |
| Maia-2 reference move-match | 0.5311 (`results/baseline.json`) |
| Maia-2 Elo | **1300–1450** — measured 1430 and 1291 on identical settings |
| maia2 `from_checkpoint` / `train.run()` | a *resume* mechanism, not a fine-tune path — the released `.pt` has no `training_metadata` |
| Novice serving | int8, 26 MB, legality held (0.9449 vs 0.9363) |
| Strong serving | **float16, 47 MB** — int8 costs 0.0219 move-match, more than the fine-tune gained |

## Two traps that cost real time

- **`torch.cuda.is_bf16_supported()` returns True on a T4**, counting emulated
  bf16 with no tensor cores. Cost ~6x throughput. `step5_train.py:41` asks the
  compute capability instead. See `jtrax-docs/bugs/bf16-on-a-t4-is-emulated.md`.
- **Validation loss is a bad proxy for playing strength, in both directions.**
  It fell 0.041 with no gain in play between iterations 16k and 30k, then held
  flat while play improved between 30k and 40k. Read `step3_probe.py` and
  `step6_elo.py`, not the training curve.

## Pipeline

```
step1_baseline   Maia-2 reference move-match
step2_export_onnx  ONNX feasibility gate (stock Maia-2)
step3_probe      self-play legal-move rate
step3b_sweep     probe published checkpoints
step4_data       stream Lichess, filter by rating band, write PGN text
step5_train      the nanoGPT training loop (Kaggle T4, fp16)
step6_elo        play Stockfish at fixed strengths, derive Elo
step7_maia_finetune   fine-tune Maia-2 on the strong corpus
step8_maia_eval  held-out move-match, stock vs fine-tuned
step9_export_novice_onnx   novice -> ONNX + int8, gated on self-play legality
step10_export_strong_onnx  strong -> ONNX + fp16/uint8, gated on move-match
```

Both export steps gate on a **behavioural** metric, not numeric drift. Drift can
look fine while the argmax flips, and a flipped argmax is a different move.

## Constraints that do not bend

- **Students are children.** Their linked Lichess games never enter this repo, a
  Kaggle notebook, or any public dataset. Public Lichess dumps only.
- Weights, checkpoints and `.onnx` stay out of git — all reproducible.
  `results/*.json` **is** tracked: those are the measurements.
- Python, not pnpm. The one JTrax repo that is not JS.
- Compare quantised models against fp32 **on the same positions**. Scoring a
  small sample against a large stored reference understated int8's loss by half.

## Kaggle

30 GPU-h/week, 12 h hard session cap, T4 x2 or P100 only. Chaining sessions has
its own set of traps — the runbook is
`jtrax-docs/ops/resuming-a-kaggle-training-run.md`. Read it before starting a run
rather than after losing one.
