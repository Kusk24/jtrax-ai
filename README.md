# JTrax chess models

This repository trains, evaluates, and exports the two learned opponents used by the JTrax web and mobile apps. Training uses public Lichess games. School and student records stay outside the training corpus.

| Opponent | Training | Browser export |
| --- | --- | --- |
| Novice | Character-level GPT trained from random initialization on lower-rated games | `results/novice_int8.onnx` |
| Strong | Maia-2 fine-tuned on higher-rated human games | `results/strong_fp16.onnx` |

On the same 50,176 held-out positions, stock Maia-2 matched the human move 50.89% of the time and the fine-tuned model matched it 52.56%. The [evaluation record](results/maia_heldout_movematch.json) identifies the checkpoint. [Novice](results/onnx_novice.json) and [strong](results/onnx_strong.json) export checks record how the compressed models performed.

## Set up

Use a Python 3.12 environment. From the repository root:

```sh
conda create -n jtrax-ai python=3.12
conda activate jtrax-ai
python -m pip install maia2 torch numpy python-chess requests zstandard onnx onnxruntime onnxconverter-common huggingface_hub
```

The Elo evaluation also needs a Stockfish executable. The [tracked results](results) can be inspected on a fresh clone. Training data, checkpoints, and exported model weights are separate artifacts.

## Follow a model through the pipeline

| Stage | Entry points |
| --- | --- |
| Build the corpus and held-out split | [Prepare public games](step4_data.py). Use separate rating bands and prefixes for each opponent. |
| Train | [Novice GPT](step5_train.py), [strong Maia-2](step7_maia_finetune.py) |
| Evaluate | [Play against Stockfish](step6_elo.py), [compare held-out moves](step8_maia_eval.py) |
| Export | [Novice INT8](step9_export_novice_onnx.py), [strong FP16](step10_export_strong_onnx.py) |

For example, inspect the data options with `python step4_data.py --help`. Keep the held-out split fixed when comparing training runs. The default exporters expect the novice checkpoint at `runs/ckpt_40000.pt`, the strong checkpoint at `runs_maia/maia2_ft_20000.pt`, and their corresponding held-out files under `data/`.

After export, follow the [web app's model setup](https://github.com/Kusk24/jtrax-web-app/blob/main/public/models/README.md) to stage the files. The vault records the [training and deployment decisions](https://github.com/Kusk24/jtrax-docs/blob/main/features/training-our-own-chess-opponents.md).
