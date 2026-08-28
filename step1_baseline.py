"""Step 1 — score the stock Maia-2 and record the number everything else is
measured against. Runs on this Mac (MPS); costs no Kaggle GPU quota.

Run:  conda activate jtrax-ai && python step1_baseline.py
"""

import json
import pathlib
import sys

RESULTS = pathlib.Path(__file__).parent / "results"


def main() -> int:
    try:
        from maia2 import dataset, inference, model
    except ImportError:
        print("maia2 not installed. Run:\n"
              "  conda activate jtrax-ai && pip install maia2 onnx onnxruntime")
        return 1

    import torch

    # "auto" picks MPS on Apple Silicon. Printed because a silent fallback to
    # CPU is the difference between two minutes and twenty.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"torch {torch.__version__} · device: {device}")

    print("\nLoading pretrained Maia-2 (downloads on first run)...")
    maia2_model = model.from_pretrained(type="rapid", device="auto")

    print("Loading the bundled example test set...")
    data = dataset.load_example_test_dataset()

    print("Running batch inference...\n")
    data, accuracy = inference.inference_batch(
        data, maia2_model, verbose=1, batch_size=1024, num_workers=0
    )

    print(f"\n{'=' * 52}")
    print(f"  BASELINE move-match accuracy: {accuracy}")
    print(f"{'=' * 52}")
    print("\nThis is the stock model on maia2's OWN example set — a sanity check,")
    print("not the real baseline. The real one comes from held-out games at the")
    print("rating bands JCA actually teaches (Step 2 of the plan).")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "baseline.json"
    out.write_text(json.dumps({
        "model": "maia2 pretrained (type=rapid)",
        "dataset": "maia2 bundled example test set",
        "accuracy": float(accuracy),
        "device": device,
        "torch": torch.__version__,
    }, indent=2) + "\n")
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
