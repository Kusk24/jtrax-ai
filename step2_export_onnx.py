"""Step 2 — the gate. Can Maia-2 be exported to ONNX?

The whole plan rests on serving the fine-tuned model in the browser with
onnxruntime-web, next to the Stockfish WASM already in the Play screen — free
serving, no GPU box. maia2's docs never mention ONNX, so that assumption needed
checking before any training time was spent on it.

Signature below is read from the installed package (maia2/main.py), not guessed:

    MAIA2Model.forward(boards, elos_self, elos_oppo)
        boards     float32, viewed internally as (batch, 18, 8, 8)
        elos_self  int64 bucket index, 11 buckets ('<1100' … '2000+')
        elos_oppo  int64 bucket index
      returns (logits_maia, logits_side_info, logits_value)

`logits_maia` is the move policy — the head that matters for the opponent.
`elos_self` is what makes one model serve every student level: pass the pupil's
band instead of shipping a separate net per level.

Run:  conda activate jtrax-ai && python step2_export_onnx.py
"""

import pathlib
import sys

OUT = pathlib.Path(__file__).parent / "results" / "maia2.onnx"

INPUT_CHANNELS = 18  # maia2/configs/maia2-training.yaml
ELO_BUCKETS = 11  # len(maia2.utils.create_elo_dict())


def main() -> int:
    try:
        import torch
        from maia2 import model
    except ImportError as exc:
        print(f"Missing dependency ({exc}). Run:\n"
              "  conda activate jtrax-ai && pip install maia2 onnx onnxruntime")
        return 1

    # Export traces on CPU regardless of where the model will eventually run.
    print("Loading pretrained Maia-2 on CPU...")
    net = model.from_pretrained(type="rapid", device="cpu")
    net.eval()

    n_params = sum(p.numel() for p in net.parameters())
    print(f"  {type(net).__name__} · {n_params / 1e6:.1f}M parameters")

    # Batch of 1 — the browser evaluates one position per move. dynamic_axes
    # below keeps larger batches valid for the offline eval harness.
    boards = torch.zeros(1, INPUT_CHANNELS, 8, 8, dtype=torch.float32)
    elos_self = torch.zeros(1, dtype=torch.long)
    elos_oppo = torch.zeros(1, dtype=torch.long)

    print("\nSanity-checking a forward pass in PyTorch first...")
    try:
        with torch.no_grad():
            ref = net(boards, elos_self, elos_oppo)
    except Exception as exc:
        print(f"  forward() failed before export: {type(exc).__name__}: {exc}")
        return 2
    print(f"  ok — {len(ref)} outputs: {[tuple(t.shape) for t in ref]}")

    OUT.parent.mkdir(exist_ok=True)
    print(f"\nExporting to {OUT.name} ...")
    try:
        with torch.no_grad():
            torch.onnx.export(
                net,
                (boards, elos_self, elos_oppo),
                str(OUT),
                opset_version=17,
                input_names=["boards", "elos_self", "elos_oppo"],
                output_names=["logits_maia", "logits_side_info", "logits_value"],
                dynamic_axes={
                    "boards": {0: "batch"},
                    "elos_self": {0: "batch"},
                    "elos_oppo": {0: "batch"},
                    "logits_maia": {0: "batch"},
                },
            )
    except Exception as exc:
        print(f"\n{'=' * 54}")
        print("  EXPORT FAILED")
        print(f"{'=' * 54}")
        print(f"\n{type(exc).__name__}: {exc}")
        print("\nThis is the answer we came for. Free browser serving is not")
        print("available without more work — settle serving before training.")
        return 3

    print("Verifying the exported graph loads and runs under onnxruntime...")
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort

        onnx.checker.check_model(str(OUT))
        sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
        outputs = sess.run(None, {
            "boards": boards.numpy(),
            "elos_self": elos_self.numpy(),
            "elos_oppo": elos_oppo.numpy(),
        })
        # An export that runs but disagrees with PyTorch is worse than one that
        # fails, because nothing downstream would notice.
        drift = float(np.abs(outputs[0] - ref[0].numpy()).max())
    except Exception as exc:
        print(f"\nExported, but onnxruntime could not run it: "
              f"{type(exc).__name__}: {exc}")
        return 4

    size_mb = OUT.stat().st_size / 1_000_000
    print(f"\n{'=' * 54}")
    print("  EXPORT OK")
    print(f"{'=' * 54}")
    print(f"  file          {OUT.name}  ({size_mb:.1f} MB)")
    print(f"  policy shape  {outputs[0].shape}")
    print(f"  max drift vs PyTorch  {drift:.2e}")

    if drift > 1e-3:
        print("\n  WARNING: outputs diverge from PyTorch. Do not ship this until")
        print("  the cause is understood — the browser would play a different")
        print("  game from the one that was evaluated.")

    print(f"\n  Elo conditioning: {ELO_BUCKETS} buckets, passed at inference as")
    print("  elos_self — one model covers every student level.")

    if size_mb > 50:
        print(f"\n  {size_mb:.0f} MB is a download every student pays on first")
        print("  load. Plan to quantise to int8 before shipping.")

    print("\nNext: build the held-out eval set before any Kaggle training run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
