"""Step 10 — export the fine-tuned Maia-2 to ONNX and quantise it.

`step2_export_onnx.py` exported the *stock* model as a feasibility gate, before
any training existed. This exports the checkpoint we actually ship —
`runs_maia/maia2_ft_20000.pt` — and quantises it for the browser.

Same lesson as step 9: numeric drift does not tell you whether a quantised model
still plays the same chess. The gate here is **held-out top-1 move-match**.

Unlike the nanoGPT, this model does **not** survive int8. Measured on the same
8,192 held-out positions (2026-09-05):

    fp32     93.2 MB   0.5387
    float16  46.7 MB   0.5397    lossless
    uint8    23.5 MB   0.5299    -0.0088
    int8     23.5 MB   0.5168    -0.0219

The fine-tune itself is worth +0.0167, so int8 would undo it and then some, and
uint8 costs half of it. float16 is the one to ship. Every candidate is scored
against **fp32 on the identical positions** — an earlier version compared 8k
positions against a 50k reference and made int8 look twice as good as it is.

Signature is Maia-2's, read from the installed package rather than guessed:

    forward(boards, elos_self, elos_oppo) -> (logits_maia, side_info, value)

`elos_self` is the rating dial — one exported file covers every student level,
which is why a fourth difficulty tier would cost a dropdown entry and no
second model.

Run:  conda activate jtrax-ai && python step10_export_strong_onnx.py --help
"""

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"

INPUT_CHANNELS = 18  # maia2/configs/maia2-training.yaml
MOVEMATCH_TOLERANCE = 0.005  # below this the quantised model plays differently


class OnnxMaiaModel:
    """Quacks like MAIA2Model so `step8_maia_eval.move_match` drives the
    exported graph unchanged.

    Reusing the scorer matters more than the few lines it saves: a second copy
    of the evaluation path is how the served model and the measured model end
    up being different models.
    """

    def __init__(self, path, torch, ort):
        self.sess = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"])
        self.torch = torch

    def eval(self):
        return self

    def __call__(self, boards, elos_self, elos_oppo):
        outs = self.sess.run(None, {
            "boards": boards.numpy(),
            "elos_self": elos_self.numpy(),
            "elos_oppo": elos_oppo.numpy(),
        })
        return tuple(self.torch.from_numpy(o) for o in outs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(HERE / "runs_maia" / "maia2_ft_20000.pt"),
                    help="step7 fine-tune checkpoint; omit --ckpt to export stock")
    ap.add_argument("--heldout", default=str(HERE / "data" / "strong_heldout.txt"))
    ap.add_argument("--band-elo", type=int, default=2200)
    ap.add_argument("--positions", type=int, default=20_000,
                    help="held-out positions for the move-match gate")
    ap.add_argument("--name", default="strong")
    args = ap.parse_args()

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from maia2 import model as maia_model
        from onnxconverter_common import float16
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        print(f"Missing dependency ({exc}). Run:\n"
              "  conda activate jtrax-ai && pip install maia2 onnx onnxruntime "
              "onnxconverter-common")
        return 1

    from step8_maia_eval import load_checkpoint_into, move_match

    print("loading released Maia-2 on CPU ...")
    net = maia_model.from_pretrained(type="rapid", device="cpu")

    step = None
    ckpt_path = pathlib.Path(args.ckpt)
    if ckpt_path.exists():
        step = load_checkpoint_into(net, ckpt_path)
        print(f"  loaded fine-tune {ckpt_path.name} (step {step})")
    else:
        print(f"  no checkpoint at {ckpt_path} — exporting stock weights")
    net.eval()
    print(f"  {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters")

    boards = torch.zeros(1, INPUT_CHANNELS, 8, 8, dtype=torch.float32)
    elos_self = torch.zeros(1, dtype=torch.long)
    elos_oppo = torch.zeros(1, dtype=torch.long)

    RESULTS.mkdir(exist_ok=True)
    fp32 = RESULTS / f"{args.name}.onnx"
    fp16 = RESULTS / f"{args.name}_fp16.onnx"
    uint8 = RESULTS / f"{args.name}_uint8.onnx"

    print(f"\nexporting {fp32.name} ...")
    with torch.no_grad():
        torch.onnx.export(
            net, (boards, elos_self, elos_oppo), str(fp32),
            opset_version=17,
            input_names=["boards", "elos_self", "elos_oppo"],
            output_names=["logits_maia", "logits_side_info", "logits_value"],
            dynamic_axes={"boards": {0: "batch"},
                          "elos_self": {0: "batch"},
                          "elos_oppo": {0: "batch"},
                          "logits_maia": {0: "batch"}},
        )
    onnx.checker.check_model(str(fp32))

    sess = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = net(boards, elos_self, elos_oppo)
    got = sess.run(None, {"boards": boards.numpy(),
                          "elos_self": elos_self.numpy(),
                          "elos_oppo": elos_oppo.numpy()})
    drift = float(np.abs(got[0] - ref[0].numpy()).max())
    print(f"  policy shape {got[0].shape} · max drift {drift:.2e}")

    print(f"\nbuilding {fp16.name} (float16) ...")
    onnx.save(float16.convert_float_to_float16(
        onnx.load(str(fp32)), keep_io_types=True), str(fp16))

    print(f"building {uint8.name} (uint8 weights) ...")
    # uint8, not int8: measured -0.0088 against int8's -0.0219 on this model.
    quantize_dynamic(str(fp32), str(uint8), weight_type=QuantType.QUInt8)

    # Every candidate is scored on the SAME positions as fp32. Comparing a
    # small sample against a large stored reference conflates sampling noise
    # with quantisation loss, and here it understated the loss by half.
    print(f"\nscoring each on the same {args.positions:,} held-out positions ...")
    scores = {}
    for label, path in [("fp32", fp32), ("float16", fp16), ("uint8", uint8)]:
        got, seen = move_match(OnnxMaiaModel(path, torch, ort), args.heldout,
                               args.band_elo, args.positions, "cpu")
        scores[label] = {"mb": round(path.stat().st_size / 1_000_000, 1),
                         "move_match": round(got, 4)}
        print(f"  {label:8s} {scores[label]['mb']:6.1f} MB · {got:.4f}",
              flush=True)

    base = scores["fp32"]["move_match"]
    print(f"\n{'=' * 54}")
    print(f"  {ckpt_path.name} -> ONNX  ({seen:,} positions)")
    print(f"{'=' * 54}")
    for label, s in scores.items():
        delta = "" if label == "fp32" else f"  ({s['move_match'] - base:+.4f})"
        print(f"  {label:8s} {s['mb']:6.1f} MB · move-match "
              f"{s['move_match']:.4f}{delta}")

    # Smallest file whose loss is inside tolerance, preferring smaller.
    shippable = [l for l in ("uint8", "float16", "fp32")
                 if base - scores[l]["move_match"] <= MOVEMATCH_TOLERANCE]
    choice = shippable[0] if shippable else "fp32"
    print(f"\n  Ship **{choice}** ({scores[choice]['mb']:.0f} MB).")
    if choice != "uint8":
        lost = base - scores["uint8"]["move_match"]
        print(f"  uint8 would save {scores[choice]['mb'] - scores['uint8']['mb']:.0f} MB "
              f"but costs {lost:.4f} move-match — the fine-tune itself is only")
        print("  worth +0.0167, so 8-bit gives back most of what it bought.")

    out = RESULTS / f"onnx_{args.name}.json"
    out.write_text(json.dumps({
        "checkpoint": ckpt_path.name if ckpt_path.exists() else "stock",
        "finetune_step": step,
        "positions": seen,
        "max_drift_fp32": drift,
        "scores": scores,
        "ship": choice,
    }, indent=2) + "\n")
    print(f"\n  written to results/{out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
