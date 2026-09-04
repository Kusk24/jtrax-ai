"""Step 9 — export the novice model to ONNX and quantise it for the browser.

`step2_export_onnx.py` proved the Maia-2 path. This is the other half: the
nanoGPT, which needs different handling on three counts.

  * `GPT.forward(idx, targets=None)` returns `(logits, loss)` and `loss` is
    `None` at inference. ONNX has no None output, so a thin wrapper returns
    just the logits.
  * It returns logits for the **last position only** — nanoGPT's own inference
    optimisation, and exactly what character sampling needs.
  * Sequence length is dynamic. A game grows one character at a time up to
    block_size 1023, so the graph must accept any length.

Quantisation is not free here, and picking the right gate took two goes.
Top-1 character agreement against fp32 looked like the obvious test, but it is
too strict: the model samples at temperature 0.5 in step3_probe, so a 1% argmax
flip is smaller than its own sampling noise. Agreement is kept as a diagnostic,
and the **decision** is on self-play legality — the same measurement
`step3_probe.py` reports for the PyTorch checkpoint, so the two are comparable.

Run:  conda activate jtrax-ai && python step9_export_novice_onnx.py --help
"""

import argparse
import json
import pathlib
import pickle
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE / "vendor"))
RESULTS = HERE / "results"

BLOCK = 1023  # the checkpoint's block_size; the graph must not exceed it

# Self-play legality is noisy across a couple of dozen games, so the gate is
# "not meaningfully worse", not "identical".
LEGALITY_TOLERANCE = 0.015


def build_wrapper(torch):
    class _Wrapper(torch.nn.Module):
        """One input, one output — the shape ONNX can express.

        Returns logits for the final position as (batch, vocab), dropping
        nanoGPT's time dimension of 1 so the browser gets a plain vector.
        """

        def __init__(self, gpt):
            super().__init__()
            self.gpt = gpt

        def forward(self, idx):
            logits, _ = self.gpt(idx)
            return logits[:, -1, :]

    return _Wrapper


def sample_prefixes(path, stoi, count, stride=97):
    """Real PGN prefixes to test against, not zeros.

    Zeros would exercise a code path the model never sees. `stride` is prime so
    the cut points do not land on the same part of every game.
    """
    prefixes = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if len(line) < 40:
                continue
            ids = [stoi[c] for c in line if c in stoi]
            for cut in range(20, len(ids), stride):
                prefixes.append(ids[max(0, cut - BLOCK):cut])
                if len(prefixes) >= count:
                    return prefixes
    return prefixes


class OnnxAsTorchModel:
    """Quacks like the PyTorch GPT so `step3_probe`'s sampler drives the
    exported graph unchanged.

    The sampler does `logits, _ = model(window)` then slices `[:, -1, :]`. The
    exported graph already returns only the last position, so the time axis is
    added back here rather than forking the sampler — a second copy of the
    sampling loop is exactly how the served model and the measured model drift
    apart without anyone noticing.
    """

    def __init__(self, path, torch, ort):
        self.sess = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"])
        self.torch = torch

    def __call__(self, idx):
        out = self.sess.run(None, {"tokens": idx.numpy()})[0]
        return self.torch.from_numpy(out).unsqueeze(1), None


def probe_onnx(path, games, torch, ort):
    """Self-play legality for an exported graph, comparable to step3_probe."""
    from step3_probe import play_game

    meta = pickle.loads((HERE / "vendor" / "meta.pkl").read_bytes())
    model = OnnxAsTorchModel(path, torch, ort)
    totals = {"plies": 0, "proposed": 0, "legal": 0, "first_try_legal": 0}
    for g in range(games):
        r = play_game(model, meta["stoi"], meta["itos"])
        for k in totals:
            totals[k] += r[k]
        print(f"    game {g + 1:2d}: {r['plies']:3d} plies · "
              f"{r['legal']}/{r['proposed']} legal", flush=True)
    return {
        "legal_move_rate": round(totals["legal"] / max(totals["proposed"], 1), 4),
        "legal_first_try_rate": round(
            totals["first_try_legal"] / max(totals["plies"], 1), 4),
        "avg_plies": round(totals["plies"] / max(games, 1), 1),
    }


def top1_agreement(session, model, torch, prefixes, np):
    """Fraction of positions where ONNX and PyTorch pick the same next char.

    A diagnostic, not the gate — see the module docstring. Useful because a
    sudden drop here localises a broken export faster than a game does.
    """
    agree = 0
    worst = 0.0
    for ids in prefixes:
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            ref = model(x).numpy()
        got = session.run(None, {"tokens": x.numpy()})[0]
        worst = max(worst, float(np.abs(got - ref).max()))
        agree += int(got.argmax() == ref.argmax())
    return agree / max(len(prefixes), 1), worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(HERE / "runs" / "ckpt_40000.pt"))
    ap.add_argument("--heldout", default=str(HERE / "data" / "novice_heldout.txt"))
    ap.add_argument("--positions", type=int, default=400,
                    help="held-out positions used to check agreement")
    ap.add_argument("--name", default="novice",
                    help="output basename: <name>.onnx and <name>_int8.onnx")
    ap.add_argument("--probe-games", type=int, default=20,
                    help="self-play games for the int8 legality check; 0 skips")
    ap.add_argument("--reference-legal", type=float, default=None,
                    help="PyTorch legal-move rate to compare against; read "
                         "from results/probe_novice_*.json when omitted")
    args = ap.parse_args()

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        print(f"Missing dependency ({exc}). Run:\n"
              "  conda activate jtrax-ai && pip install onnx onnxruntime")
        return 1

    from nanogpt_model import GPT, GPTConfig

    ckpt_path = pathlib.Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}")
        return 1

    print(f"loading {ckpt_path.name} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gpt = GPT(GPTConfig(**ckpt["model_args"]))
    # Checkpoints saved under torch.compile carry an "_orig_mod." prefix.
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    gpt.load_state_dict(state)
    gpt.eval()

    n_params = sum(p.numel() for p in gpt.parameters())
    print(f"  {n_params / 1e6:.2f}M parameters · iter {ckpt.get('iter_num')}")

    model = build_wrapper(torch)(gpt)
    model.eval()

    meta = pickle.loads((HERE / "vendor" / "meta.pkl").read_bytes())
    stoi = meta["stoi"]

    print(f"sampling held-out positions from "
          f"{pathlib.Path(args.heldout).name} ...")
    prefixes = sample_prefixes(args.heldout, stoi, args.positions)
    if not prefixes:
        print(f"  no usable prefixes in {args.heldout}")
        return 1
    lengths = [len(p) for p in prefixes]
    print(f"  {len(prefixes)} positions · lengths {min(lengths)}-{max(lengths)}")

    RESULTS.mkdir(exist_ok=True)
    fp32 = RESULTS / f"{args.name}.onnx"
    int8 = RESULTS / f"{args.name}_int8.onnx"

    # A mid-length example so the trace sees a realistic shape. dynamic_axes
    # below is what actually makes any length valid.
    example = torch.tensor([prefixes[len(prefixes) // 2]], dtype=torch.long)

    print(f"\nexporting {fp32.name} ...")
    with torch.no_grad():
        torch.onnx.export(
            model, (example,), str(fp32),
            opset_version=17,
            input_names=["tokens"],
            output_names=["logits"],
            dynamic_axes={"tokens": {0: "batch", 1: "sequence"},
                          "logits": {0: "batch"}},
        )
    onnx.checker.check_model(str(fp32))

    sess = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    agree_fp32, drift_fp32 = top1_agreement(sess, model, torch, prefixes, np)
    print(f"  agreement {agree_fp32:.4f} · max drift {drift_fp32:.2e}")

    print(f"\nquantising to {int8.name} (int8 weights) ...")
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    sess8 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    agree_int8, drift_int8 = top1_agreement(sess8, model, torch, prefixes, np)

    mb32 = fp32.stat().st_size / 1_000_000
    mb8 = int8.stat().st_size / 1_000_000

    print(f"\n{'=' * 54}")
    print(f"  {ckpt_path.name} -> ONNX")
    print(f"{'=' * 54}")
    print(f"  fp32   {mb32:6.1f} MB · top-1 agreement {agree_fp32:.4f}"
          f" · drift {drift_fp32:.2e}")
    print(f"  int8   {mb8:6.1f} MB · top-1 agreement {agree_int8:.4f}"
          f" · drift {drift_int8:.2e}")
    print(f"  saving {(1 - mb8 / mb32) * 100:.0f}% of the download")

    # The decision. Agreement above is a diagnostic; what matters is whether
    # the quantised graph still follows the rules as well as the checkpoint.
    reference = args.reference_legal
    if reference is None:
        prior = RESULTS / f"probe_novice_{ckpt.get('iter_num')}.json"
        if prior.exists():
            reference = json.loads(prior.read_text())["legal_move_rate"]

    played = None
    ok = None
    if args.probe_games:
        print(f"\nself-play with the int8 graph, {args.probe_games} games ...")
        played = probe_onnx(int8, args.probe_games, torch, ort)
        print(f"\n  int8 legal-move rate   {played['legal_move_rate']:.4f}"
              f" · first try {played['legal_first_try_rate']:.4f}"
              f" · {played['avg_plies']} plies")
        if reference is not None:
            drop = reference - played["legal_move_rate"]
            ok = drop <= LEGALITY_TOLERANCE
            print(f"  PyTorch reference      {reference:.4f}"
                  f"   ({-drop:+.4f})")
            if ok:
                print(f"\n  Quantisation costs nothing it can measure. Ship the "
                      f"{mb8:.0f} MB file.")
            else:
                print(f"\n  int8 drops legality by {drop:.4f}, past the "
                      f"{LEGALITY_TOLERANCE} tolerance. Serve fp32 "
                      f"({mb32:.0f} MB) until that is understood.")
        else:
            print("\n  No PyTorch reference found — run step3_probe.py on the "
                  "same checkpoint to make this comparable.")

    out = RESULTS / f"onnx_{args.name}.json"
    out.write_text(json.dumps({
        "checkpoint": ckpt_path.name,
        "iter_num": ckpt.get("iter_num"),
        "positions": len(prefixes),
        "fp32": {"mb": round(mb32, 1), "agreement": round(agree_fp32, 4),
                 "max_drift": drift_fp32},
        "int8": {"mb": round(mb8, 1), "agreement": round(agree_int8, 4),
                 "max_drift": drift_int8, "self_play": played},
        "pytorch_legal_move_rate": reference,
        "int8_shippable": ok,
    }, indent=2) + "\n")
    print(f"\n  written to results/{out.name}")

    print("\nNext: the browser feeds the game so far and samples characters")
    print("until a space. There is no KV cache in this graph, so each character")
    print("re-runs the whole sequence — measure it in the Play screen before")
    print("assuming it is fast enough.")
    return 0 if ok is not False else 5


if __name__ == "__main__":
    sys.exit(main())
