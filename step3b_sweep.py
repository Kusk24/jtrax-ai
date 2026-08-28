"""Step 3b — find a starting point that is bad enough to be worth improving.

`lichess_6layers` scores 0.848 legal already, which leaves little visible room.
Karvonen also published 31 checkpoints saved *during* a training run
(adamkarvonen/chess_llm_30_checkpoints, iter 0 → 600k). Somewhere early in that
run is a model that half-knows chess.

This sweeps the early ones and reports legal-move rate for each, so the base
model is chosen from measurements rather than a guess. Pick the checkpoint
closest to the floor you want, then fine-tune that.

Run:  conda activate jtrax-ai && python step3b_sweep.py
"""

import json
import pathlib
import sys

from huggingface_hub import hf_hub_download

from step3_probe import RESULTS, probe

REPO = "adamkarvonen/chess_llm_30_checkpoints"
# Early run only — by 100k the curve has usually flattened. Cheap to extend.
ITERS = [0, 20000, 40000, 60000, 80000, 100000]
GAMES = 8  # fewer than step3: this is a scouting pass, not the final number


def main() -> int:
    rows = []
    for it in ITERS:
        name = f"ckpt_iter_{it}.pt"
        print(f"\n=== {name} ===")
        try:
            path = hf_hub_download(REPO, name, local_dir="hf_ckpts")
        except Exception as exc:
            print(f"  download failed: {exc}")
            continue
        try:
            r = probe(pathlib.Path(path), GAMES, quiet=True)
        except Exception as exc:
            # A near-random model can emit sequences the loader chokes on;
            # that is itself a result, not a crash worth stopping for.
            print(f"  probe failed: {type(exc).__name__}: {exc}")
            continue
        rows.append(r)
        print(f"  legal {r['legal_move_rate']:.3f} · "
              f"first-try {r['legal_first_try_rate']:.3f} · "
              f"{r['avg_plies']} plies avg")

    if not rows:
        print("\nNo checkpoints probed successfully.")
        return 1

    print(f"\n{'=' * 62}")
    print(f"  {'checkpoint':<24}{'legal':>8}{'first-try':>12}{'plies':>9}")
    print(f"{'=' * 62}")
    for r in rows:
        print(f"  {r['checkpoint']:<24}{r['legal_move_rate']:>8.3f}"
              f"{r['legal_first_try_rate']:>12.3f}{r['avg_plies']:>9.1f}")
    print(f"  {'lichess_6layers (final)':<24}{0.848:>8.3f}{0.919:>12.3f}"
          f"{56.5:>9.1f}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "sweep_checkpoints.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\n  written to results/{out.name}")
    print("\nPick the row nearest the floor you want, then fine-tune that "
          "checkpoint in step5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
