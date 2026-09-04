"""Step 8 — move-match on held-out games, stock Maia-2 vs the fine-tune.

`step7_maia_finetune.py` reported move-match on *training* batches, which shows
the model learned the data it saw — not that it generalises. This scores both
checkpoints on `strong_heldout.txt`, which neither was trained on, so the
comparison means something.

Top-1 move-match: given a position from a real game between 2000+ players, does
the model's highest-probability legal move equal the one the human played? It is
the same metric Maia's papers report (they get 0.5311 on their own mixed-rating
test set, which is a different set — not directly comparable to this number).

Run:  conda activate jtrax-ai && python step8_maia_eval.py
"""

import argparse
import json
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"


def load_checkpoint_into(model, path):
    """Load a step7 fine-tune checkpoint over the released weights."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck.get("model_state_dict", ck)
    model.load_state_dict(state)
    return ck.get("step")


@torch.no_grad()
def move_match(model, corpus, band_elo, max_positions, device, batch_size=256):
    """Fraction of positions where the top legal move is the human's move."""
    from maia2.inference import preprocessing
    from maia2.utils import create_elo_dict, get_all_possible_moves

    from step7_maia_finetune import positions_from_corpus

    elo_dict = create_elo_dict()
    all_moves_dict = {m: i for i, m in enumerate(get_all_possible_moves())}

    model.eval()
    correct = seen = 0
    boards, elos_s, elos_o, legals, targets = [], [], [], [], []

    def flush():
        nonlocal correct, seen, boards, elos_s, elos_o, legals, targets
        if not boards:
            return
        b = torch.stack(boards).to(device)
        es = torch.tensor(elos_s).to(device)
        eo = torch.tensor(elos_o).to(device)
        lg = torch.stack(legals).to(device)
        tg = torch.tensor(targets).to(device)
        logits, _, _ = model(b, es, eo)
        # Rank only legal moves, exactly as inference does.
        logits = logits.masked_fill(lg == 0, float("-inf"))
        correct += (logits.argmax(dim=-1) == tg).sum().item()
        seen += tg.numel()
        boards, elos_s, elos_o, legals, targets = [], [], [], [], []

    for fen, es, eo, target in positions_from_corpus(corpus, band_elo):
        if seen >= max_positions:
            break
        try:
            bi, e_s, e_o, legal = preprocessing(fen, es, eo, elo_dict,
                                                all_moves_dict)
        except ValueError:
            continue
        boards.append(bi)
        elos_s.append(e_s)
        elos_o.append(e_o)
        legals.append(legal)
        targets.append(target)
        if len(boards) >= batch_size:
            flush()
            if seen and seen % 12800 == 0:
                print(f"    {seen:,} positions · {correct / seen:.4f}",
                      flush=True)
    flush()
    return correct / max(seen, 1), seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heldout", default=str(HERE / "data" / "strong_heldout.txt"))
    ap.add_argument("--finetuned", default=str(HERE / "runs_maia" / "maia2_ft_20000.pt"),
                    help="checkpoint from step7; omit to score the stock model only")
    ap.add_argument("--band-elo", type=int, default=2200)
    ap.add_argument("--positions", type=int, default=50_000)
    args = ap.parse_args()

    from maia2 import model as maia_model

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device} · held-out: {pathlib.Path(args.heldout).name}")
    print(f"scoring up to {args.positions:,} positions\n")

    print("stock Maia-2 ...")
    model = maia_model.from_pretrained(type="rapid", device=device)
    stock, n = move_match(model, args.heldout, args.band_elo,
                          args.positions, device)
    print(f"  stock       {stock:.4f}  ({n:,} positions)")

    tuned = step = None
    ft_path = pathlib.Path(args.finetuned)
    if ft_path.exists():
        print(f"\nfine-tuned ({ft_path.name}) ...")
        step = load_checkpoint_into(model, ft_path)
        tuned, _ = move_match(model, args.heldout, args.band_elo,
                              args.positions, device)
        print(f"  fine-tuned  {tuned:.4f}")
    else:
        print(f"\n(no fine-tuned checkpoint at {ft_path} — skipping)")

    print(f"\n{'=' * 54}")
    print("  MOVE-MATCH ON HELD-OUT GAMES (2000-2800)")
    print(f"{'=' * 54}")
    print(f"  stock Maia-2      {stock:.4f}")
    if tuned is not None:
        delta = tuned - stock
        print(f"  fine-tuned        {tuned:.4f}   ({delta:+.4f})")
        if delta <= 0:
            print("\n  No improvement on data it never saw. The training-batch")
            print("  gain did not generalise — Maia-2 already knew this well.")
        elif delta < 0.005:
            print("\n  Improvement is real but under half a point; at this size")
            print("  it is close to measurement noise.")
        else:
            print("\n  Improvement holds on unseen games.")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "maia_heldout_movematch.json"
    out.write_text(json.dumps({
        "heldout": args.heldout,
        "positions": n,
        "band_elo": args.band_elo,
        "stock": round(stock, 4),
        "finetuned": round(tuned, 4) if tuned is not None else None,
        "finetune_step": step,
    }, indent=2) + "\n")
    print(f"\n  written to results/{out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
