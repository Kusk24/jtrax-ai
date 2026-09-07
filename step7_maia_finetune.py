"""Step 7 — fine-tune Maia-2 on strong human games.

Maia-2 is the only Maia that can be fine-tuned: Maia-1 needs the dead lc0 /
TensorFlow toolchain, and Maia-3 ships inference code only. But maia2's own
`train.run()` will not take the released checkpoint — it is a *resume*
mechanism that validates `training_metadata`, `checkpoint_year/month` and a
source sha256, none of which the public `.pt` carries. So this is a plain
PyTorch loop that loads the released weights and trains them.

Data comes from the same PGN text file the GPT uses, not from a 30 GB monthly
archive. maia2's `preprocessing()` takes a FEN, so any source of positions
works — replaying `data/strong_train.txt` with python-chess gives exactly the
(board tensor, elo, legal mask, played move) tuples the model needs.

Two details that will silently corrupt training if missed:

  * `preprocessing()` MIRRORS the board when it is Black to move, so the target
    move has to be mirrored too, or Black's moves teach the model nonsense.
  * Loss is masked to legal moves. Maia-2 masks at inference; training without
    the mask spends capacity learning that illegal moves are unlikely, which it
    is never asked at inference.

Elo conditioning: `step4_data.py` filtered the corpus to a band but did not
record per-game ratings, so every position is labelled with the band midpoint.
For 2000-2800 that is bucket 10 (">=2000"), which is what we want the model to
associate these moves with.

Run:  conda activate jtrax-ai && python step7_maia_finetune.py --help
"""

import argparse
import json
import pathlib
import sys
import time

import chess
import torch

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"
ON_KAGGLE = pathlib.Path("/kaggle/working").exists()
OUT_DIR = pathlib.Path("/kaggle/working") if ON_KAGGLE else HERE / "runs_maia"


def positions_from_corpus(path, band_elo, limit_games=None):
    """Replay PGN-text games, yielding one training example per move."""
    from maia2.utils import get_all_possible_moves, mirror_move

    all_moves_dict = {m: i for i, m in enumerate(get_all_possible_moves())}

    with open(path) as fh:
        for n, line in enumerate(fh):
            if limit_games and n >= limit_games:
                return
            line = line.strip().lstrip(";")
            if not line:
                continue
            board = chess.Board()
            for token in line.split():
                if token.endswith(".") or "." in token and token[0].isdigit():
                    # Move numbers are glued to the move: "1.e4" -> "e4".
                    token = token.split(".", 1)[1]
                    if not token:
                        continue
                try:
                    move = board.parse_san(token)
                except (chess.IllegalMoveError, chess.InvalidMoveError,
                        chess.AmbiguousMoveError):
                    break  # corrupt game; drop the rest of it

                fen = board.fen()
                uci = move.uci()
                # Black positions are mirrored by preprocessing(), so the label
                # must be mirrored to match.
                if board.turn == chess.BLACK:
                    uci = mirror_move(uci)
                idx = all_moves_dict.get(uci)
                if idx is not None:
                    yield fen, band_elo, band_elo, idx
                board.push(move)


class CorpusDataset(torch.utils.data.IterableDataset):
    def __init__(self, path, band_elo, limit_games=None):
        self.path = path
        self.band_elo = band_elo
        self.limit_games = limit_games

    def __iter__(self):
        from maia2.inference import preprocessing
        from maia2.utils import create_elo_dict, get_all_possible_moves

        elo_dict = create_elo_dict()
        all_moves_dict = {m: i for i, m in enumerate(get_all_possible_moves())}

        for fen, es, eo, target in positions_from_corpus(
                self.path, self.band_elo, self.limit_games):
            try:
                board_input, elo_self, elo_oppo, legal = preprocessing(
                    fen, es, eo, elo_dict, all_moves_dict)
            except ValueError:
                continue  # position with no legal moves
            yield (board_input, torch.tensor(elo_self),
                   torch.tensor(elo_oppo), legal, torch.tensor(target))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(HERE / "data" / "strong_train.txt"))
    ap.add_argument("--band-elo", type=int, default=2200,
                    help="rating label for every position in the corpus")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="low by default: this is a fine-tune, not a fresh run")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    from maia2 import model as maia_model

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} · output: {out_dir}")

    print("loading released Maia-2 weights...")
    model = maia_model.from_pretrained(type="rapid", device=device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    ds = CorpusDataset(args.data, args.band_elo)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         num_workers=0)

    print(f"corpus: {args.data}")
    print(f"every position labelled Elo {args.band_elo} "
          f"(bucket for '>=2000' is 10)")
    print(f"{args.steps:,} steps · batch {args.batch_size} · lr {args.lr}\n")

    history = []
    t0 = time.time()
    step = 0
    running = correct = seen = 0.0

    while step < args.steps:
        for board_input, elo_self, elo_oppo, legal, target in loader:
            if step >= args.steps:
                break
            board_input = board_input.to(device)
            elo_self = elo_self.to(device)
            elo_oppo = elo_oppo.to(device)
            legal = legal.to(device)
            target = target.to(device)

            with torch.autocast(device_type=device, enabled=use_amp,
                                dtype=torch.float16):
                logits, _, _ = model(board_input, elo_self, elo_oppo)
                # Mask illegal moves to -inf: at inference Maia-2 only ever
                # ranks legal moves, so training should score the same way.
                logits = logits.masked_fill(legal == 0, float("-inf"))
                loss = torch.nn.functional.cross_entropy(logits, target)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            correct += (logits.argmax(dim=-1) == target).sum().item()
            seen += target.numel()
            step += 1

            if step % args.log_every == 0:
                acc = correct / max(seen, 1)
                mins = (time.time() - t0) / 60
                eta = mins / step * (args.steps - step)
                print(f"  step {step:6,} · loss {running / args.log_every:.4f}"
                      f" · move-match {acc:.4f} · {mins:.1f}m · ~{eta:.0f}m left",
                      flush=True)
                history.append({"step": step,
                                "loss": running / args.log_every,
                                "move_match": round(acc, 4)})
                running = correct = seen = 0.0

            if step % args.ckpt_every == 0:
                path = out_dir / f"maia2_ft_{step}.pt"
                torch.save({"model_state_dict": model.state_dict(),
                            "step": step, "band_elo": args.band_elo,
                            "data": args.data}, path)
                (out_dir / "history.json").write_text(json.dumps(history, indent=2))
                print(f"    saved {path.name}", flush=True)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes.")
    print("\nNext: measure it.")
    print("  python step6_elo.py --player maia2 --games 40   # baseline ~1300-1450")
    print("  (point --player at the fine-tuned checkpoint to compare)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
