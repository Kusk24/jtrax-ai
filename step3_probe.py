"""Step 3 — how bad is the base model, exactly?

`lichess_6layers` (1.3M params) is a character-level GPT over PGN text. It does
not know what a board is; it predicts the next character of a game transcript.
So the first question is not "how strong is it" but "does it produce legal
chess at all".

Two numbers come out, and both are the "before" side of the fine-tune:

  legal-move rate  — of the moves it proposes, how many are legal
  illegal-first    — how often its FIRST guess is illegal (retries hidden)

A model that scores well here has learned real structure. A model near zero has
learned to imitate notation without tracking the position.

Run:  conda activate maia2 && python step3_probe.py
"""

import json
import pathlib
import pickle
import sys

import chess
import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE / "vendor"))

CKPT = HERE / "hf_models" / "lichess_6layers_ckpt_no_optimizer.pt"
META = HERE / "vendor" / "meta.pkl"
RESULTS = HERE / "results"

GAMES = 20
MAX_MOVES = 80  # plies; long enough to leave book, short enough to stay quick
MAX_RETRIES = 5  # resample an illegal move this many times before giving up
TEMPERATURE = 0.5  # low: we want its best guess, not creative writing


def load_model():
    from nanogpt_model import GPT, GPTConfig

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))

    # nanoGPT checkpoints saved under torch.compile carry an "_orig_mod."
    # prefix that GPT itself does not have.
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def next_move_san(model, stoi, itos, prompt, temperature):
    """Sample characters until the move ends (space), return the SAN string."""
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)
    out = []
    for _ in range(8):  # no legal SAN move is longer than this
        # block_size is 1023; keep the tail if the game runs long.
        window = idx[:, -1023:]
        with torch.no_grad():
            logits, _ = model(window)
        logits = logits[:, -1, :] / temperature
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        ch = itos[nxt.item()]
        if ch == " ":
            break
        out.append(ch)
        idx = torch.cat((idx, nxt), dim=1)
    return "".join(out)


def play_game(model, stoi, itos):
    """Let the model play both sides. Returns per-game counts."""
    board = chess.Board()
    # Karvonen's models expect a leading ';' — it is the game delimiter token,
    # and output degrades without it.
    pgn = ";1."
    proposed = legal = first_try_legal = 0

    for ply in range(MAX_MOVES):
        if board.is_game_over():
            break
        move = None
        for attempt in range(MAX_RETRIES):
            san = next_move_san(model, stoi, itos, pgn, TEMPERATURE)
            proposed += 1
            if not san:
                continue
            try:
                move = board.parse_san(san)
            except (chess.IllegalMoveError, chess.InvalidMoveError,
                    chess.AmbiguousMoveError):
                continue
            legal += 1
            if attempt == 0:
                first_try_legal += 1
            break
        if move is None:
            break  # model could not find a legal move; game ends here

        board.push(move)
        pgn += san + " "
        if board.turn == chess.WHITE:
            pgn += f"{board.fullmove_number}."

    return {
        "plies": board.ply(),
        "proposed": proposed,
        "legal": legal,
        "first_try_legal": first_try_legal,
        "pgn": pgn,
    }


def main() -> int:
    if not CKPT.exists():
        print(f"Checkpoint missing: {CKPT}\n"
              "Fetch it with:\n"
              "  python -c \"from huggingface_hub import hf_hub_download as d; "
              "d('adamkarvonen/chess_llms',"
              "'lichess_6layers_ckpt_no_optimizer.pt',local_dir='hf_models')\"")
        return 1

    meta = pickle.loads(META.read_bytes())
    stoi, itos = meta["stoi"], meta["itos"]

    model, ckpt = load_model()
    n = sum(p.numel() for p in model.parameters())
    print(f"lichess_6layers · {n / 1e6:.2f}M params · "
          f"trained to iter {ckpt['iter_num']:,}")
    print(f"Playing {GAMES} games against itself, "
          f"max {MAX_MOVES} plies, temperature {TEMPERATURE}\n")

    totals = {"plies": 0, "proposed": 0, "legal": 0, "first_try_legal": 0}
    longest = None
    for g in range(GAMES):
        r = play_game(model, stoi, itos)
        for k in totals:
            totals[k] += r[k]
        if longest is None or r["plies"] > longest["plies"]:
            longest = r
        print(f"  game {g + 1:2d}: {r['plies']:3d} plies · "
              f"{r['legal']}/{r['proposed']} legal")

    legal_rate = totals["legal"] / max(totals["proposed"], 1)
    first_rate = totals["first_try_legal"] / max(totals["plies"], 1)
    avg_plies = totals["plies"] / GAMES

    print(f"\n{'=' * 54}")
    print("  BEFORE fine-tuning")
    print(f"{'=' * 54}")
    print(f"  legal-move rate      {legal_rate:.3f}   "
          f"({totals['legal']}/{totals['proposed']} proposals)")
    print(f"  legal on first try   {first_rate:.3f}")
    print(f"  average game length  {avg_plies:.1f} plies")
    print(f"\n  longest game:\n  {longest['pgn'][:300]}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "probe_before.json"
    out.write_text(json.dumps({
        "model": "lichess_6layers (1.3M, pre-finetune)",
        "games": GAMES,
        "temperature": TEMPERATURE,
        "legal_move_rate": round(legal_rate, 4),
        "legal_first_try_rate": round(first_rate, 4),
        "avg_plies": round(avg_plies, 1),
        "totals": totals,
    }, indent=2) + "\n")
    print(f"\n  written to {out.relative_to(HERE)}")
    print("\nThis is the floor. step5 trains, step6 re-runs this to compare.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
