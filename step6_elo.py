"""Step 6 — measure playing strength in Elo, by playing Stockfish.

Legal-move rate says whether a model follows the rules. It says nothing about
whether the moves are any good. This plays real games against Stockfish pinned
to known strengths and derives a rating from the results.

Two ways to weaken Stockfish, and the choice matters:

  UCI_Elo      1320-3190, calibrated by Stockfish itself. Trustworthy, but it
               cannot go below 1320 — too strong to measure a beginner model.
  Skill Level  0-20, reaches much weaker play, but the Elo equivalents are
               community estimates rather than anything Stockfish guarantees.

So ratings at or above 1320 are solid; anything below is labelled approximate,
because it is.

Players supported:
  --player maia2            the published Maia-2 (set --elo for its dial)
  --player ckpt:<path>      any nanoGPT chess checkpoint

Run:  conda activate jtrax-ai && python step6_elo.py --player maia2 --games 20
"""

import argparse
import json
import math
import pathlib
import pickle
import sys

import chess
import chess.engine

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE / "vendor"))
RESULTS = HERE / "results"

STOCKFISH = "/opt/homebrew/bin/stockfish"
MOVE_TIME = 0.05  # seconds per Stockfish move; strength is set by the rungs
MAX_PLIES = 200

# Skill Level rungs are community estimates — Stockfish makes no promise about
# them. UCI_Elo rungs are Stockfish's own calibration.
# Calibrated rungs FIRST. Stockfish calibrates UCI_Elo itself; the Skill Level
# equivalents are community guesses and measurably wrong (Maia-2 lost to
# "Skill 3 (~1000)" while drawing a calibrated 1320). Running Skill first let a
# bad result there abort the ladder before any trustworthy rung was reached,
# which is exactly how Maia-3 got reported as 820.
EXACT_RUNGS = [
    {"uci_elo": 1320, "elo": 1320, "exact": True},
    {"uci_elo": 1600, "elo": 1600, "exact": True},
    {"uci_elo": 1900, "elo": 1900, "exact": True},
    {"uci_elo": 2200, "elo": 2200, "exact": True},
]

# Only reached if the player cannot score against 1320, Stockfish's floor.
SKILL_RUNGS = [
    {"skill": 6, "elo": 1200, "exact": False},
    {"skill": 3, "elo": 1000, "exact": False},
    {"skill": 0, "elo": 800, "exact": False},
]


class Maia2Player:
    """The published Maia-2. Always legal — it masks to legal moves itself."""

    name = "maia2"

    def __init__(self, elo=1500):
        from maia2 import inference, model

        self.model = model.from_pretrained(type="rapid", device="cpu")
        self.prepared = inference.prepare()
        self.inference = inference
        self.elo = elo
        self.name = f"maia2@{elo}"

    def move(self, board, history):
        out = self.inference.inference_each(
            self.model, self.prepared, board.fen(), self.elo, self.elo)
        # inference_each returns (move_probs, win_prob) with moves ranked.
        probs = out[0] if isinstance(out, tuple) else out
        best = max(probs.items(), key=lambda kv: kv[1])[0]
        return chess.Move.from_uci(best)


class UCIEnginePlayer:
    """Any UCI engine, e.g. lc0.

    Node count matters more than anything else here. Leela is a *search*
    engine: its rating is whatever you give it thinking time to reach, so it
    has no single Elo. At `--nodes 1` it plays its raw policy network with no
    search, which is the only setting comparable to Maia-2 or a nanoGPT — both
    of which are searchless by construction.
    """

    def __init__(self, path, nodes=1, name=None):
        # popen_uci takes a list when the engine needs arguments (maia3-uci).
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.limit = chess.engine.Limit(nodes=nodes)
        first = path[0] if isinstance(path, list) else path
        self.name = name or f"{pathlib.Path(first).name}@{nodes}nodes"

    def move(self, board, history):
        return self.engine.play(board, self.limit).move

    def close(self):
        self.engine.quit()


class NanoGPTPlayer:
    """A character-level GPT. Retries on illegal output, like the Play screen
    would have to — a model that proposes an illegal move is not resigning."""

    def __init__(self, path, retries=8, temperature=0.3):
        from step3_probe import load_model

        meta = pickle.loads((HERE / "vendor" / "meta.pkl").read_bytes())
        self.stoi, self.itos = meta["stoi"], meta["itos"]
        self.model, _ = load_model(path)
        self.retries = retries
        self.temperature = temperature
        self.name = pathlib.Path(path).stem
        self.illegal = 0

    def move(self, board, history):
        from step3_probe import next_move_san

        for attempt in range(self.retries):
            san = next_move_san(self.model, self.stoi, self.itos,
                                history, self.temperature)
            if not san:
                continue
            try:
                return board.parse_san(san)
            except (chess.IllegalMoveError, chess.InvalidMoveError,
                    chess.AmbiguousMoveError):
                self.illegal += 1
        return None  # gave up: scored as a loss, which is the honest outcome


def play_game(player, engine, limit, player_is_white):
    """One game. Returns 1.0 win / 0.5 draw / 0.0 loss, from player's view."""
    board = chess.Board()
    history = ";1."

    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        our_turn = (board.turn == chess.WHITE) == player_is_white
        if our_turn:
            mv = player.move(board, history)
            if mv is None or mv not in board.legal_moves:
                return 0.0  # could not produce a legal move
        else:
            mv = engine.play(board, limit).move

        san = board.san(mv)
        board.push(mv)
        history += san + " "
        if board.turn == chess.WHITE:
            history += f"{board.fullmove_number}."

    if board.ply() >= MAX_PLIES:
        return 0.5  # adjudicate an unfinished game as a draw
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == player_is_white else 0.0


def performance_rating(opponent_elo, score, games):
    """Standard performance rating from a score against a known opponent."""
    if games == 0:
        return None
    p = score / games
    p = min(max(p, 0.01), 0.99)  # a clean sweep implies infinity; clamp it
    return opponent_elo - 400 * math.log10(1 / p - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--player", required=True,
                    help="'maia2', 'maia3', 'ckpt:<path to .pt>', or 'lc0'")
    ap.add_argument("--maia3-model", default="maia3-5m",
                    help="maia3-5m / maia3-23m / maia3-79m")
    ap.add_argument("--nodes", type=int, default=1,
                    help="search nodes for lc0; 1 = policy only, no search")
    ap.add_argument("--elo", type=int, default=1500,
                    help="rating dial for maia2")
    ap.add_argument("--games", type=int, default=10,
                    help="games per rung; more is slower but tighter")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not pathlib.Path(STOCKFISH).exists():
        print(f"Stockfish not found at {STOCKFISH}. Run: brew install stockfish")
        return 1

    if args.player == "maia2":
        player = Maia2Player(args.elo)
    elif args.player.startswith("ckpt:"):
        player = NanoGPTPlayer(args.player.split(":", 1)[1])
    elif args.player == "lc0":
        player = UCIEnginePlayer("/opt/homebrew/bin/lc0", args.nodes)
    elif args.player == "maia3":
        # Rating-conditioned and searchless, so nodes=1 is the only honest
        # setting — the same footing as Maia-2 and the nanoGPT.
        player = UCIEnginePlayer(
            ["maia3-uci", "--model", args.maia3_model,
             "--use-uci-history", "--elo", str(args.elo)],
            nodes=1, name=f"{args.maia3_model}@{args.elo}")
    else:
        print("--player must be 'maia2', 'ckpt:<path>', or 'lc0'")
        return 1

    print(f"player: {player.name}")
    print(f"{args.games} games per rung, alternating colours\n")

    limit = chess.engine.Limit(time=MOVE_TIME)

    def run_rung(rung):
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
        if "uci_elo" in rung:
            engine.configure({"UCI_LimitStrength": True,
                              "UCI_Elo": rung["uci_elo"]})
            label = f"UCI_Elo {rung['uci_elo']}"
        else:
            engine.configure({"Skill Level": rung["skill"]})
            label = f"Skill {rung['skill']} (~{rung['elo']})"

        score = 0.0
        for g in range(args.games):
            score += play_game(player, engine, limit,
                               player_is_white=(g % 2 == 0))
        engine.quit()

        perf = performance_rating(rung["elo"], score, args.games)
        row = {"opponent": label, "opponent_elo": rung["elo"],
               "exact": rung["exact"], "score": score,
               "games": args.games, "performance": round(perf)}
        print(f"  vs {label:22s} {score:5.1f}/{args.games}  "
              f"-> performance {perf:.0f}"
              + ("" if rung["exact"] else "  (approx)"), flush=True)
        return row

    rows = []
    for rung in EXACT_RUNGS:
        rows.append(run_rung(rung))
        # Climbing further is pointless once it is being beaten comfortably.
        if rows[-1]["score"] / args.games < 0.05:
            break

    # Below Stockfish's 1320 floor there is nothing calibrated to play, so drop
    # to Skill Level and say so loudly in the summary.
    if rows[0]["score"] / args.games < 0.05:
        print("  (scored under 5% at Stockfish's 1320 floor — "
              "dropping to Skill Level, which is only approximate)")
        for rung in SKILL_RUNGS:
            rows.append(run_rung(rung))
            if rows[-1]["score"] / args.games >= 0.05:
                break

    # Only competitive rungs carry information: a 0/12 or 12/12 tells you the
    # opponent was out of range, not what the rating is.
    useful = [r for r in rows if 0.05 <= r["score"] / r["games"] <= 0.95]

    # Calibrated rungs win outright. Measured 2026-08-29: Maia-2 lost 1.5/12 to
    # "Skill 3 (~1000)" while drawing 6.5/12 against a calibrated UCI_Elo 1320.
    # Both cannot be true, so the Skill Level Elo equivalents are wrong, and
    # averaging them in dragged a ~1450 player down to 1027. Skill Level is only
    # used when no calibrated rung was competitive, and it is flagged loudly.
    exact_useful = [r for r in useful if r["exact"]]
    pool = exact_useful or useful or rows
    estimate = sum(r["performance"] for r in pool) / len(pool)
    exact_only = [r for r in pool if r["exact"]]

    print(f"\n{'=' * 54}")
    print(f"  ESTIMATED ELO: {estimate:.0f}")
    print(f"{'=' * 54}")
    if not exact_only:
        print("  Based only on Skill Level rungs, whose Elo equivalents are")
        print("  community estimates. Treat as a ballpark, not a rating.")
    if isinstance(player, NanoGPTPlayer):
        print(f"  illegal moves proposed and retried: {player.illegal}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / (args.out or f"elo_{player.name.replace('@', '_')}.json")
    out.write_text(json.dumps({"player": player.name,
                               "games_per_rung": args.games,
                               "estimated_elo": round(estimate),
                               "rungs": rows}, indent=2) + "\n")
    print(f"\n  written to results/{out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
