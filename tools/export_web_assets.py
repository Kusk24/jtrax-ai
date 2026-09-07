"""Export the lookup tables and test fixtures the web app needs.

Two things have to agree exactly between Python and TypeScript or the served
model plays nonsense while looking healthy:

  * the 1880-entry move vocabulary, whose *order* is the policy head's index
    space. Reimplementing `get_all_possible_moves()` in TS would be a second
    source of truth; exporting it is one.
  * `board_to_tensor` / `preprocessing` — 18 channels of 8x8, plus the mirror
    that Maia-2 applies when Black is to move. This writes fixtures so the TS
    port is tested against the real thing rather than reviewed by eye.

Also exports the novice model's 32-character vocabulary.

Run:  conda activate jtrax-ai && python tools/export_web_assets.py
"""

import argparse
import json
import pathlib
import pickle
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
WEB = HERE.parent / "jtrax-web-app"

# Positions chosen to exercise the parts most likely to be ported wrongly:
# the mirror, castling rights, en passant, and promotion.
FIXTURE_FENS = [
    ("startpos white to move",
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("black to move — triggers the mirror",
     "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
    ("en passant available, white",
     "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"),
    ("no castling rights either side",
     "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1"),
    ("white kingside only",
     "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1"),
    ("promotion available, black to move",
     "8/8/8/8/8/8/1k4p1/6K1 b - - 0 1"),
    ("midgame, black to move",
     "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 6 5"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--web", default=str(WEB), help="jtrax-web-app checkout")
    args = ap.parse_args()

    try:
        import chess
        from maia2.inference import preprocessing
        from maia2.utils import (create_elo_dict, get_all_possible_moves,
                                 mirror_move)
    except ImportError as exc:
        print(f"Missing dependency ({exc}). Run:\n"
              "  conda activate jtrax-ai && pip install maia2")
        return 1

    web = pathlib.Path(args.web)
    if not web.exists():
        print(f"No web app at {web}")
        return 1
    models = web / "public" / "models"
    fixtures = web / "lib" / "engines" / "__fixtures__"
    models.mkdir(parents=True, exist_ok=True)
    fixtures.mkdir(parents=True, exist_ok=True)

    # ---- move vocabulary ---------------------------------------------------
    moves = get_all_possible_moves()
    all_moves_dict = {m: i for i, m in enumerate(moves)}
    (models / "maia-moves.json").write_text(json.dumps(moves))
    print(f"  maia-moves.json          {len(moves)} moves")

    # ---- elo buckets -------------------------------------------------------
    elo_dict = create_elo_dict()
    (models / "maia-elo-buckets.json").write_text(
        json.dumps(list(elo_dict.keys())))
    print(f"  maia-elo-buckets.json    {len(elo_dict)} buckets")

    # ---- novice vocabulary -------------------------------------------------
    meta = pickle.loads((HERE / "vendor" / "meta.pkl").read_bytes())
    itos = [meta["itos"][i] for i in range(len(meta["itos"]))]
    (models / "novice-vocab.json").write_text(json.dumps({
        "itos": itos,
        "block_size": 1023,
    }))
    print(f"  novice-vocab.json        {len(itos)} characters")

    # ---- encoding fixtures -------------------------------------------------
    cases = []
    for label, fen in FIXTURE_FENS:
        board = chess.Board(fen)
        board_input, elo_self, elo_oppo, legal = preprocessing(
            fen, 1500, 1500, elo_dict, all_moves_dict)
        flat = board_input.flatten()
        # The tensor is all 0s and 1s, so the set indices describe it exactly
        # and keep the fixture file readable.
        ones = [int(i) for i in (flat == 1.0).nonzero().flatten().tolist()]
        assert float(flat.sum()) == len(ones), "tensor is not binary"

        # The move the app must send back to chess.js: chosen in mirrored
        # space, un-mirrored when Black is to move.
        sample = next(iter(board.legal_moves)).uci()
        mirrored_sample = (mirror_move(sample) if board.turn == chess.BLACK
                           else sample)

        cases.append({
            "label": label,
            "fen": fen,
            "turn": "w" if board.turn == chess.WHITE else "b",
            "eloSelf": elo_self,
            "eloOppo": elo_oppo,
            "onesCount": len(ones),
            "ones": ones,
            "legalIndices": sorted(int(i) for i in
                                   legal.nonzero().flatten().tolist()),
            "sampleMoveOnRealBoard": sample,
            "sampleMoveInModelSpace": mirrored_sample,
        })
        print(f"  fixture: {label:38s} {len(ones):3d} set cells · "
              f"{int(legal.sum()):3d} legal")

    (fixtures / "maia-encoding.json").write_text(json.dumps({
        "note": "Generated by jtrax-ai/tools/export_web_assets.py. "
                "Do not hand-edit — regenerate.",
        "channels": 18,
        "cases": cases,
    }, indent=1) + "\n")

    print(f"\nwrote to {models} and {fixtures}")
    print("\nThe .onnx files are NOT copied here — they are gitignored and "
          "fetched separately;\nsee public/models/README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
