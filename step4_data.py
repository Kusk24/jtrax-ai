"""Step 4 — build the training text from real Lichess games.

Streams a monthly Lichess archive straight from database.lichess.org, keeps
only games where BOTH players sit in the target rating band, and writes them in
the exact character format the model reads:

    ;1.e4 e5 2.Nf3 Nc6 3.Bb5 a6

Never holds the archive in memory or on disk. The monthly files are ~30 GB
compressed and ~300 GB open; this decompresses a chunk at a time, keeps the
handful of games it wants, and stops as soon as it has enough. In practice it
downloads a few hundred MB, not thirty gigabytes.

The vocabulary is 32 characters and nothing else is allowed through:

    ' #+-.0123456789;=BKNOQRabcdefghx'

So clock comments, evaluations, result tokens ("1-0", "1/2-1/2") and black
move numbers ("1...") are all stripped. A single stray character the model has
no token for would corrupt the sequence.

Run:  conda activate jtrax-ai && python step4_data.py
"""

import argparse
import pathlib
import re
import sys
import time

import requests
import zstandard

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"

MONTH = "2023-01"  # older months are smaller; any month works
URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{}.pgn.zst"

# Strong human play. 2000+ on Lichess is a solid club/expert player — the
# strongest band with enough games to train on. Above ~2400 the games become
# too rare to fill a corpus from one month.
# For a beginner-like opponent instead, pass --elo-min 800 --elo-max 1200.
ELO_MIN, ELO_MAX = 2000, 2800
MIN_PLIES = 20  # skip instant resignations; they teach nothing
GAMES = 200_000
HELDOUT = 5_000  # frozen for evaluation, never trained on

VOCAB = set(" #+-.0123456789;=BKNOQRabcdefghx")

# Lichess movetext: "1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:02:58] } 2. Nf3"
COMMENT = re.compile(r"\{[^}]*\}")
BLACK_NUM = re.compile(r"\b\d+\.\.\.\s*")
WHITE_NUM = re.compile(r"\b(\d+)\.\s*")
RESULT = re.compile(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$")
ANNOTATION = re.compile(r"[?!]+")


def normalise(movetext):
    """Lichess movetext -> the model's character format, or None if unusable."""
    s = COMMENT.sub(" ", movetext)
    s = RESULT.sub("", s)
    s = ANNOTATION.sub("", s)
    s = BLACK_NUM.sub("", s)
    s = WHITE_NUM.sub(r"\1.", s)  # "1. e4" -> "1.e4"
    s = " ".join(s.split())
    if not s:
        return None
    # Anything outside the 32-char vocabulary means the model has no token for
    # it, so drop the whole game rather than feed it a character it cannot read.
    if not set(s) <= VOCAB:
        return None
    return ";" + s


class ResumableReader:
    """A file-like read() over HTTP that survives a dropped connection.

    Collecting a million games means streaming a couple of GB, and a single
    socket rarely lasts that long — the first attempt died with a broken pipe.
    On failure this reopens with a Range header at the exact byte offset, so
    the decompressor downstream sees one unbroken stream and never knows.
    """

    def __init__(self, url, retries=8, verbose=True):
        self.url = url
        self.retries = retries
        self.verbose = verbose
        self.pos = 0
        self.reconnects = 0
        self._open()

    def _open(self):
        headers = {"Range": f"bytes={self.pos}-"} if self.pos else {}
        resp = requests.get(self.url, stream=True, timeout=(30, 120),
                            headers=headers)
        resp.raise_for_status()
        if self.pos and resp.status_code != 206:
            raise RuntimeError("server ignored Range; cannot resume")
        self.raw = resp.raw

    def read(self, n):
        for attempt in range(self.retries):
            try:
                chunk = self.raw.read(n)
                self.pos += len(chunk)
                return chunk
            except Exception as exc:
                self.reconnects += 1
                wait = min(2 ** attempt, 30)
                if self.verbose:
                    print(f"    connection lost ({type(exc).__name__}) at "
                          f"{self.pos / 1e6:.0f} MB — resuming in {wait}s "
                          f"[{self.reconnects}]", flush=True)
                time.sleep(wait)
                try:
                    self._open()
                except Exception:
                    continue
        raise RuntimeError(f"gave up after {self.retries} reconnects")


def games_from_stream(url, elo_min, elo_max, min_plies, wanted, verbose=True):
    """Yield normalised games, decompressing the archive a chunk at a time."""
    dctx = zstandard.ZstdDecompressor()
    reader = dctx.stream_reader(ResumableReader(url, verbose=verbose))

    headers = {}
    kept = seen = 0
    tail = ""

    while kept < wanted:
        chunk = reader.read(1 << 22)  # 4 MB at a time
        if not chunk:
            break
        text = tail + chunk.decode("utf-8", errors="ignore")
        lines = text.split("\n")
        tail = lines.pop()  # last line may be cut mid-way

        for line in lines:
            line = line.strip()
            if line.startswith("["):
                key, _, rest = line[1:].partition(" ")
                headers[key] = rest.strip('] "')
                continue
            if not line or not line[0].isdigit():
                continue

            # A movetext line: this game is complete.
            seen += 1
            try:
                we = int(headers.get("WhiteElo", 0))
                be = int(headers.get("BlackElo", 0))
            except ValueError:
                headers = {}
                continue

            in_band = elo_min <= we <= elo_max and elo_min <= be <= elo_max
            long_enough = line.count(".") >= min_plies // 2
            if in_band and long_enough:
                g = normalise(line)
                if g:
                    kept += 1
                    yield g
                    if verbose and kept % 5000 == 0:
                        print(f"  kept {kept:,} / scanned {seen:,} "
                              f"({100 * kept / seen:.1f}%)", flush=True)
            headers = {}
            if kept >= wanted:
                break

    if verbose:
        print(f"  done: kept {kept:,} of {seen:,} games scanned")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--month", default=MONTH)
    ap.add_argument("--games", type=int, default=GAMES)
    ap.add_argument("--heldout", type=int, default=HELDOUT)
    ap.add_argument("--elo-min", type=int, default=ELO_MIN)
    ap.add_argument("--elo-max", type=int, default=ELO_MAX)
    ap.add_argument("--prefix", default="",
                    help="filename prefix, e.g. 'novice_' — keeps corpora "
                         "for different rating bands side by side")
    args = ap.parse_args()

    url = URL.format(args.month)
    total = args.games + args.heldout

    print(f"Lichess {args.month} · keeping games where BOTH players are "
          f"{args.elo_min}-{args.elo_max}")
    print(f"Target {total:,} games ({args.games:,} train + "
          f"{args.heldout:,} held out)")
    print(f"Streaming {url}\n")

    DATA.mkdir(exist_ok=True)
    train_path = DATA / f"{args.prefix}train.txt"
    val_path = DATA / f"{args.prefix}heldout.txt"
    partial = DATA / f"{args.prefix}collected.partial"

    # Written as they arrive rather than buffered to the end: an hour-long
    # download that dies at minute 55 should not lose an hour of work. The
    # existing train.txt is left untouched until the new corpus is complete.
    count = 0
    try:
        with partial.open("w") as fh:
            for g in games_from_stream(url, args.elo_min, args.elo_max,
                                       MIN_PLIES, total):
                fh.write(g + "\n")
                count += 1
    except KeyboardInterrupt:
        print(f"\n  interrupted after {count:,} games — keeping them")
    except requests.HTTPError as exc:
        print(f"\nDownload failed: {exc}")
        print("Check the month exists at https://database.lichess.org/standard/")
        return 1
    except RuntimeError as exc:
        print(f"\nStream gave up: {exc}")
        print(f"  {count:,} games were collected and kept.")

    collected = partial.read_text().splitlines()
    if len(collected) < 100:
        print(f"\nOnly {len(collected)} games matched. Widen the band or pick "
              "a different month.")
        return 1

    # Held-out games come off the END, so they are games the training set never
    # saw. Frozen once training starts: regenerating this after a run would
    # invalidate the before/after comparison.
    heldout = collected[-args.heldout:] if len(collected) > args.heldout else []
    train = collected[:-len(heldout)] if heldout else collected

    train_path.write_text("\n".join(train) + "\n")
    if heldout:
        val_path.write_text("\n".join(heldout) + "\n")
    partial.unlink()

    train_mb = train_path.stat().st_size / 1e6
    print(f"\n{'=' * 54}")
    print(f"  train     {len(train):,} games  {train_mb:.1f} MB  "
          f"-> data/{train_path.name}")
    if heldout:
        print(f"  held out  {len(heldout):,} games  "
              f"-> data/{val_path.name}")
    print(f"{'=' * 54}")
    print(f"\n  sample:\n  {train[0][:160]}")
    print("\nNext: upload data/ to Kaggle as a private dataset, then step5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
