from database.mtgtools import Database
from mtgtools.PCardList import PCardList
import pandas as pd
import re
import os
import glob
from os.path import join
from persistent.list import PersistentList
import random
import sys

DATA_DIR = os.path.abspath("./data")

db = Database()

# ---------------------------------------------------------------------------
# Build lookup indexes ONCE from db.cards (O(n) scan) instead of calling
# where_exactly (also O(n)) for every single row/deck-card.
# where_exactly is a pure linear scan over 60k+ cards with no index.
# With ~5000 CSV rows + 595 decks x ~60 cards ~= 40k calls, the naive
# approach does ~2.4 billion card comparisons.
# With indexes it drops to: 1 scan to build + O(1) dict lookups.
# ---------------------------------------------------------------------------

def _build_indexes(cards):
    """Build name->[cards] and (name,set)->[cards] dicts from a PCardList."""
    by_name = {}
    by_name_set = {}
    total = len(cards)
    for i, card in enumerate(cards):
        name = (card.name or "").lower()
        set_code = (card.set or "").lower() if hasattr(card, "set") else ""
        by_name.setdefault(name, []).append(card)
        by_name_set.setdefault((name, set_code), []).append(card)
        if i % 5000 == 0:
            sys.stdout.write(f"\rBuilding lookup indexes... {i}/{total}")
            sys.stdout.flush()
    print(f"\rBuilding lookup indexes... {total}/{total}  ({len(by_name)} unique names, {len(by_name_set)} name+set pairs)")
    return by_name, by_name_set


def _lookup(name, set_code=None):
    """Look up a card in the pre-built indexes."""
    name_key = name.lower()
    if set_code:
        matches = _name_set_index.get((name_key, set_code.lower()), [])
        if matches:
            return matches[0]
    matches = _name_index.get(name_key, [])
    return random.choice(matches) if matches else None


def _load_deck_from_file(path):
    """Parse a decklist file using the pre-built index (no where_exactly)."""
    deck = PCardList()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            # Optional sideboard prefix
            is_sb = line.startswith("SB:")
            if is_sb:
                line = line[3:].strip()

            # "4 Lightning Bolt"  ->  count=4  name="Lightning Bolt"
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            try:
                count = int(parts[0])
            except ValueError:
                continue

            card_name = parts[1]
            # Strip [set] / (set) annotations
            card_name = re.sub(r"\s*[\[\(][^\]\)]+[\]\)]\s*", " ", card_name).strip()

            card = _lookup(card_name)
            if card:
                for _ in range(count):
                    deck.append(card)
            else:
                print(f"  NOT FOUND: {card_name}")
    return deck


_name_index, _name_set_index = _build_indexes(db.cards)

# ---------------------------------------------------------------------------
# 1. Basic Collection from CSV
# ---------------------------------------------------------------------------
cards = PCardList()
cards.name = "Basic Collection"
print(f"\nLoading {cards.name} from CSV ...")
_csv_cols = ["Count", "Name", "Edition Code"]
_csv_files = glob.glob(join(DATA_DIR, "mtgcb-collection-*.csv"))
if not _csv_files:
    raise FileNotFoundError(f"No mtgcb-collection-*.csv found in {DATA_DIR}")
_csv_path = sorted(_csv_files)[-1]  # use the most recent file
print(f"  using {os.path.basename(_csv_path)}")
data = pd.read_csv(
    _csv_path,
    usecols=_csv_cols,
)  # type: ignore[arg-type]
print(f"  {len(data)} rows")
for idx in data.index:
    amount, name, edition_code = data.iloc[idx]
    name = re.sub(r" \(.*\)", "", name)
    card = _lookup(name, edition_code)
    if card:
        for _ in range(amount):
            cards.append(card)
    else:
        print(f"  NOT FOUND: {name} ({edition_code})")

print(cards)
print(f"Total: {len(cards)}")
db.root.basic_collection = cards
db.commit()
print("  committed basic_collection")

# ---------------------------------------------------------------------------
# 2. WCC Collection from deck files
# ---------------------------------------------------------------------------
cards = PCardList()
cards.name = "WCC Collection"
wccdecks = PersistentList()
print(f"\nLoading {cards.name} from deck files ...")
wcc_dir = join(DATA_DIR, "magic_WCC_decks")
wcc_files = sorted(os.listdir(wcc_dir))
for i, f in enumerate(wcc_files):
    sys.stdout.write(f"\r  [{i+1}/{len(wcc_files)}] {f[:50]}")
    sys.stdout.flush()
    wccdeck = _load_deck_from_file(join(wcc_dir, f))
    cards.extend(wccdeck)
    wccdecks.append(wccdeck)
print(f"\r  [{len(wcc_files)}/{len(wcc_files)}]  done.")

print(cards)
print(f"Total: {len(cards)}")
db.root.wcc_collection = cards
db.root.wcc_decks = wccdecks
db.commit()
print("  committed wcc_collection and wcc_decks")
