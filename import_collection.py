from database.mtgtools import Database
from mtgtools.PCardList import PCardList
import pandas as pd
import re
import os
import glob
from os.path import join
from persistent.list import PersistentList
from persistent.mapping import PersistentMapping
import random
import sys

DATA_DIR = os.path.abspath("./data")

db = Database()

# ---------------------------------------------------------------------------
# Build lookup indexes ONCE from db.cards (O(n) scan) instead of calling
# where_exactly (also O(n)) for every single row/deck-card.
# ---------------------------------------------------------------------------
def _build_indexes(cards):
    """Build name->[cards] and (name,set)->[cards] dicts from a PCardList."""
    by_name = {}
    by_name_set = {}
    total = len(cards)
    for i, card in enumerate(cards):
        names = {_normalize_name(card.name)}
        printed = getattr(card, "printed_name", None)
        if printed:
            names.add(_normalize_name(printed))
        set_code = (card.set or "").lower() if hasattr(card, "set") else ""
        for n in names:
            by_name.setdefault(n, []).append(card)
            by_name_set.setdefault((n, set_code), []).append(card)
        if i % 5000 == 0:
            sys.stdout.write(f"\rBuilding lookup indexes... {i}/{total}")
            sys.stdout.flush()
    print(f"\rBuilding lookup indexes... {total}/{total}  ({len(by_name)} unique names, {len(by_name_set)} name+set pairs)")
    return by_name, by_name_set


def _normalize_name(name):
    """Normalize card name for lookup: lowercase and replace ae-ligature."""
    return (name or "").lower().replace("æ", "ae")


def _lookup(name, set_code=None):
    """Look up a card in the pre-built indexes (canonical name or printed_name)."""
    name_key = _normalize_name(name)
    if set_code:
        matches = _name_set_index.get((name_key, set_code.lower()), [])
        if matches:
            return matches[0]
    matches = _name_index.get(name_key, [])
    if not matches:
        matches = _name_index.get(name.lower(), [])
    return random.choice(matches) if matches else None


KNOWN_FORMATS = {
    "standard", "historic", "modern", "vintage cube", "pioneer",
    "legacy", "pauper", "timeless", "commander", "brawl",
}


def _infer_format(comments, source=None, path=None):
    """Infer a format tag from deck comments, source name, or filename."""
    for c in comments:
        low = c.lower()
        for fmt in KNOWN_FORMATS:
            if fmt in low:
                return fmt.title() if fmt != "vintage cube" else "Vintage Cube"
    if source:
        low = source.lower().replace("_", " ").replace("-", " ")
        for fmt in KNOWN_FORMATS:
            if fmt in low:
                return fmt.title() if fmt != "vintage cube" else "Vintage Cube"
    if path:
        low = os.path.basename(path).lower()
        for fmt in KNOWN_FORMATS:
            if fmt.replace(" ", "") in low.replace(" ", ""):
                return fmt.title() if fmt != "vintage cube" else "Vintage Cube"
    return None


def _load_deck_from_file(path):
    """Parse a decklist file. Returns (maindeck, sideboard, format_tag)."""
    maindeck = PCardList()
    sideboard = PCardList()
    comments = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("//"):
                comments.append(line[2:].strip())
                continue

            is_sb = line.startswith("SB:")
            if is_sb:
                line = line[3:].strip()
                target = sideboard
            else:
                target = maindeck

            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            try:
                count = int(parts[0])
            except ValueError:
                continue

            card_name = parts[1]
            card_name = re.sub(r"\s*[\[\(][^\]\)]+[\]\)]\s*", " ", card_name).strip()

            card = _lookup(card_name)
            if card:
                for _ in range(count):
                    target.append(card)
            else:
                print(f"  NOT FOUND: {card_name}")

    fmt = _infer_format(comments, path=path)
    return maindeck, sideboard, fmt


def _discover_sources():
    """Return {source_name: directory_path}."""
    sources = {}
    decks_root = join(DATA_DIR, "decks")
    if os.path.isdir(decks_root):
        for name in sorted(os.listdir(decks_root)):
            p = join(decks_root, name)
            if os.path.isdir(p):
                sources[name] = p
    wcc_dir = join(DATA_DIR, "magic_WCC_decks")
    if os.path.isdir(wcc_dir):
        sources.setdefault("wcc", wcc_dir)
    return sources


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
_csv_path = sorted(_csv_files)[-1]
print(f"  using {os.path.basename(_csv_path)}")
data = pd.read_csv(_csv_path, usecols=_csv_cols)
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
# 2. Deck Collections from (possibly multiple) sources
# ---------------------------------------------------------------------------
deck_collections = PersistentMapping()
deck_sideboards = PersistentMapping()
deck_formats = PersistentMapping()
deck_pools = PersistentMapping()

sources = _discover_sources()
if not sources:
    raise FileNotFoundError(f"No deck source directories found under {DATA_DIR}")

for source, src_dir in sources.items():
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".txt"))
    maindecks = PersistentList()
    sideboards = PersistentList()
    formats = PersistentList()
    combined = PCardList()
    combined.name = f"{source} Collection"
    print(f"\nLoading {source} from {src_dir} ({len(files)} files) ...")
    for i, f in enumerate(files):
        sys.stdout.write(f"\r  [{i+1}/{len(files)}] {f[:50]}")
        sys.stdout.flush()
        main, side, fmt = _load_deck_from_file(join(src_dir, f))
        # Fall back to source-derived format tag if file had none
        if fmt is None:
            fmt = _infer_format([], source=source)
        maindecks.append(main)
        sideboards.append(side)
        formats.append(fmt)
        combined.extend(main)
    print(f"\r  [{len(files)}/{len(files)}]  done.")

    deck_collections[source] = maindecks
    deck_sideboards[source] = sideboards
    deck_formats[source] = formats
    deck_pools[source] = combined

    print(combined)
    print(f"Total: {len(combined)}")

db.root.deck_collections = deck_collections
db.root.deck_sideboards = deck_sideboards
db.root.deck_formats = deck_formats
db.root.deck_pools = deck_pools

# Backward-compatible aliases
if "wcc" in deck_collections:
    db.root.wcc_decks = deck_collections["wcc"]
    db.root.wcc_collection = deck_pools["wcc"]
    db.root.wcc_sideboards = deck_sideboards["wcc"]

db.commit()
print("  committed deck_collections, deck_pools, and backward-compatible aliases")