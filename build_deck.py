import sys
from typing import cast
from mtgtools.PCard import PCard
from database.mtgtools import Database
from database.VectorDB import VectorSearcher
from environment.Deck import Deck
from environment.Constructor import Constructor
from ai.CardEncoder import CardEncoder

def _color_count(mana_cost: str) -> dict:
    if not mana_cost:
        return {}
    counts = {}
    for ch in mana_cost.upper():
        if ch in "WUBRG":
            counts[ch] = counts.get(ch, 0) + 1
    return counts

def _land_count(total_cmc: float, spell_count: int) -> int:
    avg_cmc = total_cmc / max(spell_count, 1)
    if avg_cmc <= 1.5:
        return 17
    elif avg_cmc <= 2.0:
        return 19
    elif avg_cmc <= 2.5:
        return 21
    elif avg_cmc <= 3.0:
        return 23
    else:
        return 25

COLOR_TO_BASIC = {"W": "plains", "U": "island", "B": "swamp", "R": "mountain", "G": "forest"}

def main(query_text: str):
    print("Loading database...", flush=True)
    db = Database(check_update=False)
    pool = list(db.loadPool())
    pool_names_orig = {c.name.lower() for c in pool}
    deck = Deck(maxsize=60)
    ctor = Constructor()
    searcher = VectorSearcher()
    query_emb = CardEncoder().encode_text(query_text)

    name_idx = {c.name.lower(): c for c in pool}

    doc_names = searcher.suggest_documents(query_text, n_results=30)
    theme_card = None
    for doc in doc_names:
        for line in doc.split("\n"):
            if line.startswith("name: "):
                cname = line[6:].strip().lower()
                cand = name_idx.get(cname)
                if cand is not None:
                    theme_card = cand
                    break
        if theme_card:
            break

    if theme_card is None:
        print(f"No matching card from your collection for '{query_text}'")
        return
    theme_card = cast(PCard, theme_card)
    print(f"Theme: {theme_card.name} ({theme_card.type_line})", flush=True)
    deck._cards.append(theme_card)
    pool = [c for c in pool if c is not theme_card]

    theme_colors = set(getattr(theme_card, "colors", []) or [])
    candidates = [
        c for c in pool
        if "basic" not in ((getattr(c, "type_line", None) or getattr(c, "type", "") or "").lower())
        and (set(getattr(c, "colors", []) or []) & theme_colors
             or not getattr(c, "colors", []))
    ]

    total_cmc = float(getattr(theme_card, "cmc", 0) or 0)
    color_pips = _color_count(getattr(theme_card, "mana_cost", None) or "")

    for i in range(60):
        ranked = ctor.rank_cards(deck, candidates[:80], alpha=0.6, query_emb=query_emb, query_weight=0.2)
        if not ranked:
            break
        best, score = ranked[0]
        if score < 0.3:
            break
        name_count = sum(1 for c in deck if c.name == best.name)
        if name_count >= 4:
            candidates = [c for c in candidates if c.name != best.name]
            continue

        target_lands = _land_count(total_cmc, len(deck))
        spells_allowed = 60 - target_lands
        if len(deck) >= spells_allowed:
            break

        print(f"  + {best.name} ({score:.3f})", flush=True)
        deck._cards.append(best)
        candidates = [c for c in candidates if c is not best]
        total_cmc += float(getattr(best, "cmc", 0) or 0)
        for col, cnt in _color_count(getattr(best, "mana_cost", None) or "").items():
            color_pips[col] = color_pips.get(col, 0) + cnt

    target_lands = _land_count(total_cmc, len(deck))
    total_pips = sum(color_pips.values())
    basics = [c for c in pool
              if "basic" in ((getattr(c, "type_line", None) or getattr(c, "type", "") or "").lower())
              and c is not theme_card]
    if total_pips == 0:
        n = 0
        for basic_name in COLOR_TO_BASIC.values():
            matches = [c for c in basics if c.name.lower() == basic_name]
            chunk = target_lands // 5 + (1 if n < target_lands % 5 else 0)
            if chunk and matches:
                for _ in range(chunk):
                    if len(deck) < 60:
                        deck._cards.append(matches[0])
            n += 1
    else:
        for col, basic_name in COLOR_TO_BASIC.items():
            needed = round(color_pips.get(col, 0) / total_pips * target_lands)
            if needed == 0:
                continue
            matches = [c for c in basics if c.name.lower() == basic_name]
            for _ in range(min(needed, 60 - len(deck))):
                if matches:
                    deck._cards.append(matches[0])

    pool_plus_basics = pool_names_orig | {"plains", "island", "swamp", "mountain", "forest"}
    for c in deck:
        if c.name.lower() not in pool_plus_basics:
            print(f"  WARNING: {c.name} NOT in user collection!", flush=True)
    assert all(c.name.lower() in pool_plus_basics for c in deck), "Cards outside collection!"
    print(f"\nDeck ({len(deck)} cards, all from your collection):", flush=True)
    for c in deck:
        print(f"  {c.name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_deck.py <search query>")
        print("Example: python build_deck.py 'aggressive red creature with haste'")
        sys.exit(1)
    main(sys.argv[1])
