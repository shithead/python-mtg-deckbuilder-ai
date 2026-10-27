try:
    import mtgtools
except ModuleNotFoundError:
    import os, shlex, sys
    cmd = "python " + shlex.quote(__file__) + " " + " ".join(shlex.quote(a) for a in sys.argv[1:])
    os.execvp("nix-shell", ["nix-shell", "--run", cmd])

import argparse
import torch
import random
import sys
from collections import Counter
from typing import cast

from mtgtools.PCard import PCard
from database.mtgtools import Database
from database.VectorDB import VectorSearcher
from environment.Deck import Deck
from environment.Constructor import Constructor, land_count
from ai.CardEncoder import CardEncoder


def _color_count(mana_cost: str) -> dict:
    if not mana_cost:
        return {}
    counts = {}
    for ch in mana_cost.upper():
        if ch in "WUBRG":
            counts[ch] = counts.get(ch, 0) + 1
    return counts




def _effect_mock(card):
    from types import SimpleNamespace
    return SimpleNamespace(
        name=getattr(card, "name", ""),
        type_line=getattr(card, "type_line", None) or getattr(card, "type", "") or "",
        mana_cost="",
        oracle_text=getattr(card, "oracle_text", "") or "",
    )

COLOR_TO_BASIC = {"W": "plains", "U": "island", "B": "swamp", "R": "mountain", "G": "forest"}


def _parse_args():
    parser = argparse.ArgumentParser(description="Build a deck from a search query / theme cards.")
    parser.add_argument("freetext", nargs="*", help="Free-text query, may include '3 color deck use X as theme'.")
    parser.add_argument("--colors", type=int, default=None, help="Number of colors to play (1-5).")
    parser.add_argument("--theme", action="append", default=[], help="Theme card name (repeatable, up to 3).")
    parser.add_argument("--query", default=None, help="Semantic query text.")
    parser.add_argument("--theme-colors", default=None, help="Color letters, e.g. WUG or W,U,G.")
    return parser.parse_args()


def _parse_freetext(text, pool_names):
    colors = None
    themes = []
    import re
    m = re.search(r"(\d)\s*(?:color|farbe|c)\b", text, re.IGNORECASE)
    if m:
        colors = int(m.group(1))
    low = text.lower()
    for name in pool_names:
        if name in low and name not in themes:
            themes.append(name)

    query = text
    for t in themes:
        query = query.replace(t, "")
    query = query.strip()
    if not query:
        query = text
    return colors, themes, query


def _theme_card_from_name(name, searcher, name_idx):
    doc_names = searcher.suggest_documents(name, n_results=30)
    for doc in doc_names:
        for line in doc.split("\n"):
            if line.startswith("name: "):
                cname = line[6:].strip().lower()
                cand = name_idx.get(cname)
                if cand is not None and cand.name.lower() == name.lower():
                    return cand
    return name_idx.get(name.lower())


def main():
    args = _parse_args()
    freetext = " ".join(args.freetext).strip()

    print("Loading database...", flush=True)
    db = Database(check_update=False)
    pool = list(db.loadPool())
    pool_names_orig = {c.name.lower() for c in pool}
    deck = Deck(maxsize=60)
    ctor = Constructor()
    searcher = VectorSearcher()

    colors = args.colors
    themes = [t.strip() for t in args.theme if t.strip()]
    query = args.query

    # theme-colors override
    theme_colors_override = None
    if args.theme_colors:
        theme_colors_override = set(c for c in args.theme_colors.upper().replace(",", "").replace(" ", "") if c in "WUBRG")

    if freetext:
        fc, ft, fq = _parse_freetext(freetext, pool_names_orig)
        if colors is None:
            colors = fc
        if not themes:
            themes = ft
        if query is None:
            query = fq

    if query is None:
        query = freetext if freetext else "creature"

    name_idx = {c.name.lower(): c for c in pool}

    # Resolve theme cards (up to 3)
    theme_cards = []
    for tname in themes:
        tc = _theme_card_from_name(tname, searcher, name_idx)
        if tc is not None:
            theme_cards.append(tc)

    if themes and not theme_cards:
        print(f"No matching theme card from your collection for: {themes}")
        return

    if not theme_cards:
        doc_names = searcher.suggest_documents(query, n_results=30)
        for doc in doc_names:
            for line in doc.split("\n"):
                if line.startswith("name: "):
                    cname = line[6:].strip().lower()
                    cand = name_idx.get(cname)
                    if cand is not None:
                        theme_cards.append(cand)
                        break
            if theme_cards:
                break

    if not theme_cards:
        print(f"No matching card from your collection for '{query}'")
        return

    # dedup
    seen = set()
    uniq = []
    for tc in theme_cards:
        if tc.name.lower() not in seen:
            seen.add(tc.name.lower())
            uniq.append(tc)
    theme_cards = uniq

    theme_colors = set()
    for tc in theme_cards:
        theme_colors |= {str(x).upper() for x in (getattr(tc, "colors", []) or [])}

    if theme_colors_override:
        theme_colors = theme_colors_override

    if colors is not None and colors > 0 and len(theme_colors) > colors:
        color_counts = Counter()
        for c in pool:
            for col in (getattr(c, "colors", []) or []):
                if col in theme_colors:
                    color_counts[col] += 1
        theme_colors = set([col for col, _ in color_counts.most_common(colors)])

    if not theme_colors:
        for tc in theme_cards:
            theme_colors |= {str(x).upper() for x in (getattr(tc, "colors", []) or [])}

    for tc in theme_cards:
        print(f"Theme: {tc.name} ({tc.type_line})", flush=True)

    # Theme-Karten so oft wie vorhanden (max 4) einfuegen
    for tc in theme_cards:
        avail = sum(1 for c in pool if c is tc or c.name.lower() == tc.name.lower())
        copies = min(4, max(1, avail))
        for _ in range(copies):
            deck._cards.append(tc)
    theme_names = {tc.name.lower() for tc in theme_cards}
    pool = [c for c in pool if c.name.lower() not in theme_names]

    total_cmc = sum(float(getattr(tc, "cmc", 0) or 0) for tc in theme_cards) / len(theme_cards)
    color_pips = {}
    for tc in theme_cards:
        for col, cnt in _color_count(getattr(tc, "mana_cost", None) or "").items():
            color_pips[col] = color_pips.get(col, 0) + cnt

    candidates = [
        c for c in pool
        if c.name.lower() in pool_names_orig
        and "basic" not in ((getattr(c, "type_line", None) or getattr(c, "type", "") or "").lower())
        and (not theme_colors or ({str(x).upper() for x in (getattr(c, "colors", []) or [])} & theme_colors)
             or not getattr(c, "colors", []))
        and c.name.lower() not in theme_names
    ]

    enc = CardEncoder()
    try:
        query_emb = enc.encode_text(query)
    except Exception:
        from types import SimpleNamespace
        mock = SimpleNamespace(name=query, type_line="", mana_cost="", oracle_text=query)
        query_emb = enc.encode(mock)

    # theme embedding (type + oracle only, no mana_cost) for text-synergy bonus
    theme_mocks = [_effect_mock(tc) for tc in theme_cards]
    theme_emb = enc.encode_many(theme_mocks).mean(dim=0)

    distinct_names = {tc.name.lower() for tc in theme_cards}

    for i in range(60):
        cand_list = candidates[:80]
        if not cand_list:
            break

        # embeddings (kein Modell-Score mehr)
        full_embs = enc.encode_many(cand_list)
        if len(deck) > 0:
            deck_mean = enc.encode_many(list(deck)).mean(dim=0)
            cos_deck = torch.nn.functional.cosine_similarity(deck_mean.unsqueeze(0), full_embs, dim=1)
        else:
            cos_deck = torch.zeros(len(cand_list))
        cos_query = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), full_embs, dim=1)
        effect_embs = enc.encode_many([_effect_mock(c) for c in cand_list])
        cos_theme = torch.nn.functional.cosine_similarity(theme_emb.unsqueeze(0), effect_embs, dim=1)

        # gewichtete score-aufteilung (kein Modell):
        # Deck 25%, Query 75%, Text-Synergie 3.0 (additiv)
        # Malus -0.1 fuer farblose (Artefakt-)Karten
        scored = []
        for idx, c in enumerate(cand_list):
            sc = (0.25 * float(cos_deck[idx])
                  + 0.75 * float(cos_query[idx])
                  + 3.0 * float(cos_theme[idx]))
            if not (getattr(c, "colors", []) or []):
                sc -= 0.1
            scored.append((c, sc))
        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            break
        best, score = scored[0]
        if score < 0.3:
            break
        name_count = sum(1 for c in deck if c.name == best.name)
        if name_count >= 4:
            candidates = [c for c in candidates if c.name != best.name]
            continue
        if best.name.lower() not in pool_names_orig:
            candidates = [c for c in candidates if c is not best]
            continue

        target_lands, _ = land_count(total_cmc, len(deck), deck)
        spells_allowed = 60 - target_lands
        if len(deck) >= spells_allowed:
            break

        print(f"  + {best.name} ({score:.3f})", flush=True)
        deck._cards.append(best)
        candidates = [c for c in candidates if c is not best]
        total_cmc += float(getattr(best, "cmc", 0) or 0)
        for col, cnt in _color_count(getattr(best, "mana_cost", None) or "").items():
            color_pips[col] = color_pips.get(col, 0) + cnt

        distinct_names.add(best.name.lower())
        # Query-Anker bleibt persistent (kein Umschalten)

    target_lands, _ = land_count(total_cmc, len(deck), deck)
    total_pips = sum(color_pips.values())
    basics = [c for c in pool
              if "basic" in ((getattr(c, "type_line", None) or getattr(c, "type", "") or "").lower())
              and c not in theme_cards]
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
            if col not in theme_colors:
                continue
            needed = round(color_pips.get(col, 0) / total_pips * target_lands)
            if needed == 0:
                continue
            matches = [c for c in basics if c.name.lower() == basic_name]
            for _ in range(min(needed, 60 - len(deck))):
                if matches:
                    deck._cards.append(matches[0])
    # fehlende Slots mit Basic Lands auffuellen
    while len(deck) < 60:
        col = next(iter(theme_colors)) if theme_colors else "G"
        basic_name = COLOR_TO_BASIC.get(col, "forest")
        matches = [c for c in basics if c.name.lower() == basic_name]
        if matches:
            deck._cards.append(matches[0])
        else:
            break

    non_collection = [c for c in deck if c.name.lower() not in pool_names_orig and "basic" not in ((getattr(c, "type_line", None) or getattr(c, "type", "") or "").lower())]
    if non_collection:
        print(f"\nWARNING: {len(non_collection)} cards not in collection: {[c.name for c in non_collection]}")
    else:
        print(f"\nDeck ({len(deck)} cards, all from your collection):", flush=True)
    for c in deck:
        print(f"  {c.name}")


if __name__ == "__main__":
    args = _parse_args()
    if len(sys.argv) < 2 and not args.theme and not args.query and not args.freetext:
        print("Usage: python build_deck.py <search query>")
        print("       python build_deck.py --colors 3 --theme 'High Alert' --query 'aggressive'")
        print("Example: python build_deck.py '3 Color Deck use High Alert as one theme card'")
        sys.exit(1)
    main()
