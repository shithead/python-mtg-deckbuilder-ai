"""Scrape / import Magic: The Gathering decklists from multiple sources.

Supported sources:
- magicgg: magic.gg decklist pages (tournament results)
- mtggoldfish: MTGGoldfish archetype pages (metagame decks)
- files: local exported decklists under data/imports/

All sources normalise to ``data/decks/<format_slug>/<source>--...txt`` so that
``import_collection.py`` can discover them automatically.
"""

import argparse
import hashlib
import html
import json
import os
import re
import time
from dataclasses import dataclass, field
from urllib.error import URLError
from urllib.parse import unquote_plus, urljoin
from urllib.request import Request, urlopen

BASE_URL = "https://magic.gg"
DECKLIST_INDEX = f"{BASE_URL}/decklists"
GOLDFISH_BASE = "https://www.mtggoldfish.com"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
DEFAULT_OUTPUT = os.path.join("data", "decks")
DEFAULT_IMPORT_DIR = os.path.join("data", "imports")
DELAY = 1.0


@dataclass
class Deck:
    title: str = ""
    event_name: str = ""
    event_date: str = ""
    format: str = ""
    source: str = ""
    maindeck: list[str] = field(default_factory=list)
    sideboard: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = []
        if self.title:
            lines.append(f"// {self.title}")
        if self.event_name:
            lines.append(f"// {self.event_name}")
        if self.event_date:
            lines.append(f"// {self.event_date}")
        if self.format:
            lines.append(f"// {self.format}")
        if self.source:
            lines.append(f"// source: {self.source}")
        lines.append("")
        for card in self.maindeck:
            lines.append(card)
        if self.sideboard:
            lines.append("")
            for card in self.sideboard:
                lines.append(f"SB: {card}")
        lines.append("")
        return "\n".join(lines)

    def content_hash(self) -> str:
        payload = "\n".join(sorted(self.maindeck + [f"SB:{c}" for c in self.sideboard]))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _fetch(url: str, retries: int = 3, delay: float = 1.0) -> str:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except (URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(delay * attempt)
    raise last_exc


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-") or "unknown"


NORMALISED_FORMATS = {
    "standard": "Standard",
    "historic": "Historic",
    "modern": "Modern",
    "vintage cube": "Vintage Cube",
    "pioneer": "Pioneer",
    "legacy": "Legacy",
    "pauper": "Pauper",
    "timeless": "Timeless",
    "commander": "Commander",
    "brawl": "Brawl",
    "explorer": "Explorer",
    "alchemy": "Alchemy",
    "vintage": "Vintage",
}


def _normalise_format(fmt: str) -> str:
    fmt = (fmt or "").strip()
    return NORMALISED_FORMATS.get(fmt.lower(), fmt or "Unknown")


def _save_deck(deck: Deck, out_dir: str, page_slug: str, index: int) -> str:
    fmt_slug = _slugify(_normalise_format(deck.format))
    target_dir = os.path.join(out_dir, fmt_slug)
    os.makedirs(target_dir, exist_ok=True)
    title_part = _slugify(deck.title)[:40] if deck.title else f"deck-{index + 1}"
    source_part = _slugify(deck.source) if deck.source else "src"
    filename = f"{source_part}--{page_slug}--{index + 1:03d}--{title_part}.txt"
    filepath = os.path.join(target_dir, filename)
    with open(filepath, "w") as f:
        f.write(deck.to_text())
    return filepath


# ---------------------------------------------------------------------------
# Source: magic.gg
# ---------------------------------------------------------------------------
def _extract_decklist_body(html_text: str) -> str:
    start_marker = "decklistBody:["
    start = html_text.index(start_marker)
    array_start = start + len(start_marker) - 1
    end_candidates = ["],standings", "],slug:", "],headerData", "],metaTitle"]
    array_end = None
    for candidate in end_candidates:
        idx = html_text.find(candidate, start)
        if idx != -1:
            array_end = idx + 1
            break
    if array_end is None:
        raise ValueError("decklistBody array end not found")
    return json.loads(html_text[array_start:array_end])[0]


_DECKLIST_RE = re.compile(r"<deck-list\s+([^>]+)>(.*?)</deck-list>", re.DOTALL)
_ATTR_RE = re.compile(r'(\S+)="([^"]*)"')
_MAINDECK_RE = re.compile(r"<main-deck>\n?(.*?)</main-deck>", re.DOTALL)
_SIDEBOARD_RE = re.compile(r"<side-board>\n?(.*?)</side-board>", re.DOTALL)
_CARD_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$", re.MULTILINE)


def _parse_attrs(attr_string: str) -> dict:
    return dict(_ATTR_RE.findall(attr_string))


def _parse_card_list(text: str) -> list[str]:
    cards = []
    for match in _CARD_RE.finditer(text):
        cards.append(f"{int(match.group(1))} {match.group(2).strip()}")
    return cards


def _parse_magicgg_body(body: str) -> list[Deck]:
    decks = []
    for deck_match in _DECKLIST_RE.finditer(body):
        attrs = _parse_attrs(deck_match.group(1))
        inner = deck_match.group(2)
        maindeck_match = _MAINDECK_RE.search(inner)
        sideboard_match = _SIDEBOARD_RE.search(inner)
        decks.append(Deck(
            title=attrs.get("deck-title", ""),
            event_name=attrs.get("event-name", ""),
            event_date=attrs.get("event-date", ""),
            format=attrs.get("format", ""),
            source="magicgg",
            maindeck=_parse_card_list(maindeck_match.group(1)) if maindeck_match else [],
            sideboard=_parse_card_list(sideboard_match.group(1)) if sideboard_match else [],
        ))
    return decks


def fetch_magicgg(max_pages=None, delay=DELAY, formats=None):
    html_text = _fetch(f"{BASE_URL}/decklists")
    urls = set()
    for match in re.finditer(r'href="(/decklists/[^"]+)"', html_text):
        full = urljoin(BASE_URL, match.group(1))
        if full != f"{BASE_URL}/decklists":
            urls.add(full)
    urls = sorted(urls)
    if max_pages:
        urls = urls[:max_pages]

    wanted = {_normalise_format(f) for f in formats} if formats else None
    decks = []
    for url in urls:
        try:
            body = _extract_decklist_body(_fetch(url))
            page_decks = _parse_magicgg_body(body)
        except Exception:
            continue
        for d in page_decks:
            if wanted and _normalise_format(d.format) not in wanted:
                continue
            decks.append(d)
        time.sleep(delay)
    return decks


# ---------------------------------------------------------------------------
# Source: MTGGoldfish
# ---------------------------------------------------------------------------
def _goldfish_deck_from_archetype(html_text: str, fmt: str, title: str) -> Deck | None:
    match = re.search(r'id="deck_input_deck"\s+value="([^"]*)"', html_text)
    if not match:
        return None
    raw = html.unescape(match.group(1))
    # MTGGoldfish uses \n between cards and "sideboard" separator
    main = []
    side = []
    target = main
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower() == "sideboard":
            target = side
            continue
        m = re.match(r"^(\d+)\s+(.+)$", line)
        if m:
            target.append(f"{m.group(1)} {m.group(2).strip()}")
    if not main:
        return None
    return Deck(title=title, format=fmt, source="mtggoldfish", maindeck=main, sideboard=side)


def fetch_mtggoldfish(formats=None, max_decks=20, delay=DELAY):
    wanted = {_normalise_format(f) for f in formats} if formats else None
    fmt_slugs = {
        "Standard": "standard", "Modern": "modern", "Pioneer": "pioneer",
        "Historic": "historic", "Explorer": "explorer", "Timeless": "timeless",
        "Pauper": "pauper", "Legacy": "legacy", "Vintage": "vintage",
    }
    if wanted:
        fmt_slugs = {k: v for k, v in fmt_slugs.items() if k in wanted}
    if not fmt_slugs:
        fmt_slugs = {"Standard": "standard"}

    decks = []
    seen = set()
    for fmt, slug in fmt_slugs.items():
        try:
            page = _fetch(f"{GOLDFISH_BASE}/metagame/{slug}#paper")
        except Exception:
            continue
        # archetype links
        links = set(re.findall(r'href="(/archetype/[^"]+)"', page))
        for link in list(links)[:max_decks]:
            try:
                arch_html = _fetch(f"{GOLDFISH_BASE}{link}")
            except Exception:
                continue
            title = link.rsplit("/", 1)[-1].replace("-", " ").title()
            deck = _goldfish_deck_from_archetype(arch_html, fmt, title)
            if deck and deck.content_hash() not in seen:
                seen.add(deck.content_hash())
                decks.append(deck)
            time.sleep(delay)
        time.sleep(delay)
    return decks


# ---------------------------------------------------------------------------
# Source: local files
# ---------------------------------------------------------------------------
def fetch_files(import_dir=DEFAULT_IMPORT_DIR, formats=None):
    if not os.path.isdir(import_dir):
        return []
    wanted = {_normalise_format(f) for f in formats} if formats else None
    decks = []
    for root, _, files in os.walk(import_dir):
        for fname in files:
            if not fname.endswith((".txt", ".dec", ".mwdeck")):
                continue
            path = os.path.join(root, fname)
            main = []
            side = []
            target = main
            fmt = None
            title = os.path.splitext(fname)[0]
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("//"):
                        val = line[2:].strip()
                        if not fmt and val.lower() in {f.lower() for f in NORMALISED_FORMATS}:
                            fmt = _normalise_format(val)
                        continue
                    if line.lower() == "sideboard":
                        target = side
                        continue
                    if line.startswith("SB:"):
                        side.append(line[3:].strip())
                        continue
                    m = re.match(r"^(\d+)\s+(.+)$", line)
                    if m:
                        target.append(f"{m.group(1)} {m.group(2).strip()}")
            if wanted and (fmt is None or _normalise_format(fmt) not in wanted):
                # fall back: try directory name
                parent = os.path.basename(root)
                fmt = _normalise_format(parent)
                if wanted and fmt not in wanted:
                    continue
            if not fmt:
                fmt = _normalise_format(os.path.basename(root))
            if not main:
                continue
            decks.append(Deck(title=title, format=fmt, source="files", maindeck=main, sideboard=side))
    return decks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
ADAPTERS = {
    "magicgg": lambda **kw: fetch_magicgg(**kw),
    "mtggoldfish": lambda **kw: fetch_mtggoldfish(**kw),
    "files": lambda **kw: fetch_files(**kw),
}


def main():
    parser = argparse.ArgumentParser(description="Scrape/import MTG decklists from multiple sources")
    parser.add_argument("--sources", type=str, default="magicgg",
                        help="Comma-separated sources: magicgg,mtggoldfish,files")
    parser.add_argument("--formats", type=str, default=None,
                        help="Comma-separated formats to keep")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--import-dir", type=str, default=DEFAULT_IMPORT_DIR)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-decks", type=int, default=20)
    parser.add_argument("--delay", type=float, default=DELAY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    formats = [f.strip() for f in args.formats.split(",")] if args.formats else None

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    all_decks = []
    for src in sources:
        if src not in ADAPTERS:
            print(f"Unknown source: {src}")
            continue
        print(f"\n=== Source: {src} ===")
        try:
            if src == "magicgg":
                decks = ADAPTERS[src](max_pages=args.max_pages, delay=args.delay, formats=formats)
            elif src == "mtggoldfish":
                decks = ADAPTERS[src](formats=formats, max_decks=args.max_decks, delay=args.delay)
            elif src == "files":
                decks = ADAPTERS[src](import_dir=args.import_dir, formats=formats)
            else:
                decks = ADAPTERS[src](formats=formats)
        except Exception as exc:
            print(f"  Source failed: {exc}")
            decks = []
        print(f"  Fetched {len(decks)} decks")
        all_decks.extend(decks)

    # Global dedup
    seen = set()
    unique = []
    for d in all_decks:
        h = d.content_hash()
        if h in seen:
            continue
        seen.add(h)
        unique.append(d)

    per_format = {}
    saved = 0
    for i, deck in enumerate(unique):
        fmt = _normalise_format(deck.format)
        if formats and fmt not in {_normalise_format(f) for f in formats}:
            continue
        if args.dry_run:
            print(f"  [dry-run] {fmt}: {deck.title} ({deck.source})")
        else:
            _save_deck(deck, out_dir, _slugify(deck.source or "src"), i)
            saved += 1
        per_format[fmt] = per_format.get(fmt, 0) + 1

    print("\nSummary:")
    for fmt, count in sorted(per_format.items()):
        print(f"  {fmt}: {count}")
    print(f"Total unique decks: {len(unique)}")
    if not args.dry_run:
        print(f"Saved: {saved} -> {os.path.abspath(out_dir)}/<format_slug>/")


if __name__ == "__main__":
    main()