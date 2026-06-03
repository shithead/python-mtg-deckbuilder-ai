"""Scrape Magic: The Gathering decklists from magic.gg.

Extracts decklists from magic.gg's Nuxt SSR-rendered pages, parsing the
<deck-list> markup embedded in the __NUXT__ state.
"""
import json
import re
import os
import time
from dataclasses import dataclass, field
from urllib.request import urlopen, Request
from urllib.parse import urljoin

BASE_URL = "https://magic.gg"
DECKLIST_INDEX = f"{BASE_URL}/decklists"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "magic_WCC_decks")
DELAY = 1.0  # seconds between requests


@dataclass
class Deck:
    title: str = ""
    event_name: str = ""
    event_date: str = ""
    format: str = ""
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
        lines.append("")
        for card in self.maindeck:
            lines.append(card)
        if self.sideboard:
            lines.append("")
            for card in self.sideboard:
                lines.append(f"SB: {card}")
        lines.append("")
        return "\n".join(lines)


def _fetch(url: str) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req) as resp:
        return resp.read().decode("utf-8")


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

    array_text = html_text[array_start:array_end]
    return json.loads(array_text)[0]


_DECKLIST_RE = re.compile(
    r"<deck-list\s+([^>]+)>(.*?)</deck-list>", re.DOTALL
)
_ATTR_RE = re.compile(r'(\S+)="([^"]*)"')
_MAINDECK_RE = re.compile(r"<main-deck>\n?(.*?)</main-deck>", re.DOTALL)
_SIDEBOARD_RE = re.compile(r"<side-board>\n?(.*?)</side-board>", re.DOTALL)
_CARD_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$", re.MULTILINE)


def _parse_attrs(attr_string: str) -> dict:
    return dict(_ATTR_RE.findall(attr_string))


def _parse_card_list(text: str) -> list[str]:
    cards = []
    for match in _CARD_RE.finditer(text):
        qty = int(match.group(1))
        name = match.group(2).strip()
        cards.append(f"{qty} {name}")
    return cards


def parse_decklist_body(body: str) -> list[Deck]:
    decks = []
    for deck_match in _DECKLIST_RE.finditer(body):
        attrs = _parse_attrs(deck_match.group(1))
        inner = deck_match.group(2)

        maindeck_match = _MAINDECK_RE.search(inner)
        sideboard_match = _SIDEBOARD_RE.search(inner)

        maindeck = _parse_card_list(maindeck_match.group(1)) if maindeck_match else []
        sideboard = _parse_card_list(sideboard_match.group(1)) if sideboard_match else []

        decks.append(Deck(
            title=attrs.get("deck-title", ""),
            event_name=attrs.get("event-name", ""),
            event_date=attrs.get("event-date", ""),
            format=attrs.get("format", ""),
            maindeck=maindeck,
            sideboard=sideboard,
        ))
    return decks


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def save_decks(decks: list[Deck], page_slug: str) -> list[str]:
    os.makedirs(DATA_DIR, exist_ok=True)
    saved = []
    for i, deck in enumerate(decks):
        title_part = _slugify(deck.title)[:40] if deck.title else f"deck-{i+1}"
        filename = f"{page_slug}--{i+1:03d}--{title_part}.txt"
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "w") as f:
            f.write(deck.to_text())
        saved.append(filepath)
    return saved


def fetch_decklists(url: str) -> tuple[list[Deck], str]:
    html_text = _fetch(url)
    body = _extract_decklist_body(html_text)
    decks = parse_decklist_body(body)
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    return decks, slug


def list_decklist_urls() -> list[str]:
    html_text = _fetch(DECKLIST_INDEX)
    urls = set()
    for match in re.finditer(r'href="(/decklists/[^"]+)"', html_text):
        full_url = urljoin(BASE_URL, match.group(1))
        if full_url != DECKLIST_INDEX:
            urls.add(full_url)
    return sorted(urls)


def main():
    urls = list_decklist_urls()
    print(f"Found {len(urls)} decklist pages")

    total_decks = 0
    for i, url in enumerate(urls):
        try:
            decks, slug = fetch_decklists(url)
            saved = save_decks(decks, slug)
            total_decks += len(decks)
            print(f"  [{i+1}/{len(urls)}] {url.rsplit('/', 1)[-1][:60]} "
                  f"→ {len(decks)} decks")
        except Exception as e:
            print(f"  [{i+1}/{len(urls)}] SKIP {url}: {e}")

        if i < len(urls) - 1:
            time.sleep(DELAY)

    print(f"\nDone. {total_decks} decks saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()
