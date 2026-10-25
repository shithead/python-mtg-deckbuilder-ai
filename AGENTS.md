# AGENTS.md — python-mtg-deckbuilder-ai

Python 3.10+ project: AI-assisted MTG deck building. PyTorch synergy classifier,
ChromaDB semantic card search, mtgtools/ZODB card data.

## Layout

- `ai/` — `CardEncoder.py`, `DeckDataset.py`, `SynergyClassifier.py`, `train.py`
- `database/` — `mtgtools.py` (ZODB `Database`), `VectorDB.py` (`VectorSearcher`)
- `environment/` — `Card.py` (`AICard`), `Deck.py`, `Constructor.py`
- `test/` — pytest, 39 tests across 5 files
- top-level: `VectorDB.py` (builds ChromaDB index), `import_collection.py`, `scrape_decklists.py`
- `data/` — `mtgdb.fs` (ZODB, gitignored), `synergy_model.pt`, CSV, `magic_WCC_decks/`

## Setup

```bash
nix-shell                      # system deps + pip install -r requirements.txt
# or
./create_virtenv.sh            # bare venv only; then pip install -r requirements.txt
pip install -e .               # required for package imports
```

`nix-shell` runs `pip install -r requirements.txt` + `pip install -e .` in shellHook.
`requirements.txt` pins `torch>=2.4.0` (torchtext removed: unused).

## Commands

```bash
pytest test/                               # all tests
pytest test/test_deck.py::TestDeck::test_add_card_within_limit
pytest test/ -k "test_forward"
coverage run -m pytest test/
coverage html --omit="test/*,_build/*,/nix/*"

python ai/train.py                         # 25 epochs, ~30s CPU, saves data/synergy_model.pt
python VectorDB.py                         # rebuild ChromaDB index (deletes+recreates collection)
python import_collection.py                # import CSV + WCC decks into ZODB (needs data/mtgcb-collection-*.csv)
```

CI (`.github/workflows/test.yml`) runs `nix-shell --run "pytest test/ -v"` on push to `main`/`vector_database` and PRs to `main`.

## Architecture

- `SynergyClassifier`: MLP `768 -> 256 -> 1`, ~200K params. Input = `cat(deck_ctx 384d, card_emb 384d)`.
- `CardEncoder` (Phase A): frozen `all-MiniLM-L6-v2` -> 384d. `encode(card)` / `encode_many(cards)`.
- `DeckDataset`: WCC decks = positive samples, random pool cards = negative.
- `Constructor`: 4-copy hard cap (`MAX_COPIES=4`) except basic lands; `suggest()` uses ChromaDB; `score_card()`/`rank_cards()` use the trained model.
- `AICard.load_from_pcard` does `json.loads(pcard.json.lower())` -- all keys/values lowercased. `Constructor.can_add_copy` checks `"Basic" in type_text` (capital), so basic-land detection is effectively broken for production-loaded cards; tests pass only because they build `AICard` directly with capitalised strings. Fix the check to use `.lower()` comparisons if touching this code.
- `_get_scorer` caches model/encoder on the instance and loads `data/synergy_model.pt` relative to CWD -- run from project root.

## Conventions

- Imports: relative inside a package (`from .Card import AICard`); direct package-qualified across packages (`from database.mtgtools import Database`). Requires `pip install -e .`.
- Naming: PascalCase classes, snake_case funcs, `__double` private, UPPER_SNAKE constants.
- Type hints: space before colon (`self.pool : PCardList = ...`) -- project convention, don't "fix".
- No formatter configured; don't add one or reformat.
- Printing: `print(..., end="", flush=True)` with `\r` for progress.
- Comments may be English or German.

## Data flow / gotchas

- `Database()` defaults `force_update=False`; downloads ~515 MB from Scryfall on first run if ZODB empty.
- `import_collection.py` builds O(1) lookup indexes from `db.cards`; needs a `data/mtgcb-collection-*.csv`.
- Top-level `VectorDB.py` deletes and recreates the `MTGCards` collection each run; requires populated ZODB. ChromaDB 1.5.9 caps batch size at 5461, so `collection.add` is chunked (see script).
- `Constructor.suggest` returns `None` for empty pool; `score_card` returns `0.5` for empty deck; `rank_cards` returns input unchanged for empty deck.
- `Deck.update_deck`: action `4` adds (raises `ValueError` at maxsize), `3` pops last, `0` noop.
- `Constructor.this_card`: action `4` picks `pool[-1]`, action `3` drops `deck[-1]`.
- `CardEncoder` downloads `all-MiniLM-L6-v2` on first use (sentence-transformers).
- No `utils/` package anymore (removed in v2 refactor); `pyproject.toml` still lists `utils*` in packages.find (harmless).

## Tests

- pytest, no config file, flat `test/test_*.py`.
- Direct package-qualified imports in tests.
- Test count: 39 (7 deck + 17 constructor + 6 dataset + 5 classifier + 4 encoder).

## Active design decisions

See `docs/superpowers/specs/2026-06-04-card-synergy-model-v2.md`. Summary:

- Binary classifier `(deck_context, card) -> compatibility`.
- Phase A encoder: frozen ChromaDB `all-MiniLM-L6-v2` -> 384d (Phase C: trainable MLP projection, future).
- Deck context: mean-pool of card embeddings.
- Training data: WCC decks positive, random pool cards negative.
- Constructor: 4-copy cap (except basic lands), deck size >= 60.
- Target: CPU, <16 GB RAM.

Stale references removed in v2 refactor: `ai/MTGDeckBuilderModel.py`, `ai/Preparer.py`, `ai/Trainer.py`, `utils/`, `ai_prepare.py`, old `data/*.bin`. Do not recreate.
