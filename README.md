# Python MTG Deckbuilder AI

AI-assisted Magic: The Gathering deck building. Uses PyTorch for neural models,
ChromaDB for semantic card search, and mtgtools for card data.

## Quick start

```bash
nix-shell                     # system deps + pip install
pip install -e .              # install the project in editable mode
```

Place a CSV collection export in deckbox format at `data/mtgcb-collection-*.csv`,
then run the import (first run downloads ~515 MB from Scryfall, one-time cost):

```bash
python import_collection.py   # import CSV + WCC decks into ZODB
python VectorDB.py            # build ChromaDB vector index
python ai/train.py            # train synergy classifier
# or one-shot:
./run_pipeline.sh             # clean rebuild from scratch (deletes generated files)
```

Alternatively, use `./create_virtenv.sh` instead of `nix-shell` for a plain venv.

## Project structure

```
├── ai/                         PyTorch model, dataset, training (Phase C)
│   ├── CardEncoder.py          frozen all-MiniLM-L6-v2 → 384d
│   ├── CardProjector.py        trainable projection 384 → 128 → 64
│   ├── DeckDataset.py          WCC deck training pairs
│   ├── SynergyClassifier.py    binary (deck, card) compatibility
│   └── train.py                training script
├── config/                     paths & constants
├── database/                   mtgtools (ZODB), ChromaDB vector search
├── environment/                Domain objects: AICard, Deck, Constructor
├── test/                       pytest (53 tests)
├── VectorDB.py                 ChromaDB index builder (top-level script)
├── import_collection.py         Import CSV + WCC decks into ZODB
├── build_deck.py               CLI: build a deck from query / themes
├── run_pipeline.sh             Clean rebuild from scratch
├── pyproject.toml              Package metadata
└── requirements.txt            Python dependencies
```

## Key components

### Deck building

```python
from environment.Card import AICard
from environment.Deck import Deck
from environment.Constructor import Constructor
from mtgtools.PCardList import PCardList

pool = PCardList()
# ... load cards into pool ...

deck = Deck(maxsize=60)
ctor = Constructor()

# Text-based search — find cards by description
card = ctor.suggest(pool, "creature with flying and lifelink")

# Check 4-copy limit (basic lands unlimited)
if ctor.can_add_copy(deck, card):
    ctor.add_card_to_deck(pool, deck, card)

# Manual pool navigation
card = ctor.other_card(pool, action=1)   # next card
card = ctor.other_card(pool, action=2)   # prev card

# Pick/drop cards between pool and deck
ctor.this_card(pool, deck, action=4)     # pick current card into deck
ctor.this_card(pool, deck, action=3)     # drop last card back to pool

# AI synergy scoring (requires trained model)
score = ctor.score_card(deck, card, alpha=0.6)       # → 0.0–1.0 (hybrid)
ranked = ctor.rank_cards(deck, card_pool, alpha=0.6)  # → [(card, score), ...] sorted
```

### Usage example: build a deck with AI

```bash
python build_deck.py 'aggressive red creature with haste'
# options:
python build_deck.py --colors 3 --theme 'High Alert' --theme 'Bedrock Tortoise' --query 'defender creatures'
python build_deck.py '3 Color Deck use High Alert as one theme card'
```

Or programmatically:

```python
from database.mtgtools import Database
from environment.Deck import Deck
from environment.Constructor import Constructor

db = Database()
pool = db.loadPool()
deck = Deck(maxsize=60)
ctor = Constructor()

# Step 1: Find a theme card
card = ctor.suggest(pool, "aggressive red creature with haste", n_results=10)
print(f"Picked: {card.name} ({card.type_line})")
ctor.add_card_to_deck(pool, deck, card)

# Step 2: Let the AI score your pool against the growing deck
for _ in range(30):
    ranked = ctor.rank_cards(deck, pool)
    best, score = ranked[0]
    if score < 0.5:
        break
    print(f"  + {best.name} ({score:.3f})")
    ctor.add_card_to_deck(pool, deck, best)

# Step 3: Fill with basic lands
for card in pool:
    if "Basic" in getattr(card, "type_line", "") or "Basic" in getattr(card, "type", ""):
        while len(deck) < 60 and ctor.can_add_copy(deck, card):
            ctor.add_card_to_deck(pool, deck, card)

print(f"\nDeck ({len(deck)} cards):")
for card in deck:
    print(f"  {card.name}")
```

### Model architecture (Phase C)

`CardProjector` + `SynergyClassifier` — trainable projection on top of frozen ChromaDB.

```
Card → ChromaDB (frozen all-MiniLM-L6-v2) → 384d embedding
                                                   ↓
                                   CardProjector: Linear(384,128) → ReLU → Linear(128,64)
                                                   ↓
                                            trainable 64d embedding
                                                   ↓
                              [deck_ctx ⊕ card_emb] → MLP(128, 256) → ReLU → Dropout → Linear(256, 1) → Sigmoid
```

- **~90K parameters** (CardProjector: ~57K + SynergyClassifier: ~33K)
- **Training data:** 595 WCC decks (~87K samples) with synthetic positives + hard negative mining
- **Loss:** Binary Cross-Entropy (projector + classifier trained jointly)
- **Training accuracy:** ~96.3% on hard negatives (top-200 similar cards per deck)
- **Early stopping:** Stops when accuracy ≥ 98% or improvement < 0.05% between epochs
- **Hybrid scoring (inference):** `score = alpha * model + (1 - alpha) * cosine_sim(deck_ctx, card_emb)` with optional `query_emb` anchor
- **Query anchor:** Cosine similarity to the original search query keeps the deck focused on the theme

```bash
python ai/train.py             # trains projector + classifier, saves both .pt files
```

### build_deck.py CLI

```bash
python build_deck.py "aggressive red creature with haste"
python build_deck.py --colors 2 --theme "Lightning Bolt" --query "burn"
python build_deck.py --colors 3 --theme "High Alert" --theme "Bedrock Tortoise" --theme-colors WUG
python build_deck.py "3 Color Deck use High Alert as one theme card"
```

Options:
- `--colors N` — number of colors to play (1–5).
- `--theme NAME` — theme card name (repeatable, up to 3).
- `--theme-colors LETTERS` — override theme colors, e.g. `WUG` or `W,U,G`.
- `--query TEXT` — semantic query text (anchoring).
- Free text may include a color count and theme card names.

## Notes

- `Database()` defaults to `force_update=False`; first run downloads ~515 MB from Scryfall if ZODB is empty. Scryfall requires a custom `User-Agent` (set in `database/mtgtools.py`).
- `Constructor.suggest` returns `None` for an empty pool; `score_card` returns `0.5` for an empty deck; `rank_cards` returns the input unchanged.
- `CardEncoder` downloads `all-MiniLM-L6-v2` on first use (sentence-transformers).
- ChromaDB 1.5.9 caps batch size at 5461; `VectorDB.py` chunks `collection.add` accordingly.
- Run training/scoring from the project root (model paths are relative to CWD).
- `build_deck.py` builds a 60-card deck from your collection only; theme cards are inserted as many times as available (max 4). Use `--theme` (repeatable, up to 3), `--colors N`, `--theme-colors WUG`, `--query`.

## Development

### Run tests

```bash
pytest test/                     # all 53 tests
pytest test/ -k "test_deck"      # single test suite
pytest test/ -v --tb=short       # verbose with short tracebacks
coverage run -m pytest test/     # with coverage
```

### Run a single test

```bash
pytest test/test_deck.py::TestDeck::test_add_card_within_limit
pytest test/ -k "test_forward"
```

### Code conventions

See `AGENTS.md` for import style, naming, type hints, and error handling conventions.
