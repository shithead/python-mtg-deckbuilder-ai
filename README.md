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
```

Alternatively, use `./create_virtenv.sh` instead of `nix-shell` for a plain venv.

## Project structure

```
├── ai/                         PyTorch model, dataset, training
├── database/                   mtgtools (ZODB), ChromaDB vector search
├── environment/                Domain objects: AICard, Deck, Constructor
├── test/                       Tests (pytest, 39 tests)
├── VectorDB.py                 ChromaDB index builder script
├── import_collection.py        Import CSV + WCC decks into ZODB
├── build_deck.py               CLI: build a deck from a search query
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
score = ctor.score_card(deck, card)          # → 0.0–1.0
ranked = ctor.rank_cards(deck, card_pool)    # → [(card, score), ...] sorted
```

### Usage example: build a deck with AI

```bash
python build_deck.py 'aggressive red creature with haste'
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

### Model architecture

`SynergyClassifier` — binary classifier predicting card-to-deck compatibility.

```
Card → ChromaDB (frozen all-MiniLM-L6-v2) → 384d embedding
Deck context → mean-pool of all card embeddings

[deck_ctx ⊕ card_emb] → MLP(768, 256) → ReLU → Dropout → Linear(256, 1) → Sigmoid
```

- **~200K parameters** (trainable on CPU)
- **Training data:** 595 WCC decks (~86K samples, positive + negative)
- **Loss:** Binary Cross-Entropy
- **Training accuracy:** ~98% after ~16 epochs
- **Early stopping:** Stops when accuracy ≥ 98% or improvement < 0.05% between epochs

```bash
python ai/train.py             # trains with early stopping, saves data/synergy_model.pt
```

## Development

### Run tests

```bash
pytest test/                     # all 40 tests
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
