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
python ai/train.py            # train synergy classifier (~30s on CPU)
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

# Semantic search — find cards by description
card = ctor.suggest(pool, "creature with flying and lifelink")

# Check 4-copy limit (basic lands unlimited)
ctor.can_add_copy(deck, card)

# Manual navigation
card = ctor.other_card(pool, action=1)   # next card
card = ctor.other_card(pool, action=2)   # prev card

# Build the deck
ctor.this_card(pool, deck, action=4)     # pick card into deck
ctor.this_card(pool, deck, action=3)     # drop card back to pool
```

### Model architecture

`SynergyClassifier` — binary classifier predicting card-to-deck compatibility.

```
Card → ChromaDB (frozen all-MiniLM-L6-v2) → 384d embedding
Deck context → mean-pool of all card embeddings

[deck_ctx ⊕ card_emb] → MLP(768, 256) → ReLU → Dropout → Linear(256, 1) → Sigmoid
```

- **~200K parameters** (trainable on CPU in ~30s)
- **Training data:** WCC decks = positive, random pool cards = negative
- **Loss:** Binary Cross-Entropy

```bash
python ai/train.py             # trains 25 epochs, saves data/synergy_model.pt
```

## Development

### Run tests

```bash
pytest test/                     # all 39 tests
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
