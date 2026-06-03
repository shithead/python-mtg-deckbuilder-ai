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
python ai_prepare.py          # create vocab, embeddings, datasets

python -c "
from ai.Trainer import Trainer_T1
from ai.MTGDeckBuilderModel import MTGDeckBuilderModel

trainer = Trainer_T1(MTGDeckBuilderModel)
trainer.run(epochs=10)
"                              # train the model
```

Alternatively, use `./create_virtenv.sh` instead of `nix-shell` for a plain venv.

## Project structure

```
├── ai/                         PyTorch model, data preparation, training
├── database/                   mtgtools (ZODB), ChromaDB vector search
├── environment/                Domain objects: AICard, Deck, Constructor
├── utils/                      Tokenization helpers
├── test/                       Tests (pytest, 53 tests)
├── VectorDB.py                 ChromaDB index builder script
├── import_collection.py        Import CSV + WCC decks into ZODB
├── ai_prepare.py               Prepare AI data (vocab, datasets)
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

# Manual navigation
card = ctor.other_card(pool, action=1)   # next card
card = ctor.other_card(pool, action=2)   # prev card

# Build the deck
ctor.this_card(pool, deck, action=4)     # pick card into deck
ctor.this_card(pool, deck, action=3)     # drop card back to pool
```

### Model architecture

The neural model (`MTGDeckBuilderModel`) predicts card relevance scores.
Each card is represented as a one-hot vector of size `pool_size`, with
configurable hidden layers and an output layer of size `4 * pool_size`.

Training uses cross-entropy loss with the datasets prepared from the card pool.

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
