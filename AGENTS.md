# AGENTS.md — python-mtg-deckbuilder-ai

## Project overview

Python project for AI-assisted Magic: The Gathering deck building. Uses PyTorch for neural models and ChromaDB for vector embeddings. Card data comes from the external `mtgtools` library backed by a ZODB database.

```
├── ai/                    PyTorch model, data preparation, training loop
├── database/              Database abstraction (mtgtools + legacy SQLite)
├── environment/           Domain objects: AICard, Deck, Constructor
├── utils/                 Tokenization helpers
├── test/                  Tests (pytest)
├── data/                  Persistent data, cache files, CSV/DB files
├── VectorDB.py            ChromaDB vector store builder (top-level script)
├── import_collection.py   Import CSV + WCC decks into ZODB
├── ai_prepare.py          Prepare AI data (vocab, embeddings, datasets)
└── requirements.txt       Python dependencies
```

## Setup

```bash
./create_virtenv.sh          # creates ./_build virtualenv with Python 3
nix-shell                    # alternative: Nix shell with all deps
pip install -r requirements.txt
```

## Commands

### Run all tests
```bash
pytest test/
# or
python -m pytest test/
```

### Run a single test
```bash
pytest test/test_pool.py::test_initPool
# or with -k for substring match:
pytest test/ -k "test_initPool"
```

### Run tests with coverage
```bash
coverage run -m pytest test/
coverage html --omit="test/*,_build/*,/nix/*"
# or use the script:
./coverage.sh
```

### Run a top-level script
```bash
python import_collection.py
python ai_prepare.py
python VectorDB.py
```

Note: `requirements.txt` pins specific versions (torch 2.3.0, torchtext 0.18.0). The Nix `shell.nix` provides additional system dependencies (chromadb, sentence-transformers, gcc, pkg-config, zlib).

## Code style

### Imports

- **At the package level** (`environment/__init__.py`): use relative imports:
  ```python
  from .Card import AICard
  from .Deck import Deck
  ```

- **Across packages**: use `sys.path` hacks (existing convention in this codebase):
  ```python
  import sys, os
  sys.path.append(os.path.abspath('../environment'))
  sys.path.append(os.path.abspath('../database'))
  from database.mtgtools import Database
  from environment.Card import AICard
  ```

- **Standard library before third-party**, followed by local imports. Group related imports together.

### Naming

| Kind | Convention | Example |
|------|-----------|---------|
| Classes | PascalCase | `AICard`, `MTGDeckBuilderModel`, `Preparer` |
| Methods / functions | snake_case | `create_word2idx()`, `get_token()`, `update_deck()` |
| Private members | double underscore name mangling | `self.__basic_path`, `self.__pos_pool` |
| Module-level constants | UPPER_SNAKE_CASE | `CHROMA_DATA_PATH`, `EMBED_MODEL`, `COLLECTION_NAME` |
| Module-level variables | lowercase | `device`, `client`, `embedding_func` |
| File names | PascalCase for classes, snake_case for utils | `AICard.py` (in `environment/Card.py`), `utils/utils.py` |

### Type hints

Type hints exist on some constructor parameters and attributes but are not required. When adding them, use the existing style with a **space before the colon** (this is the project convention, not a PEP 8 violation to be "fixed"):

```python
self.pool : PCardList = ...
self.input_layer_size : int = 0
self.keys : set = set()
```

Add return type annotations for public methods when practical.

### Error handling

- Avoid bare `except: pass`. At minimum catch specific exceptions and log the error.
- Raise proper exceptions — never `return Exception("message")` (see `database/mtgtools.py:55-56`). Use `raise ValueError(...)` or `raise RuntimeError(...)`.
- Use `pytest.raises` for testing exception paths, as done in `test/test_pool.py`.

### Docstrings

Docstrings exist on some public classes and methods but are sparse. When adding:
- Use triple-quoted strings immediately after `def`/`class`
- Describe purpose, parameters, and return values briefly
- Follow the existing informal style rather than strict Google/NumPy/Sphinx

### Logging / printing

The project uses `print()` with `end="", flush=True` for progress indicators. There is no logging framework. Keep this pattern for consistency; use carriage-return `\r` for in-place progress updates during long loops.

### Comments

- Comments can be in English or German (mixed usage in existing code)
- Avoid commenting out large blocks of code — delete them instead
- Use `# TODO description` for pending work

### Formatting

No formatter is configured. Run `black` or `isort` if you add a `pyproject.toml`, but do not reformat the entire codebase without discussion.

## Tests

- Framework: **pytest** (no config file)
- Test files: flat under `test/`, named `test_*.py`
- No fixtures, no conftest.py, no parametrize in use yet — add these if helpful
- Tests import the module under test using `sys.path` hacks, matching main code convention
- The existing test suite (`test/test_pool.py`) imports `environment.Pool` which has been removed. New tests should target existing modules: `AICard`, `Deck`, `Constructor`, `Database`, `Preparer`, `Trainer_T1`

## Known issues

- `ai/Trainer.py` — `train_loop()` and `test_loop()` are missing `self` in their signatures; they won't work as instance methods
- `database/mtgtools.py` — `loadPool()` and `loadWccPool()` return Exception objects instead of raising them
- `database/SQLite.py` — imports `environment.Pool` and old card classes that no longer exist; this file is defunct
- `environment/Constructor.py:42` — `del pool.currnet` is a typo for `pool.current`
- `VectorDB.py:13` — `get_toke@qa` is a typo (presumably meant `get_token`)
- `environment/Card.py` — `MTGDataset.__getitem__` uses `torch` and `np` without importing them
- Project has no `.gitignore` — add one for `__pycache__/`, `_build/`, `*.pyc`, `.fs`, `*.sqlite3`, ChromaDB binary files

## External dependencies

- **mtgtools** — Magic: The Gathering card data library providing `PCard`, `PCardList`, `MtgDB`. This is the backbone for all card data operations. Documentation: https://github.com/shithead/mtgtools
- **chromadb** — Vector database for semantic card search
- **sentence-transformers** — Text embeddings (model: `all-MiniLM-L6-v2`)
- **pandas** — DataFrames for dataset construction
- **torch / torchtext** — PyTorch neural network model and text tokenizers
- **ZODB** (`persistent`) — Object database backing `mtgtools` data
