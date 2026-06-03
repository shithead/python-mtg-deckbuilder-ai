# AGENTS.md — python-mtg-deckbuilder-ai

## Project overview

Python project for AI-assisted Magic: The Gathering deck building. Uses PyTorch for neural models and ChromaDB for vector embeddings. Card data comes from the external `mtgtools` library backed by a ZODB database.

```
├── ai/                    PyTorch model, data preparation, training loop
├── database/              Database abstraction (mtgtools, ChromaDB vector search)
├── environment/           Domain objects: AICard, Deck, Constructor
├── utils/                 Tokenization helpers
├── test/                  Tests (pytest)
├── data/                  Persistent data, cache files, CSV/DB files
├── VectorDB.py            ChromaDB vector store builder (top-level script)
├── import_collection.py   Import CSV + WCC decks into ZODB
├── ai_prepare.py          Prepare AI data (vocab, embeddings, datasets)
├── pyproject.toml         Package metadata and build configuration
└── requirements.txt       Python dependencies
```

## Setup

```bash
./create_virtenv.sh          # creates ./_build virtualenv with Python 3
nix-shell                    # alternative: Nix shell with all system deps + pip install
pip install -r requirements.txt
pip install -e .             # install the project in editable mode
```

`requirements.txt` pins `torch==2.3.0`, `torchtext==0.18.0`. The Nix `shell.nix` provides additional system deps (chromadb, sentence-transformers, gcc, pkg-config, zlib) and runs `pip install -r requirements.txt` automatically in its shellHook.

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

## Code style

### Imports

- **Within a package** (`environment/__init__.py`): use relative imports:
  ```python
  from .Card import AICard
  from .Deck import Deck
  ```

- **Across packages** (e.g. `ai/Preparer.py` importing `environment` and `database`): use direct package-qualified imports. The `pyproject.toml` + `pip install -e .` makes all packages importable without path hacks:
  ```python
  from database.mtgtools import Database
  from environment.Card import AICard
  from utils.utils import get_token
  ```

- **Top-level scripts** (`VectorDB.py`, `import_collection.py`) also use direct imports — `pip install -e .` places the project root on `sys.path`.

- **Standard library first**, then third-party, then local imports. Group related imports together. Blank lines between import groups are common but not strict.

### Naming

| Kind | Convention | Example |
|------|-----------|---------|
| Classes | PascalCase | `AICard`, `MTGDeckBuilderModel`, `Preparer`, `Trainer_T1` |
| Methods / functions | snake_case | `create_word2idx()`, `get_token()`, `update_deck()` |
| Private members | double underscore name mangling | `self.__basic_path`, `self.__pos_pool` |
| Module-level constants | UPPER_SNAKE_CASE | `CHROMA_DATA_PATH`, `EMBED_MODEL`, `COLLECTION_NAME` |
| Module-level variables | lowercase | `device`, `client`, `embedding_func` |
| File names | PascalCase for single-class modules, snake_case for utils/multi-purpose modules | `Card.py` (class files reside in packages), `utils/utils.py` |

Note: `ai/Trainer.py` and `ai/Preparer.py` set `torch.set_default_dtype(torch.float16)` at module level — globally uses half-precision floats. Be aware of this when adding new tensors.

### Type hints

Type hints exist on some constructor parameters and attributes but are not required. When adding them, use the existing style with a **space before the colon** (this is the project convention, not a PEP 8 violation to be "fixed"):

```python
self.pool : PCardList = ...
self.input_layer_size : int = 0
self.keys : set = set()
self.model : MTGDeckBuilderModel = None
```

Add return type annotations for public methods when practical. Use `@property` decorator for getter-style access (see `Trainer_T1.Model` and `Preparer.Model`).

### Error handling

- Avoid bare `except: pass`. At minimum catch specific exceptions (e.g. `FileNotFoundError`) and print the error.
- Raise proper exceptions — never `return Exception("message")` (see `database/mtgtools.py:55-56`). Use `raise ValueError(...)` or `raise RuntimeError(...)`.
- Use `pytest.raises` for testing exception paths.
- `try`/`except FileNotFoundError: print(e); pass` is an accepted pattern for loading optional cached files (see `Preparer.load()` and `Trainer_T1.load()`).

### Docstrings

Docstrings exist on some public classes and methods but are sparse. When adding:
- Use triple-quoted strings immediately after `def`/`class`
- Describe purpose, parameters, and return values briefly
- Follow the existing informal style — no strict Google/NumPy/Sphinx format required

### Logging / printing

The project uses `print()` with `end="", flush=True` for progress indicators. There is no logging framework. Keep this pattern for consistency; use carriage-return `\r` for in-place progress updates during long loops. Use `\n` in the same `print()` to finalize a line after a loop.

### Comments

- Comments can be in English or German (mixed usage in existing code)
- Avoid commenting out large blocks of code — delete them instead
- Use `# TODO description` for pending work
- Use `# XXX comment` to mark known hacks or temporary workarounds

### Formatting

No formatter is configured. Do not add one or reformat the entire codebase without discussion.

## Tests

- Framework: **pytest** (no config file)
- Test files: flat under `test/`, named `test_*.py`
- No fixtures, no conftest.py, no parametrize in use yet — add these if helpful
- Tests import the module under test using direct package-qualified imports (same as main code)

## Known issues

- No known issues currently.

## External dependencies

- **mtgtools** — MTG card data library providing `PCard`, `PCardList`, `MtgDB`. Backbone for all card data. Docs: https://github.com/shithead/mtgtools
- **chromadb** — Vector database for semantic card search
- **sentence-transformers** — Text embeddings (model: `all-MiniLM-L6-v2`)
- **pandas** — DataFrames for dataset construction
- **torch / torchtext** — PyTorch neural network and text tokenizers (`torch==2.3.0`, `torchtext==0.18.0`)
- **torch-summary** — Model architecture summary via `torchsummary.summary()`
- **ZODB** (`persistent`) — Object database backing `mtgtools` data
- **coverage / pytest** — Test tooling (`requirements.txt`, dev only)
