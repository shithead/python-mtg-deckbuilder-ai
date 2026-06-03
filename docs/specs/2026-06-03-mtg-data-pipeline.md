# MTG Data Pipeline & Training Spec

**Date:** 2026-06-03
**Status:** implementation-phase

## Architecture overview

```
Scryfall API → ZODB (mtgdb.fs)   mtgdb.fs cached, force_update=False by default
     ↓
┌────┴──────────────────────────────────────────┐
│  import_collection.py                          │
│    myCollection.csv → basic_collection (ZODB)  │
│    WCC deck files → wcc_collection (ZODB)      │
└────────────────────┬──────────────────────────┘
                     ↓
    loadPool() + loadWccPool() → AICard[] pool
                     ↓
    ┌────────────────┼────────────────┐
    ↓                ↓                ↓
 VectorDB.py    ai_prepare.py    ai_train.py
 ChromaDB        vocab/datasets   Training loop
 (semantic       + model init     (DataLoader +
  card search)                    47 hidden x)
    ↓                               ↓
 Constructor.suggest()       MTGDeckBuilderModel
 "creature with flying"      card relevance scores
```

## Data sources

| Source | Format | Purpose | Status |
|--------|--------|---------|--------|
| Scryfall bulk | JSON via mtgtools → ZODB | Master card database | ✓ cached |
| `myCollection.csv` | deckbox CSV | User's owned cards | ✓ `import_collection.py` |
| WCC deck files | `.txt` (1 card/line) | Training targets | ✓ manual |
| magic.gg/decklists | HTML pages | Additional WCC data | ✗ pending |

## Completed (this session)

1. **Bugfixes** — 8 bugs fixed (Trainer self, forward(), Exception→raise, typos, Deck/Constructor action sync, etc.)
2. **Package structure** — `pyproject.toml` + `pip install -e .`, sys.path hacks removed
3. **Tests** — 53 tests in 5 files (Deck, Constructor, Model, Dataset, Trainer)
4. **Trainer overhaul** — DataLoader integration, `create_dataloader()`, `run()` orchestration
5. **ChromaDB integration** — `database/VectorDB.py` → `Constructor.suggest(pool, query_text)`
6. **CI** — GitHub Actions with nix-shell pytest (Python 3.12, nixos-24.11)
7. **Scryfall caching** — `Database(force_update=False)` skips download when ZODB populated
8. **PCardList API fix** — `add→append`, `size→len()`, `current→[-1]`, `del→pop(-1)`
9. **Scripts** — `ai_train.py` for convenience training launch

## Pending

### Scrape WCC decklists from magic.gg

Magic.gg hosts decklists at `https://magic.gg/decklists`. We want:
- **Standard Ranked** decklists
- **Historic Decklists**

Option A — Auto-scrape (new deps: requests, beautifulsoup4)
Option B — Manual download as `.txt` files into `data/magic_WCC_decks/`

**Decision:** pending (next session)

### Training with real data

Once WCC decklists are available:
1. Run `import_collection.py` with user's CSV + WCC decks
2. Run `VectorDB.py` to build ChromaDB index
3. Run `ai_prepare.py` to create vocab/datasets/model
4. Run `ai_train.py` to train on deck combinations
5. Use `Constructor.suggest()` + trained model to build optimal deck from user's collection

## Key files

```
database/mtgtools.py     # Database class, ZODB-backed
import_collection.py      # CSV + WCC → ZODB
ai_prepare.py             # vocab, word2idx, datasets, model init
ai_train.py               # training loop
VectorDB.py               # ChromaDB build + query
database/VectorDB.py      # VectorSearcher module
environment/Constructor.py # Deck building with suggest()
ai/Trainer.py             # Trainer_T1 with DataLoader + run()
```

## Test results

```
53 passed, 0 failed in 2.92s
```

Test files: `test_deck.py`, `test_constructor.py`, `test_model.py`, `test_mtgdataset.py`, `test_trainer.py`
