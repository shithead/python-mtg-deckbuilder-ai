# Card Synergy Model v2

**Date:** 2026-06-04
**Status:** design-phase
**Supersedes:** implicit model in ai/MTGDeckBuilderModel, ai/Preparer, ai/Trainer

## Motivation

The current `MTGDeckBuilderModel` has a broken architecture:
- Input: one-hot of `pool_size` (card position, no card features)
- 40 growing hidden layers when `output > input` → 8.1B parameters
- Degenerate `MTGDataset.__getitem__` returns `(x, x)` — no learning signal
- Embedding layer lives in Preparer, not in the model

Goal: Learn card synergies from WCC decks on a consumer laptop (CPU, <16 GB RAM).

## Architecture (Phase A — MVP)

```
┌─────────────────────────────────────────────────────────┐
│ ChromaDB (frozen)                                        │
│   card → all-MiniLM-L6-v2 → 384d embedding               │
└───────────────────────┬─────────────────────────────────┘
                        ↓
Deck context: mean-pool of card embeddings for all cards
currently in the deck.

Classifier:
  input:  [card_embed (384d) ⊕ deck_embed (384d)]  = 768d
  hidden: Linear(768, 256) → ReLU → Dropout(0.2)
  output: Linear(256, 1) → Sigmoid
  params: ~200K (trainable on CPU in minutes)

Training:
  - Positive: card that IS in this WCC deck
  - Negative: card that is NOT in this deck (random sample from pool)
  - Loss: BCEWithLogitsLoss
  - Optimizer: Adam (lr=1e-3)
  - Batch: 64
```

### Phase C upgrade path

Replace frozen ChromaDB with trainable projection:
```
ChromaDB 384d → Linear(384, 128) → ReLU → Linear(128, 64)
                                                    ↓
                                    trainable card encoding (64d)
                                                    ↓
                                    classifier uses 64d embeddings
```

The card encoder interface is `(PCard) → torch.Tensor`, so phases are pluggable.

## Data Pipeline

```
ZODB (basic_collection + wcc_collection)
  → DeckDataset: for each WCC deck, produce (deck_context, candidate, label)
    - deck_context: mean of all card embeddings in deck (excluding candidate)
    - candidate: card embedding
    - label: 1 if candidate ∈ deck, 0 if candidate ∉ deck
  → DataLoader (shuffle, batch=64)
```

## Constructor Changes

4-copy hard constraint:
- Model scores all cards in user's pool against current deck context
- Cards sorted by score
- Constructor picks highest-scored cards, enforcing:
  - Max 4 copies per card (except basic lands)
   - Deck size ≥ 60 cards
  - Format legality (already handled by loadPool filter)

## What gets deleted

| File | Reason |
|------|--------|
| `ai/MTGDeckBuilderModel.py` | Replaced by synergy classifier |
| `ai/Preparer.py` | Vocab/embedding/dataset building no longer needed |
| `ai/Trainer.py` | Replaced by new Trainer_Synergy |
| `environment/Card.py` → AICard class | Simplify to a thin wrapper |
| `utils/utils.py` → tokenization helpers | No longer needed |
| `test/test_model.py` | Tests old model |
| `test/test_mtgdataset.py` | Tests old dataset |
| `test/test_trainer.py` | Tests old trainer |
| `data/word2idx.bin`, `data/embeddings.bin`, `data/datasets.bin` | Old pipeline artifacts |

## What stays

| File | Reason |
|------|--------|
| `database/mtgtools.py` | Database layer unchanged |
| `database/VectorDB.py` | VectorSearcher unchanged |
| `VectorDB.py` (top-level) | ChromaDB builder unchanged |
| `import_collection.py` | Import pipeline unchanged |
| `environment/Constructor.py` | Updated for new model + hard constraints |
| `environment/Deck.py` | Unchanged |
| `test/test_deck.py` | Unchanged |
| `test/test_constructor.py` | Updated for new model interface |
| `scrape_decklists.py` | Unchanged |

## New files

| File | Purpose |
|------|---------|
| `ai/CardEncoder.py` | ChromaDB-based card → embedding (Phase A), pluggable for Phase C |
| `ai/SynergyClassifier.py` | Binary classifier: (deck_ctx, card) → compat score |
| `ai/DeckDataset.py` | Builds training pairs from WCC decks |
| `ai/train.py` | Training script with configurable epochs/batch |
| `test/test_card_encoder.py` | CardEncoder tests |
| `test/test_synergy_classifier.py` | SynergyClassifier tests |
| `test/test_deck_dataset.py` | DeckDataset tests |

## Model sizes (target)

| Component | Parameters | Memory (fp32) |
|-----------|-----------|---------------|
| CardEncoder (Phase A) | 0 (frozen) | N/A |
| SynergyClassifier | ~200K | ~0.8 MB |
| Total | ~200K | <1 MB GPU / <4 MB CPU |

Training on 595 WCC decks (~35K positive samples + 35K negative):
~70K total samples × 200K params → trains in <5 min on a modern CPU.
