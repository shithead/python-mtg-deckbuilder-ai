import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from database.mtgtools import Database
from ai.CardEncoder import CardEncoder
from ai.CardProjector import CardProjector
from ai.DeckDataset import DeckDataset
from ai.SynergyClassifier import SynergyClassifier

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class _Subset(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


def _per_format_accuracy(model, projector, dataset, sample_indices, device):
    model.eval()
    projector.eval()
    stats = {}
    with torch.no_grad():
        for i in sample_indices:
            x, y = dataset[i]
            fmt = dataset._deck_formats[dataset._samples[i][0]] or "unknown"
            X = x.unsqueeze(0).to(device)
            deck_raw, card_raw = X[:, :384], X[:, 384:]
            deck_proj = projector(deck_raw)
            card_proj = projector(card_raw)
            Xp = torch.cat([deck_proj, card_proj], dim=1)
            pred = model(Xp).item()
            pred_label = 1.0 if pred > 0.5 else 0.0
            correct = 1.0 if pred_label == y.item() else 0.0
            s = stats.setdefault(fmt, {"correct": 0.0, "total": 0})
            s["correct"] += correct
            s["total"] += 1
    return {k: (v["correct"] / v["total"] if v["total"] else 0.0) for k, v in stats.items()}


def train(epochs: int = 25, batch_size: int = 64, lr: float = 1e-3,
          num_negatives: int = 1, num_synthetic: int = 2, hard_negatives: int = 200,
          sources=None, formats=None, balance: bool = True, val_split: float = 0.2,
          seed: int = 42):
    print("Loading database...")
    db = Database(check_update=False)
    print("Loading encoder...")
    encoder = CardEncoder()
    print("Creating projector...")
    projector = CardProjector().to(device)
    print("Building dataset...")
    dataset = DeckDataset(
        db, encoder, num_negatives=num_negatives, num_synthetic=num_synthetic,
        hard_negatives=hard_negatives, sources=sources, formats=formats, balance=balance,
    )
    print(f"  {len(dataset)} samples ({len(dataset._decks)} decks)")

    # Deck-based split (no card leakage)
    random.seed(seed)
    all_deck_ids = list(range(len(dataset._decks)))
    random.shuffle(all_deck_ids)
    n_val = int(len(all_deck_ids) * val_split) if val_split and len(all_deck_ids) > 1 else 0
    n_val = max(0, min(n_val, len(all_deck_ids) - 1)) if len(all_deck_ids) > 1 else 0
    val_deck_ids = set(all_deck_ids[:n_val])
    train_idx = [i for i, s in enumerate(dataset._samples) if s[0] not in val_deck_ids]
    val_idx = [i for i, s in enumerate(dataset._samples) if s[0] in val_deck_ids]
    print(f"  train samples: {len(train_idx)}, val samples: {len(val_idx)}")

    dataloader = DataLoader(_Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)

    model = SynergyClassifier(card_embed_dim=64).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(projector.parameters()), lr=lr
    )

    size = len(train_idx)
    prev_acc = 0.0
    for epoch in range(epochs):
        model.train()
        projector.train()
        running_loss = 0.0
        correct = 0
        for batch, (X_raw, y) in enumerate(dataloader):
            X_raw, y = X_raw.to(device), y.to(device)
            deck_raw, card_raw = X_raw[:, :384], X_raw[:, 384:]
            deck_proj = projector(deck_raw)
            card_proj = projector(card_raw)
            X = torch.cat([deck_proj, card_proj], dim=1)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item() * X.size(0)
            correct += ((pred > 0.5).float() == y).float().sum().item()

            if batch % 100 == 0:
                current = (batch + 1) * len(X)
                print(f"  loss: {loss.item():.4f}  [{current}/{size}]")

        epoch_loss = running_loss / max(size, 1)
        epoch_acc = correct / max(size, 1)
        val_acc = 0.0
        if val_idx:
            vcorrect = 0
            with torch.no_grad():
                for i in val_idx:
                    x, y = dataset[i]
                    X = x.unsqueeze(0).to(device)
                    deck_raw, card_raw = X[:, :384], X[:, 384:]
                    deck_proj = projector(deck_raw)
                    card_proj = projector(card_raw)
                    Xp = torch.cat([deck_proj, card_proj], dim=1)
                    pred = model(Xp)
                    if ((pred > 0.5).float() == y.to(device)).float().item():
                        vcorrect += 1
            val_acc = vcorrect / len(val_idx)
        fmt_acc = _per_format_accuracy(model, projector, dataset, val_idx, device) if val_idx else {}
        print(f"Epoch {epoch+1}/{epochs}  loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}  val_acc: {val_acc:.4f}")
        if fmt_acc:
            print("  per-format val acc:", "  ".join(f"{k}={v:.3f}" for k, v in sorted(fmt_acc.items())))

        if epoch_acc >= 0.98:
            print("Accuracy >= 98%, stopping early")
            break
        if epoch > 0 and (epoch_acc - prev_acc) < 0.0005:
            print(f"Improvement < 0.05% ({epoch_acc - prev_acc:.4f}), stopping early")
            break

        prev_acc = epoch_acc

    torch.save(model.state_dict(), "data/synergy_model.pt")
    print("Saved data/synergy_model.pt")
    torch.save(projector.state_dict(), "data/card_projector.pt")
    print("Saved data/card_projector.pt")


def _parse_list(s):
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-negatives", type=int, default=1)
    parser.add_argument("--num-synthetic", type=int, default=2)
    parser.add_argument("--hard-negatives", type=int, default=200)
    parser.add_argument("--sources", type=str, default=None)
    parser.add_argument("--formats", type=str, default=None)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_negatives=args.num_negatives,
        num_synthetic=args.num_synthetic,
        hard_negatives=args.hard_negatives,
        sources=_parse_list(args.sources),
        formats=_parse_list(args.formats),
        balance=not args.no_balance,
        val_split=args.val_split,
    )