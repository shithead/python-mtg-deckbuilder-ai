import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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


def train(epochs: int = 25, batch_size: int = 64, lr: float = 1e-3):
    print("Loading database...")
    db = Database(check_update=False)
    print("Loading encoder...")
    encoder = CardEncoder()
    print("Creating projector...")
    projector = CardProjector().to(device)
    print("Building dataset...")
    dataset = DeckDataset(db, encoder, num_negatives=1, num_synthetic=2, hard_negatives=200)
    print(f"  {len(dataset)} samples ({len(db.root.wcc_decks)} decks)")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SynergyClassifier(card_embed_dim=64).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(projector.parameters()), lr=lr
    )

    size = len(dataset)
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

        epoch_loss = running_loss / size
        epoch_acc = correct / size
        print(f"Epoch {epoch+1}/{epochs}  loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}")

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


if __name__ == "__main__":
    train()
