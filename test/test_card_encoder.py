import pytest
import torch
from ai.CardEncoder import CardEncoder

class TestCardEncoder:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.encoder = CardEncoder(model_name="all-MiniLM-L6-v2")

    def test_encode_returns_tensor(self):
        class MockCard:
            name = "Lightning Bolt"
            type_line = "Instant"
            mana_cost = "{R}"
            oracle_text = "Lightning Bolt deals 3 damage to any target."

        card = MockCard()
        emb = self.encoder.encode(card)
        assert isinstance(emb, torch.Tensor)
        assert emb.shape == (384,)

    def test_encode_many_returns_batch(self):
        class MockCard:
            def __init__(self, name):
                self.name = name
                self.type_line = "Creature"
                self.mana_cost = "{2}{G}"
                self.oracle_text = "Trample"

        cards = [MockCard("Grizzly Bears"), MockCard("Hill Giant")]
        embs = self.encoder.encode_many(cards)
        assert embs.shape == (2, 384)

    def test_encode_no_grad(self):
        class MockCard:
            name = "Counterspell"
            type_line = "Instant"
            mana_cost = "{U}{U}"
            oracle_text = "Counter target spell."

        card = MockCard()
        emb = self.encoder.encode(card)
        assert not emb.requires_grad

    def test_encode_many_deterministic(self):
        class MockCard:
            def __init__(self, name):
                self.name = name
                self.type_line = "Sorcery"
                self.mana_cost = "{B}"
                self.oracle_text = "Destroy target creature."

        cards = [MockCard("Doom Blade")]
        e1 = self.encoder.encode_many(cards)
        e2 = self.encoder.encode_many(cards)
        assert torch.allclose(e1, e2)
