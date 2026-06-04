import pytest
import torch
from unittest.mock import MagicMock
from ai.DeckDataset import DeckDataset


class MockCard:
    def __init__(self, name, type_line="Creature", mana_cost="{1}", oracle_text=""):
        self.name = name
        self.type_line = type_line
        self.mana_cost = mana_cost
        self.oracle_text = oracle_text


class TestDeckDataset:
    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        deck1 = [MockCard("Card A"), MockCard("Card B"), MockCard("Card C")]
        deck2 = [MockCard("Card A"), MockCard("Card D")]
        db.root.wcc_decks = [deck1, deck2]

        pool_cards = [MockCard("Card A"), MockCard("Card B"),
                      MockCard("Card C"), MockCard("Card D")]
        db.loadPool.return_value.unique_names.return_value = pool_cards
        db.loadWccPool.return_value.unique_names.return_value = []
        return db

    @pytest.fixture
    def mock_encoder(self):
        enc = MagicMock()
        def encode_many_side_effect(cards):
            out = torch.zeros(len(cards), 384)
            for i, c in enumerate(cards):
                out[i, 0] = float(ord(c.name[-1]))
            return out
        enc.encode_many.side_effect = encode_many_side_effect
        return enc

    def test_len_positive_and_negative(self, mock_db, mock_encoder):
        dataset = DeckDataset(mock_db, mock_encoder, num_negatives=1)
        assert len(dataset) == 10

    def test_getitem_positive_returns_label_1(self, mock_db, mock_encoder):
        dataset = DeckDataset(mock_db, mock_encoder, num_negatives=1)
        x, y = dataset[0]
        assert x.shape == (768,)
        assert y.item() == 1.0

    def test_getitem_negative_returns_label_0(self, mock_db, mock_encoder):
        dataset = DeckDataset(mock_db, mock_encoder, num_negatives=1)
        x, y = dataset[3]
        assert x.shape == (768,)
        assert y.item() == 0.0

    def test_negative_card_not_in_deck(self, mock_db, mock_encoder):
        dataset = DeckDataset(mock_db, mock_encoder, num_negatives=1)
        deck1_neg_indices = [1, 3, 5]
        for i in deck1_neg_indices:
            x, y = dataset[i]
            assert y.item() == 0.0

    def test_num_negatives_respected(self, mock_db, mock_encoder):
        dataset = DeckDataset(mock_db, mock_encoder, num_negatives=2)
        assert len(dataset) == 15

    def test_deck_context_excludes_candidate(self, mock_db, mock_encoder):
        dataset = DeckDataset(mock_db, mock_encoder, num_negatives=0)
        x, y = dataset[0]
        assert abs(x[0].item() - 66.5) < 0.1
        assert abs(x[384].item() - 65.0) < 0.1
