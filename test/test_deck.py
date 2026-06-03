import pytest
from environment.Card import AICard
from environment.Deck import Deck


class TestDeck:
    def test_default_maxsize(self):
        deck = Deck()
        assert deck._Deck__maxsize == 60
        assert deck._Deck__minsize == 60

    def test_custom_maxsize(self):
        deck = Deck(maxsize=100)
        assert deck._Deck__maxsize == 100

    def test_add_card_within_limit(self):
        deck = Deck(maxsize=2)
        card = AICard({"name": "Test Card", "type_line": "Creature"})
        deck.update_deck(card=card, action=4)
        assert deck.size == 1

    def test_add_card_exceeds_maxsize_raises(self):
        deck = Deck(maxsize=1)
        card1 = AICard({"name": "Card 1", "type_line": "Creature"})
        card2 = AICard({"name": "Card 2", "type_line": "Instant"})
        deck.update_deck(card=card1, action=4)
        with pytest.raises(ValueError, match="max size"):
            deck.update_deck(card=card2, action=4)

    def test_remove_card(self):
        deck = Deck(maxsize=2)
        card = AICard({"name": "Test Card", "type_line": "Creature"})
        deck.update_deck(card=card, action=4)
        assert deck.size == 1
        deck.update_deck(action=3)
        assert deck.size == 0

    def test_action_noop(self):
        deck = Deck(maxsize=2)
        card = AICard({"name": "Test Card", "type_line": "Creature"})
        deck.update_deck(card=card, action=0)
        assert deck.size == 0

    def test_size_tracking_after_multiple_adds(self):
        deck = Deck(maxsize=5)
        for i in range(3):
            card = AICard({"name": f"Card {i}", "type_line": "Creature"})
            deck.update_deck(card=card, action=4)
        assert deck.size == 3
