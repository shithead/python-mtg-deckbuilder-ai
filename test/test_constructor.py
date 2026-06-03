import pytest
from environment.Card import AICard
from environment.Deck import Deck
from environment.Constructor import Constructor
from mtgtools.PCardList import PCardList


class TestConstructor:
    def test_other_card_next(self):
        pool = PCardList()
        card1 = AICard({"name": "Card 1", "type_line": "Creature"})
        card2 = AICard({"name": "Card 2", "type_line": "Creature"})
        pool.add(card1)
        pool.add(card2)
        ctor = Constructor()
        assert ctor.other_card(pool, action=1) == pool[0]
        assert ctor.other_card(pool, action=1) == pool[1]

    def test_other_card_no_action(self):
        pool = PCardList()
        card = AICard({"name": "Card 1", "type_line": "Creature"})
        pool.add(card)
        ctor = Constructor()
        result = ctor.other_card(pool, action=0)
        assert result is pool[0]

    def test_pick_card_from_pool_to_deck(self):
        pool = PCardList()
        card = AICard({"name": "Pick Me", "type_line": "Creature"})
        pool.add(card)
        deck = Deck(maxsize=5)
        ctor = Constructor()
        ctor.this_card(pool, deck, action=4)
        assert deck.size == 1
        assert pool.size == 0

    def test_drop_card_from_deck_to_pool(self):
        pool = PCardList()
        card = AICard({"name": "Drop Me", "type_line": "Creature"})
        deck = Deck(maxsize=5)
        deck.update_deck(card=card, action=4)
        assert deck.size == 1
        ctor = Constructor()
        ctor.this_card(pool, deck, action=3)
        assert deck.size == 0
        assert pool.size == 1

    def test_this_card_no_action(self):
        pool = PCardList()
        card = AICard({"name": "Card", "type_line": "Creature"})
        pool.add(card)
        deck = Deck(maxsize=5)
        ctor = Constructor()
        ctor.this_card(pool, deck, action=0)
        assert deck.size == 0
        assert pool.size == 1
