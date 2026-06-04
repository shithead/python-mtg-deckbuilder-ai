import pytest
from unittest.mock import MagicMock, patch
from environment.Card import AICard
from environment.Deck import Deck
from environment.Constructor import Constructor
from mtgtools.PCardList import PCardList


class TestConstructorSuggest:
    def test_suggest_returns_card_from_pool(self):
        pool = PCardList()
        card1 = AICard({"name": "Lightning Bolt", "type_line": "Instant"})
        card2 = AICard({"name": "Grizzly Bears", "type_line": "Creature"})
        pool.append(card1)
        pool.append(card2)
        ctor = Constructor()
        with (
            patch("environment.Constructor.Constructor._get_searcher") as mock_searcher,
        ):
            mock_instance = MagicMock()
            mock_instance.suggest_indices.return_value = [0]
            mock_searcher.return_value = mock_instance
            result = ctor.suggest(pool, "damage spell", n_results=1)
        assert result is card1

    def test_suggest_returns_none_for_empty_pool(self):
        pool = PCardList()
        ctor = Constructor()
        with (
            patch("environment.Constructor.Constructor._get_searcher") as mock_searcher,
        ):
            mock_instance = MagicMock()
            mock_searcher.return_value = mock_instance
            result = ctor.suggest(pool, "anything")
        assert result is None

    def test_suggest_searcher_cached(self):
        pool = PCardList()
        pool.append(AICard({"name": "Card", "type_line": "Creature"}))
        ctor = Constructor()
        ctor._get_searcher = MagicMock()
        ctor.suggest(pool, "query 1")
        ctor.suggest(pool, "query 2")
        assert ctor._get_searcher.call_count == 2

    def test_suggest_falls_back_when_index_out_of_range(self):
        pool = PCardList()
        card = AICard({"name": "Only Card", "type_line": "Creature"})
        pool.append(card)
        ctor = Constructor()
        with (
            patch("environment.Constructor.Constructor._get_searcher") as mock_searcher,
        ):
            mock_instance = MagicMock()
            mock_instance.suggest_indices.return_value = [999, 0]
            mock_searcher.return_value = mock_instance
            result = ctor.suggest(pool, "test", n_results=5)
        assert result is card


class TestConstructor:
    def test_other_card_next(self):
        pool = PCardList()
        card1 = AICard({"name": "Card 1", "type_line": "Creature"})
        card2 = AICard({"name": "Card 2", "type_line": "Creature"})
        card3 = AICard({"name": "Card 3", "type_line": "Creature"})
        pool.append(card1)
        pool.append(card2)
        pool.append(card3)
        ctor = Constructor()
        assert ctor.other_card(pool, action=1) == pool[0]  # return current, move to 1
        assert ctor.other_card(pool, action=1) == pool[1]  # return current, move to 2
        assert ctor.other_card(pool, action=1) == pool[2]  # return current, stay at 2

    def test_other_card_prev(self):
        pool = PCardList()
        card1 = AICard({"name": "Card 1", "type_line": "Creature"})
        card2 = AICard({"name": "Card 2", "type_line": "Creature"})
        pool.append(card1)
        pool.append(card2)
        ctor = Constructor()
        ctor.other_card(pool, action=1)  # move to 1
        assert ctor.other_card(pool, action=2) == pool[1]  # return current, move to 0
        assert ctor.other_card(pool, action=2) == pool[0]  # return current, stay at 0

    def test_other_card_empty_pool(self):
        pool = PCardList()
        ctor = Constructor()
        assert ctor.other_card(pool, action=0) is None
        assert ctor.other_card(pool, action=1) is None

    def test_other_card_no_action(self):
        pool = PCardList()
        card = AICard({"name": "Card 1", "type_line": "Creature"})
        pool.append(card)
        ctor = Constructor()
        result = ctor.other_card(pool, action=0)
        assert result is pool[0]

    def test_pick_card_from_pool_to_deck(self):
        pool = PCardList()
        card = AICard({"name": "Pick Me", "type_line": "Creature"})
        pool.append(card)
        deck = Deck(maxsize=5)
        ctor = Constructor()
        ctor.this_card(pool, deck, action=4)
        assert len(deck) == 1
        assert len(pool) == 0

    def test_drop_card_from_deck_to_pool(self):
        pool = PCardList()
        card = AICard({"name": "Drop Me", "type_line": "Creature"})
        deck = Deck(maxsize=5)
        deck.update_deck(card=card, action=4)
        assert len(deck) == 1
        ctor = Constructor()
        ctor.this_card(pool, deck, action=3)
        assert len(deck) == 0
        assert len(pool) == 1

    def test_this_card_no_action(self):
        pool = PCardList()
        card = AICard({"name": "Card", "type_line": "Creature"})
        pool.append(card)
        deck = Deck(maxsize=5)
        ctor = Constructor()
        ctor.this_card(pool, deck, action=0)
        assert len(deck) == 0
        assert len(pool) == 1


class TestConstructorCopyLimit:
    def _make_card(self, name, type="Creature"):
        return AICard({"name": name, "type": type})

    def test_can_add_first_copy(self):
        ctor = Constructor()
        deck = Deck(maxsize=60)
        card = self._make_card("Lightning Bolt")
        assert ctor.can_add_copy(deck, card) is True

    def test_can_add_4th_copy(self):
        ctor = Constructor()
        deck = Deck(maxsize=60)
        for _ in range(3):
            deck.append(self._make_card("Lightning Bolt"))
        card = self._make_card("Lightning Bolt")
        assert ctor.can_add_copy(deck, card) is True

    def test_cannot_add_5th_copy(self):
        ctor = Constructor()
        deck = Deck(maxsize=60)
        for _ in range(4):
            deck.append(self._make_card("Lightning Bolt"))
        card = self._make_card("Lightning Bolt")
        assert ctor.can_add_copy(deck, card) is False

    def test_basic_lands_unlimited(self):
        ctor = Constructor()
        deck = Deck(maxsize=60)
        for _ in range(10):
            deck.append(self._make_card("Island", "Basic Land — Island"))
        island = self._make_card("Island", "Basic Land — Island")
        assert ctor.can_add_copy(deck, island) is True

    def test_wastes_is_basic(self):
        ctor = Constructor()
        deck = Deck(maxsize=60)
        for _ in range(20):
            deck.append(self._make_card("Wastes", "Basic Land"))
        wastes = self._make_card("Wastes", "Basic Land")
        assert ctor.can_add_copy(deck, wastes) is True

    def test_different_names_independent(self):
        ctor = Constructor()
        deck = Deck(maxsize=60)
        for _ in range(4):
            deck.append(self._make_card("Lightning Bolt"))
        shock = self._make_card("Shock")
        assert ctor.can_add_copy(deck, shock) is True
