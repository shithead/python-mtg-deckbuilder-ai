import sys
from database.mtgtools import Database
from environment.Deck import Deck
from environment.Constructor import Constructor

def main(query_text: str):
    print("Loading database...")
    db = Database()
    pool = db.loadPool()
    deck = Deck(maxsize=60)
    ctor = Constructor()

    card = ctor.suggest(pool, query_text, n_results=10)
    if card is None:
        print(f"No cards found for '{query_text}'")
        return
    print(f"Theme: {card.name} ({card.type_line})")
    ctor.add_card_to_deck(pool, deck, card)

    for _ in range(30):
        ranked = ctor.rank_cards(deck, pool)
        best, score = ranked[0]
        if score < 0.5:
            break
        if not ctor.can_add_copy(deck, best):
            continue
        print(f"  + {best.name} ({score:.3f})")
        ctor.add_card_to_deck(pool, deck, best)

    for card in pool:
        type_text = getattr(card, "type_line", None) or getattr(card, "type", "") or ""
        if "Basic" in type_text:
            while len(deck) < 60 and ctor.can_add_copy(deck, card):
                ctor.add_card_to_deck(pool, deck, card)

    print(f"\nDeck ({len(deck)} cards):")
    for c in deck:
        print(f"  {c.name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_deck.py <search query>")
        print("Example: python build_deck.py 'aggressive red creature with haste'")
        sys.exit(1)
    main(sys.argv[1])
