import pytest
import sys
import os
sys.path.append(os.path.abspath('../environment'))
sys.path.append(os.path.abspath('../database'))
sys.path.append(os.path.abspath('.'))
from environment.Pool import Pool
from environment.Card import TCard, MTGCard

card1 = MTGCard({"name": "card1"})
card2 = MTGCard({"name": "card2"})
card3 = MTGCard({"name": "card3"})
cardfirst = MTGCard({"name": "cardfirst"})
cardlast = MTGCard({"name": "cardlast"})
print(card1)
print(card2)
print(card3)
print(cardfirst)
print(cardlast)

def test_initPool():
    pool = Pool()
    pool.insert(card1)
    assert pool.is_first() and pool.is_last() and pool.current is card1 and pool.size == 1

def test_delete_last_card():
    pool = Pool()
    pool.insert(card1)
    del pool.current
    assert pool.current is None and pool.is_first() and pool.is_last() and len(pool) == 0

def test_double_linked():
    pool = Pool()
    pool.insert(card1)
    pool.insert(card3)
    assert pool.first is card1
    pool.insert(card2)
    assert pool.first.next is card2
    assert pool.first.prev is card3
    assert pool.last is card3
    assert pool.last.next is card1
    assert pool.last.prev is card2
    pool.insert(cardfirst, pos = 2)
    assert pool.first is cardfirst
    assert pool.first.next is card1
    assert pool.first.prev is card3
    assert pool.last is card3
    assert pool.last.next is cardfirst
    assert pool.last.prev is card2
    pool.insert(cardlast, pos = 1)
    assert pool.first is cardfirst
    assert pool.first.next is card1
    assert pool.first.prev is cardlast
    assert pool.last is cardlast
    assert pool.last.next is cardfirst
    assert pool.last.prev is card3

def test_delete_cards():
    pool = Pool()
    pool.insert(card1)
    pool.insert(card2)
    pool.insert(card3)
    pool.insert(cardlast)
    assert len(pool) == 4
    assert pool.first is card1
    pool.next
    del pool.current
    assert pool.current is card1
    assert pool.current.next is card3
    assert pool.current.prev is cardlast
    assert len(pool) == 3
    assert pool.first is card1
    assert pool.first.next is card3
    assert pool.first.prev is cardlast
    assert pool.last is cardlast
    assert pool.last.next is card1
    assert pool.last.prev is card3
    pool.last
    del pool.current
    assert len(pool) == 2
    assert pool.first is card1
    assert pool.first.next is card3
    assert pool.first.prev is card3
    assert pool.last is card3
    assert pool.last.next is card1
    assert pool.last.prev is card1
    del pool.current
    assert len(pool) == 1
    assert pool.first is card1
    assert pool.first.next is card1
    assert pool.first.prev is card1
    assert pool.last is card1
    assert pool.last.next is card1
    assert pool.last.prev is card1

def test_properties():
    pool = Pool()
    assert pool.current is None
    assert pool.first is None
    assert pool.last is None
    assert pool.prev is None
    assert pool.next is None
    pool.insert(card1)
    assert pool.current is card1
    assert pool.first is card1
    assert pool.last is card1
    assert pool.prev is card1
    assert pool.next is card1
    pool.insert(card2)
    assert pool.current is card2
    assert pool.first is card1
    # changed current to first card1
    assert pool.last is card2
    # changed current to last card2
    assert pool.prev is card1
    # changed current to card1
    assert pool.next is card2
    # changed current to card2

def test_Exception():
    pool = Pool()
    with pytest.raises(AttributeError) as exc_info:
        pool.insert(None, pos = 1)
    assert exc_info.type is AttributeError
    with pytest.raises(AttributeError) as exc_info:
        pool.insert(None, pos = 2)
    assert exc_info.type is AttributeError

def test_insert_last_in_empty_list():
    pool = Pool()
    pool.insert(cardlast, pos = 1)
    assert pool.current is cardlast
    assert pool.first is cardlast
    assert pool.last is cardlast
    assert pool.prev is cardlast
    assert pool.next is cardlast

def test___str__():
    card = MTGCard({'object': 'card', 'id': '92461f3c-21dc-48a8-b532-9c4e2e1e80b5', 'oracle_id': '017f4471-db5a-4ad0-97b3-b1f12b52578a', 'multiverse_ids': [407603], 'mtgo_id': 59347, 'mtgo_foil_id': 59348, 'tcgplayer_id': 111097, 'cardmarket_id': 287375, 'name': 'Zulaport Chainmage', 'lang': 'en', 'released_at': '2016-01-22', 'uri': 'https://api.scryfall.com/cards/92461f3c-21dc-48a8-b532-9c4e2e1e80b5', 'scryfall_uri': 'https://scryfall.com/card/ogw/93/zulaport-chainmage?utm_source=api', 'layout': 'normal', 'highres_image': True, 'image_status': 'highres_scan', 'image_uris': {'small': 'https://c1.scryfall.com/file/scryfall-cards/small/front/9/2/92461f3c-21dc-48a8-b532-9c4e2e1e80b5.jpg?1562924529', 'normal': 'https://c1.scryfall.com/file/scryfall-cards/normal/front/9/2/92461f3c-21dc-48a8-b532-9c4e2e1e80b5.jpg?1562924529', 'large': 'https://c1.scryfall.com/file/scryfall-cards/large/front/9/2/92461f3c-21dc-48a8-b532-9c4e2e1e80b5.jpg?1562924529', 'png': 'https://c1.scryfall.com/file/scryfall-cards/png/front/9/2/92461f3c-21dc-48a8-b532-9c4e2e1e80b5.png?1562924529', 'art_crop': 'https://c1.scryfall.com/file/scryfall-cards/art_crop/front/9/2/92461f3c-21dc-48a8-b532-9c4e2e1e80b5.jpg?1562924529', 'border_crop': 'https://c1.scryfall.com/file/scryfall-cards/border_crop/front/9/2/92461f3c-21dc-48a8-b532-9c4e2e1e80b5.jpg?1562924529'}, 'mana_cost': '{3}{B}', 'cmc': 4.0, 'type_line': 'Creature — Human Shaman Ally', 'oracle_text': 'Cohort — {T}, Tap an untapped Ally you control: Target opponent loses 2 life.', 'power': '4', 'toughness': '2', 'colors': ['B'], 'color_identity': ['B'], 'keywords': ['Cohort'], 'legalities': {'standard': 'not_legal', 'future': 'not_legal', 'historic': 'not_legal', 'gladiator': 'not_legal', 'pioneer': 'legal', 'explorer': 'not_legal', 'modern': 'legal', 'legacy': 'legal', 'pauper': 'legal', 'vintage': 'legal', 'penny': 'legal', 'commander': 'legal', 'brawl': 'not_legal', 'historicbrawl': 'not_legal', 'alchemy': 'not_legal', 'paupercommander': 'legal', 'duel': 'legal', 'oldschool': 'not_legal', 'premodern': 'not_legal'}, 'games': ['paper', 'mtgo'], 'reserved': False, 'foil': True, 'nonfoil': True, 'finishes': ['nonfoil', 'foil'], 'oversized': False, 'promo': False, 'reprint': False, 'variation': False, 'set_id': 'cd51d245-8f95-45b0-ab5f-e2b3a3eb5dfe', 'set': 'ogw', 'set_name': 'Oath of the Gatewatch', 'set_type': 'expansion', 'set_uri': 'https://api.scryfall.com/sets/cd51d245-8f95-45b0-ab5f-e2b3a3eb5dfe', 'set_search_uri': 'https://api.scryfall.com/cards/search?order=set&q=e%3Aogw&unique=prints', 'scryfall_set_uri': 'https://scryfall.com/sets/ogw?utm_source=api', 'rulings_uri': 'https://api.scryfall.com/cards/92461f3c-21dc-48a8-b532-9c4e2e1e80b5/rulings', 'prints_search_uri': 'https://api.scryfall.com/cards/search?order=released&q=oracleid%3A017f4471-db5a-4ad0-97b3-b1f12b52578a&unique=prints', 'collector_number': '93', 'digital': False, 'rarity': 'common', 'flavor_text': 'The chains obey her. Everything else would do best to get out of her way.', 'card_back_id': '0aeebaf5-8c7d-4636-9e82-8c27447861f7', 'artist': 'Chris Rallis', 'artist_ids': ['a8e7b854-b15a-421a-b66d-6e68187ae285'], 'illustration_id': '5e7552d9-8800-42fc-8245-f482a5daacb7', 'border_color': 'black', 'frame': '2015', 'full_art': False, 'textless': False, 'booster': True, 'story_spotlight': False, 'edhrec_rank': 20839, 'penny_rank': 12440, 'prices': {'usd': '0.03', 'usd_foil': '0.10', 'usd_etched': None, 'eur': '0.02', 'eur_foil': '0.05', 'tix': '0.03'}, 'related_uris': {'gatherer': 'https://gatherer.wizards.com/Pages/Card/Details.aspx?multiverseid=407603', 'tcgplayer_infinite_articles': 'https://infinite.tcgplayer.com/search?contentMode=article&game=magic&partner=scryfall&q=Zulaport+Chainmage&utm_campaign=affiliate&utm_medium=api&utm_source=scryfall', 'tcgplayer_infinite_decks': 'https://infinite.tcgplayer.com/search?contentMode=deck&game=magic&partner=scryfall&q=Zulaport+Chainmage&utm_campaign=affiliate&utm_medium=api&utm_source=scryfall', 'edhrec': 'https://edhrec.com/route/?cc=Zulaport+Chainmage'}, 'purchase_uris': {'tcgplayer': 'https://www.tcgplayer.com/product/111097?page=1&utm_campaign=affiliate&utm_medium=api&utm_source=scryfall', 'cardmarket': 'https://www.cardmarket.com/en/Magic/Products/Search?referrer=scryfall&searchString=Zulaport+Chainmage&utm_campaign=card_prices&utm_medium=text&utm_source=scryfall', 'cardhoarder': 'https://www.cardhoarder.com/cards/59347?affiliate_id=scryfall&ref=card-profile&utm_campaign=affiliate&utm_medium=card&utm_source=scryfall'}})
    print(card)
    print(card.image_uris["small"])
