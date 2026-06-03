from database.mtgtools import Database
from mtgtools.PCardList import PCardList
import pandas as pd
import re
import os
from os.path import join
from persistent.list import PersistentList

DATA_DIR = os.path.abspath("./data")

db = Database()
# TODO Import arguments for basic and wcc

cards : PCardList = PCardList()
cards.name = "Basic Collection"
print(f"Start loading {cards.name}")
data = pd.read_csv(join(DATA_DIR,"mtgcb-collection-2026-06-03.csv"), usecols=["Count", "Name", "Edition Code"])
print(data)
for idx in data.index:
    amount, name, edition_code = data.iloc[idx]
    name = re.sub( r' \(.*\)', "", name)
    card = db.cards.where_exactly(name=name, set=edition_code.lower())
    if len(card):
        for n in range(amount):
            cards.append(card[0])
    else:
        print(f"{name} {edition_code}")

print(cards)
print(len(cards))
db.root.basic_collection = cards
db.commit()

cards : PCardList = PCardList()
cards.name = "WCC Collection"
wccdecks: PersistentList = PersistentList()
print(f"Start loading {cards.name}")
for idx, f in enumerate(os.listdir(join(DATA_DIR, "magic_WCC_decks/"))):
    print(f)
    wccdeck = db.load_from_file(join(DATA_DIR, "magic_WCC_decks",f))
    cards.extend(wccdeck)
    wccdecks.append(wccdeck)

print(cards)
print(len(cards))
db.root.wcc_collection = cards
db.root.wcc_decks = wccdecks
db.commit()
