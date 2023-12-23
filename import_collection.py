from database.mtgtools import Database
from mtgtools.PCardList import PCardList
import pandas as pd
import re
import os
from os.path import join

DATA_DIR = os.path.abspath("./data")

db = Database()
# TODO Import arguments for basic and wcc

cards : PCardList = PCardList()
cards.name = "Basic Collection"
print(f"Start loading {cards.name}")
data = pd.read_csv(join(DATA_DIR,"myCollection.csv"), usecols=["Count", "Name", "Edition"])
print(data)
for idx in data.index:
    amount, name, edition = data.iloc[idx]
    edition = re.sub(r' Core Set', "", edition)        
    if "Modern Masters" in edition:
        edition = re.sub(r' Edition', "", edition)        
    edition = re.sub(r' Promos', "", edition)        
    name = re.sub( r' \(.*\)', "", name)
    card = db.cards.where_exactly(name=name, set_name=edition)
    if len(card):
        for n in range(amount):
            cards.append(card[0])
    else:
        print(f"{name} {edition}")

print(cards)
print(len(cards))
db.root.basic_collection = cards
db.commit()

cards : PCardList = PCardList()
cards.name = "WCC Collection"
print(f"Start loading {cards.name}")
for f in os.listdir(join(DATA_DIR, "magic_WCC_decks/")):
    cards = db.load_from_file(join(DATA_DIR, "magic_WCC_decks",f))

print(cards)
print(len(cards))
db.root.wcc_collection = cards
db.commit()
