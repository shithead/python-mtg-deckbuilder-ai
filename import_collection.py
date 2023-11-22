from database.mtgtools import Database

db = Database()


cards = db.load_from_file("./data/collection.txt")
cards.name = "Basic Collection"
db.root.basic_collection = cards
db.commit()
