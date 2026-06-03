import chromadb
from chromadb.utils import embedding_functions
from database.mtgtools import Database
from environment.Card import AICard
from mtgtools.PCardList import PCardList
from utils.utils import get_token

CHROMA_DATA_PATH = "data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "MTGCards"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

collection : chromadb.Collection = None

client.delete_collection(name=COLLECTION_NAME)
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
        )
else:
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )
    db = Database()
    pool : PCardList = db.loadPool().unique_names() + db.loadWccPool().unique_names()
    documents = []
    metadatas = []
    
    for i, card in  enumerate(pool):
        print(f"\rcreate vectorDB entry from card: {i+1}/{len(pool)}", end="", flush=True)
    
    
        document = ""
        data_dict = dict()
        data_dict.update({"type": card.type_line})
        if "planswalker" in card.type_line:
            print(card)

        document += f"name: {card.name}\n"
        data_dict.update({"name": card.name})
        if card.mana_cost is not None:
            data_dict.update({"mana_cost": card.mana_cost})
            document += f"mana_cost: {card.mana_cost}\n"
        if card.power is not None:
            data_dict.update({"power": card.power})
            document += f"power: {card.power}\n"
        if card.toughness is not None:
            data_dict.update({"toughness": card.toughness})
            document += f"toughness: {card.toughness}\n"
        if card.loyalty is not None:
            data_dict.update({"loyalty": card.loyalty})
            document += f"loyalty: {card.loyalty}\n"
        if card.oracle_text is not None:
            data_dict.update({"oracle_text": card.oracle_text})
            document += f"oracle_text: {card.oracle_text}\n"
        metadatas.append(data_dict)
        documents.append(document)
    
    collection.add(
        documents=documents,
        ids=[f"id{i}" for i in range(len(documents))],
        metadatas=metadatas
    )

query_results = collection.query(
    query_texts=["Find some cards with life"],
    n_results=60
)

print(query_results.keys())

print(query_results["documents"])

print(query_results["ids"])

print(query_results["distances"])

print(query_results["metadatas"])

print(query_results["data"])
