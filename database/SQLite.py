import sqlite3
import base64
import json
import copy

import sys
sys.path.append("..")
sys.path.append(".")

from environment.Card import MTGCard, TCard
from environment.Pool import Pool

class SQLite():
    def __init__(self, database: str):
        self.con = sqlite3.connect(database)

    def loadPool(self) -> Pool:
        cur = self.con.cursor()
        res = cur.execute("SELECT  max(U.Count) as Amount, Rawdata FROM mtg,(SELECT ID, Count FROM user_collection_2 ) AS U WHERE U.ID = mtg.ID GROUP BY Name;")
        pool = Pool()
        for amount, rawdata in res.fetchall():
            jdata = json.loads(base64.b64decode(rawdata))
            jdata.update({"amount": amount})
            card = MTGCard(jdata)
            pool.insert(copy.deepcopy(card))

        return pool
