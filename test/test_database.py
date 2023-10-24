import pytest
import sqlite3
import sys
import os
sys.path.append(os.path.abspath('../environment'))
sys.path.append(os.path.abspath('../database'))
sys.path.append(os.path.abspath('.'))
from database.SQLite import SQLite
from environment.Pool import Pool
from environment.Card import TCard

def initDB():
    return SQLite(os.path.abspath("../MTG-search/mtg.db"))

def test_initDB():
    assert isinstance(initDB().con, sqlite3.Connection)

def test_loadPool():
    assert isinstance(initDB().loadPool(), Pool)

def test_poolSize():
    pool = initDB().loadPool()
    f_card: TCard = pool.first
    countedsize = 1
    while(not pool.is_last()):
        if countedsize > len(pool):
            pytest.fail("countedsize too big")
        pool.next
        countedsize += 1

    assert len(pool) == countedsize
