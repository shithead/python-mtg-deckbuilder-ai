import pytest
import sys
import os
from mtgtools.MtgDB import MtgDB
sys.path.append(os.path.abspath('../environment'))
sys.path.append(os.path.abspath('../database'))
sys.path.append(os.path.abspath('.'))
from database.mtgtools import Database

def initDB():
    return Database()

def test_loadPool():
    pool = initDB().loadPool()
    print(len(pool))
    print(len(pool.unique_names()))
    print(len(pool.unique_cards()))
