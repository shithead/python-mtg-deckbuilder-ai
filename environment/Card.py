import json

from mtgtools.PCard import PCard


class AICard(PCard):
    def __init__(self, response_dict: dict, amount: int = 1):
        super().__init__(response_dict)
        self.amount = amount

    @staticmethod
    def load_from_pcard(pcard: PCard):
        return AICard(json.loads(pcard.json.lower()))
