import torch
from sentence_transformers import SentenceTransformer


class CardEncoder:
    def __init__(self, model_name : str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def _card_to_text(self, card) -> str:
        parts = [f"name: {card.name}"]
        type_text = getattr(card, "type_line", None) or getattr(card, "type", "") or ""
        if type_text:
            parts.append(f"type: {type_text}")
        if hasattr(card, "mana_cost") and card.mana_cost:
            parts.append(f"mana_cost: {card.mana_cost}")
        if hasattr(card, "oracle_text") and card.oracle_text:
            parts.append(f"oracle_text: {card.oracle_text}")
        return "\n".join(parts)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        return self._model.encode(text, convert_to_tensor=True)

    @torch.no_grad()
    def encode(self, card) -> torch.Tensor:
        text = self._card_to_text(card)
        return self._model.encode(text, convert_to_tensor=True)

    @torch.no_grad()
    def encode_many(self, cards) -> torch.Tensor:
        texts = [self._card_to_text(c) for c in cards]
        return self._model.encode(texts, convert_to_tensor=True)
