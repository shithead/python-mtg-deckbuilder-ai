
from torchtext.data.utils import get_tokenizer

def get_token(s:str):
    tokenizer = get_tokenizer("basic_english")
    return tokenizer(s)
