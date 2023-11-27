
from torchtext.data.utils import get_tokenizer

def get_token(s:str):
    tokenizer = get_tokenizer("basic_english")
    return tokenizer(s)

def get_optimized_token(s:str, vocab: list):
    otoken = get_token(s)
    for t in vocab:
        wtoken = get_token(t)
        if len(wtoken) == 0:
            continue
        otidx = 0 
        for i in range(otoken.count(wtoken[0])):
            try:
                otidx = otoken.index(wtoken[0], otidx)
                #print(f"search {str(wtoken)} find {wtoken[0]} at index {otidx}")
                for wt in range(1,len(wtoken)):
                    i = otoken.index(wtoken[wt],otidx)
                    if i != (otidx + wt):
                        raise ValueError
                otidx_last = otoken.index(wtoken[-1],otidx)
                if otidx_last != (otidx + len(wtoken) - 1):
                    raise ValueError
            except ValueError:
                otidx+=1
                # trigger Exception ValueError is a goal to control for loop.
                continue

            otoken[otidx] = t
            for idx in list(reversed(range(otidx+1, otidx_last+1))):
                otoken.pop(idx)

    return otoken
