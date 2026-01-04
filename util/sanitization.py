from typing import List, Union, Tuple
import numpy as np

def maybe_valcov_to_definitely_valcov(evaluated : np.ndarray | Tuple[np.ndarray, np.ndarray]):
        if isinstance(evaluated, tuple):
            hist, cov = evaluated

            if len(hist.shape) != 1:
                raise ValueError("Unpacking valcov pair yielded val with shape %s! (expected 1D)"%hist.shape)
            if len(cov.shape) != 2:
                raise ValueError("Unpacking valcov pair yielded cov with shape %s! (expected 2D)"%cov.shape)

            if cov.shape != (len(hist), len(hist)):
                raise ValueError("cov shape not the square of val shape!")
            
            thelen = len(hist)
            thedtype = hist.dtype
        else:
            if len(evaluated.shape) == 1:
                hist = evaluated
                cov = None
                thelen = len(hist)
                thedtype = hist.dtype
            elif len(evaluated.shape) == 2:
                hist = None
                cov = evaluated
                thelen = cov.shape[0]
                thedtype = cov.dtype
            else:
                raise ValueError("Unexpected data shape %s! (Expected 1D hist or 2D covariance)"%(evaluated.shape))

        return hist, cov, thelen, thedtype

def ensure_same_length(*args):
    result = []
    for arg in args:
        if isinstance(arg, list):
            result.append(arg)
        else:
            result.append([arg])
    
    maxlen = max([len(x) for x in result])

    for i in range(len(result)):
        if len(result[i]) == 1:
            result[i] = result[i] * maxlen
        elif len(result[i]) != maxlen:
            raise ValueError("All input arguments must have the same length or be of length 1")

    return result


def all_same_key(things : List, skip : Union[int, None]=None):
    
    indices = list(range(len(things)))
    if skip is not None:
        indices.remove(skip)

    if len(indices) == 0:
        return True

    thing0 = things[indices[0]]

    for i in indices[1:]:
        if thing0.key != things[i].key:
            return False
        
    return True
