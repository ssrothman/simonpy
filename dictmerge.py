
from typing import Sequence


def accumulate_dict(original : dict, update : dict) -> dict:
    for key in original:
        if key not in update:
            raise RuntimeError("key %s in original but not in update!"%key)
        
    for key in update:
        if key not in original:
            raise RuntimeError("key %s in update but not in original!"%key)
        
    for key in original:
        if isinstance(original[key], dict):
            original[key] = accumulate_dict(original[key], update[key])
        elif isinstance(original[key], str):
            original[key] = update[key]
        elif isinstance(original[key], int) or isinstance(original[key], float):
            original[key] += update[key]
        elif isinstance(original[key], list) or isinstance(original[key], tuple):
            # assume all entries in the list are the same type
            if isinstance(original[key][0], dict):
                original[key] = [accumulate_dict(a, b) for a, b in zip(original[key], update[key])]
            elif isinstance(original[key][0], str):
                original[key] = update[key] #overwrite strings
            elif isinstance(original[key][0], int) or isinstance(original[key][0], float):
                original[key] = [a + b for a, b in zip(original[key], update[key])]
            else:
                raise TypeError("Unrecognized type in accumulate_dict(): List[%s]"%type(original[key][0]))
        else:
            raise TypeError("Unrecognized type in accumulate_dict(): %s"%type(original[key]))
        
    return original
            

def merge_dict(original : dict, update : dict, allow_new_keys : bool, replace_dict : Sequence[str] = []) -> dict:
    '''
    Recursively merges two dictionaries, overwriting values in the original dict with those from the update dict.
    
    :param original: Original dictionary to be updated
    :type original: dict
    :param update: Dictionary with updates to apply
    :type update: dict
    :param allow_new_keys: If False, raises an error if update contains keys not in original
    :type allow_new_keys: bool
    :param replace_dict: List of keys for which dict values should be replaced entirely rather than merged
    :type replace_dict: Sequence[str]
    :return: Merged dictionary
    :rtype: dict[Any, Any]
    '''
    result = original.copy()
    for key, value in update.items():
        if key in result and type(result[key]) == type(value):
            if isinstance(value, dict) and key not in replace_dict:
                result[key] = merge_dict(result[key], value, allow_new_keys, replace_dict)
            else:
                result[key] = value
        elif key in result:
            raise TypeError(f"Type mismatch for key %s: should be %s but got %s" % (key, type(result[key]), type(value)))
        elif allow_new_keys:
            result[key] = value
        else:
            raise KeyError(f"Key '{key}' not found in original dictionary and new keys are not allowed.")
        
    return result