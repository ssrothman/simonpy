import re
from typing import List, Union

def strip_units(s: str) -> str:
    if '[' in s and ']' in s:
        start = s.index('[')
        end = s.index(']')
        return s[:start].strip() + s[end+1:].strip()
    else:
        return s.strip()
    
def strip_dollar_signs(s: str) -> str:
    return s.replace('$', '')

def attempt_regex_match(pattern: str, axiskey: str) -> bool:
    #escape special characters
    pattern_escaped = re.escape(pattern)
    #replace '*' wildcard with alphanumeric match
    pattern_escaped = pattern_escaped.replace(r'\*', r'[a-zA-Z0-9]*')
    #ensure no leading or trailing characters
    pattern_escaped = "^" + pattern_escaped + "$" 

    return re.fullmatch(pattern_escaped, axiskey) is not None

def find_match(keys : List[str], patterns : List[str],
               ignore_case : bool) -> Union[str, None]:
    
    for key in keys:
        for pattern in patterns:
            if ignore_case:
                if attempt_regex_match(pattern.lower(), key.lower()):
                    return pattern
            else:
                if attempt_regex_match(pattern, key):
                    return pattern
                
    return None