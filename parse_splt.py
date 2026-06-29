import simonplot as splt

def parse_splt(instring):
    # WARNING: THIS RUNS ARBITRARY PYTHON CODE. DO NOT USE ON UNTRUSTED INPUT.
    instring = instring.strip()
    instring = instring.replace('var::', 'splt.variable.')
    instring = instring.replace('cut::', 'splt.cut.')
    return eval(instring)