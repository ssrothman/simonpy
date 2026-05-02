

import hashlib
from typing import Any

#code from https://stackoverflow.com/questions/3431825/how-to-generate-an-md5-checksum-of-a-file-in-python

def checksum_file(filepath: str, fs:Any) -> str:
    """Calculate the MD5 checksum of a file."""
    with fs.open(filepath, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()
