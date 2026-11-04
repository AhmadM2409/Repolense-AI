"""Prepare the hash-verified cl100k_base cache; use --check for offline verification."""

import argparse
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
CACHE_KEY = hashlib.sha1(URL.encode()).hexdigest()


def prepare(cache_dir, offline=False):
    path = Path(cache_dir) / CACHE_KEY
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == SHA256:
        return path
    if offline:
        raise RuntimeError(
            "Missing/corrupt cl100k_base cache; run prepare_tokenizer.py online first"
        )
    with urlopen(URL, timeout=30) as response:
        data = response.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError("Tokenizer asset integrity check failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    temporary.write_bytes(data)
    temporary.replace(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", default=os.environ.get("TIKTOKEN_CACHE_DIR", ".test-tmp/tokenizer-cache")
    )
    parser.add_argument(
        "--check", action="store_true", help="Offline: never download; fail if absent or corrupt"
    )
    options = parser.parse_args()
    prepare(options.cache_dir, options.check)
    print("cl100k_base cache verified: " + SHA256)
