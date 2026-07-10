"""Evaluate the local method demo with tiny qrels.

Method: Version-aware RAG

Version-aware RAG indexes multiple document versions and applies time, product, release, and supersession constraints at retrieval and answer time.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import evaluate

METHOD_KEY = '18-version-aware-rag'


def main() -> None:
    print(json.dumps(evaluate(METHOD_KEY, top_k=3), indent=2))


if __name__ == "__main__":
    main()
