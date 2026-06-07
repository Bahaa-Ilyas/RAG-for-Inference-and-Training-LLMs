"""Evaluate the local method demo with tiny qrels.

Method: Hybrid RAG: BM25 + dense vector retrieval

Hybrid RAG combines sparse lexical retrieval such as BM25 with dense vector retrieval, then fuses the result set before generation.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import evaluate

METHOD_KEY = '01-hybrid-rag'


def main() -> None:
    print(json.dumps(evaluate(METHOD_KEY, top_k=3), indent=2))


if __name__ == "__main__":
    main()
