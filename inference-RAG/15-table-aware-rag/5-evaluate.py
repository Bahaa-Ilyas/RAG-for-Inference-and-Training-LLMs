"""Evaluate the local method demo with tiny qrels.

Method: Table-aware RAG

Table-aware RAG extracts, indexes, retrieves, and reasons over tables as structured data rather than flattening them into ambiguous text.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import evaluate

METHOD_KEY = '15-table-aware-rag'


def main() -> None:
    print(json.dumps(evaluate(METHOD_KEY, top_k=3), indent=2))


if __name__ == "__main__":
    main()
