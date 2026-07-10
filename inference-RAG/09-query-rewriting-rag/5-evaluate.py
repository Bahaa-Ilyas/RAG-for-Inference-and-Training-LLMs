"""Evaluate the local method demo with tiny qrels.

Method: Query rewriting / query expansion RAG

Query rewriting transforms the user's question into retrieval-optimized forms such as normalized search queries, hypothetical answers, synonyms, or domain-specific expansions.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import evaluate

METHOD_KEY = '09-query-rewriting-rag'


def main() -> None:
    print(json.dumps(evaluate(METHOD_KEY, top_k=3), indent=2))


if __name__ == "__main__":
    main()
