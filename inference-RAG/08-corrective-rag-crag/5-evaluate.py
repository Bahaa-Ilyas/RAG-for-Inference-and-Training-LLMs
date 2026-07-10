"""Evaluate the local method demo with tiny qrels.

Method: Corrective RAG / CRAG

Corrective RAG evaluates retrieved evidence before generation and triggers fallback retrieval, filtering, or web search when evidence is weak.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import evaluate

METHOD_KEY = '08-corrective-rag-crag'


def main() -> None:
    print(json.dumps(evaluate(METHOD_KEY, top_k=3), indent=2))


if __name__ == "__main__":
    main()
