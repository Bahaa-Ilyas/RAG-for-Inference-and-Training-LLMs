"""Evaluate the local method demo with tiny qrels.

Method: RAPTOR / hierarchical retrieval

RAPTOR recursively clusters chunks and summarizes them into a tree so retrieval can search both granular passages and higher-level summaries.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import evaluate

METHOD_KEY = '06-raptor-hierarchical-rag'


def main() -> None:
    print(json.dumps(evaluate(METHOD_KEY, top_k=3), indent=2))


if __name__ == "__main__":
    main()
