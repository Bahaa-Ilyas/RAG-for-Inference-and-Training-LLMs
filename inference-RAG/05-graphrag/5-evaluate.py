"""Evaluate the local method demo with tiny qrels.

Method: GraphRAG

GraphRAG builds an entity and relationship graph from a corpus and retrieves local neighborhoods or global community summaries for graph-grounded generation.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import evaluate

METHOD_KEY = '05-graphrag'


def main() -> None:
    print(json.dumps(evaluate(METHOD_KEY, top_k=3), indent=2))


if __name__ == "__main__":
    main()
