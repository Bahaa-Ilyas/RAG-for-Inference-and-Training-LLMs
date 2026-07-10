"""Evaluate the local method demo with tiny qrels.

Method: Contextual compression RAG

Contextual compression filters, extracts, or summarizes retrieved passages so only query-relevant evidence enters the generation context.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import evaluate

METHOD_KEY = '14-contextual-compression-rag'


def main() -> None:
    print(json.dumps(evaluate(METHOD_KEY, top_k=3), indent=2))


if __name__ == "__main__":
    main()
