"""Evaluate the local method demo with tiny qrels.

Method: Small-to-big retrieval / parent-child retrieval

Small-to-big retrieval searches small chunks for precision, then expands selected hits to larger parent sections for generation.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import evaluate

METHOD_KEY = '13-small-to-big-parent-child-rag'


def main() -> None:
    print(json.dumps(evaluate(METHOD_KEY, top_k=3), indent=2))


if __name__ == "__main__":
    main()
