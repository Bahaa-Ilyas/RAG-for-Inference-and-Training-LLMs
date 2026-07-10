"""Explore the local mixed-source corpus.

Method: Multi-query RAG

Multi-query RAG generates several diverse search queries for one user question, retrieves for each, and fuses results.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import explain_corpus


def main() -> None:
    print(json.dumps(explain_corpus(), indent=2))


if __name__ == "__main__":
    main()
