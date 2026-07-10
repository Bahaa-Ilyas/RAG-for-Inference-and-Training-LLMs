"""Run BM25, dense-style, and hybrid retrieval.

Method: Multi-query RAG

Multi-query RAG generates several diverse search queries for one user question, retrieves for each, and fuses results.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import hybrid_retrieve


def main() -> None:
    query = "Where does vendor onboarding require security review?"
    print(json.dumps(hybrid_retrieve(query, top_k=5), indent=2))


if __name__ == "__main__":
    main()
