"""Run BM25, dense-style, and hybrid retrieval.

Method: Table-aware RAG

Table-aware RAG extracts, indexes, retrieves, and reasons over tables as structured data rather than flattening them into ambiguous text.

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
