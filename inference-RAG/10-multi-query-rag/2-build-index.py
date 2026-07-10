"""Build and persist a local teaching index.

Method: Multi-query RAG

Multi-query RAG generates several diverse search queries for one user question, retrieves for each, and fuses results.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import INDEX_PATH, build_index


def main() -> None:
    index = build_index()
    print(json.dumps({
        "index_path": str(INDEX_PATH),
        "document_count": len(index["documents"]),
        "vocabulary_size": len(index["idf"]),
        "note": "This is a local teaching index, not a production vector database."
    }, indent=2))


if __name__ == "__main__":
    main()
