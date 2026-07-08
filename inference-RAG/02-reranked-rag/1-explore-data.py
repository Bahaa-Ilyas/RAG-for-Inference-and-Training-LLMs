"""Explore the local mixed-source corpus.

Method: Reranked RAG

Reranked RAG retrieves a broad candidate set and applies a stronger cross-encoder or LLM reranker before passing compact evidence to the generator.

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
