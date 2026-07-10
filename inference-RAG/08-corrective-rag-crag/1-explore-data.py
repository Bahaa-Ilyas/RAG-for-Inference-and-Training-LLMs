"""Explore the local mixed-source corpus.

Method: Corrective RAG / CRAG

Corrective RAG evaluates retrieved evidence before generation and triggers fallback retrieval, filtering, or web search when evidence is weak.

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
