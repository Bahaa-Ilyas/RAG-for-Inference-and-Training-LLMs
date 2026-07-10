"""Explore the local mixed-source corpus.

Method: Webpage RAG with crawler and freshness control

Webpage RAG builds a controlled crawler, cleaned HTML corpus, freshness metadata, and recrawl policy so answers reflect current web content.

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
