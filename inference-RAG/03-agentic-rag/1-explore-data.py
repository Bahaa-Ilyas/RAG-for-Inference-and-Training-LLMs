"""Explore the local mixed-source corpus.

Method: Agentic RAG / multi-step retrieval

Agentic RAG lets an LLM plan, call retrievers or tools repeatedly, inspect intermediate evidence, and decide when enough grounding exists.

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
