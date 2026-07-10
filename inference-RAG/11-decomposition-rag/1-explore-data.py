"""Explore the local mixed-source corpus.

Method: Decomposition RAG / sub-question RAG

Decomposition RAG breaks a complex question into sub-questions, retrieves evidence for each, and synthesizes the final answer from the sub-results.

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
