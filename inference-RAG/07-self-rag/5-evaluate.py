"""Evaluate the local method demo with tiny qrels.

Method: Self-RAG / adaptive retrieval

Self-RAG trains or prompts a model to decide when to retrieve, evaluate retrieved passages, and critique its own generations.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import evaluate

METHOD_KEY = '07-self-rag'


def main() -> None:
    print(json.dumps(evaluate(METHOD_KEY, top_k=3), indent=2))


if __name__ == "__main__":
    main()
