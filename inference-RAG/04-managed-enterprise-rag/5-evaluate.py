"""Evaluate the local method demo with tiny qrels.

Method: Managed enterprise RAG platforms: Azure AI Search, Vertex AI RAG Engine, AWS Bedrock Knowledge Bases

Managed enterprise RAG delegates indexing, retrieval, security integration, scaling, and managed model connections to cloud platforms.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import evaluate

METHOD_KEY = '04-managed-enterprise-rag'


def main() -> None:
    print(json.dumps(evaluate(METHOD_KEY, top_k=3), indent=2))


if __name__ == "__main__":
    main()
