"""Explore the local mixed-source corpus.

Method: Managed enterprise RAG platforms: Azure AI Search, Vertex AI RAG Engine, AWS Bedrock Knowledge Bases

Managed enterprise RAG delegates indexing, retrieval, security integration, scaling, and managed model connections to cloud platforms.

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
