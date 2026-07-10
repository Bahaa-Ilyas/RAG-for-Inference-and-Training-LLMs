"""Run the method-specific cookbook demo for Managed enterprise RAG platforms: Azure AI Search, Vertex AI RAG Engine, AWS Bedrock Knowledge Bases.

Method: Managed enterprise RAG platforms: Azure AI Search, Vertex AI RAG Engine, AWS Bedrock Knowledge Bases

Managed enterprise RAG delegates indexing, retrieval, security integration, scaling, and managed model connections to cloud platforms.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import run_method

METHOD_KEY = '04-managed-enterprise-rag'


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the cookbook demo for Managed enterprise RAG platforms: Azure AI Search, Vertex AI RAG Engine, AWS Bedrock Knowledge Bases')
    parser.add_argument("--query", default="Where does vendor onboarding require security review?")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_method(METHOD_KEY, args.query, top_k=args.top_k), indent=2))


if __name__ == "__main__":
    main()
