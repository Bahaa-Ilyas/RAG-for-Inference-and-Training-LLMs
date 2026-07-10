"""Run the method-specific cookbook demo for Webpage RAG with crawler and freshness control.

Method: Webpage RAG with crawler and freshness control

Webpage RAG builds a controlled crawler, cleaned HTML corpus, freshness metadata, and recrawl policy so answers reflect current web content.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import run_method

METHOD_KEY = '17-webpage-rag-freshness'


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the cookbook demo for Webpage RAG with crawler and freshness control')
    parser.add_argument("--query", default="Where does vendor onboarding require security review?")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_method(METHOD_KEY, args.query, top_k=args.top_k), indent=2))


if __name__ == "__main__":
    main()
