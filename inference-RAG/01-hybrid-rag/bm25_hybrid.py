"""Run the Hybrid RAG cookbook demo.

This is the shortest entrypoint in the project. It keeps the public command
simple while the actual teaching implementation lives in `utils/cookbook_core.py`.

Run:
    python bm25_hybrid.py
    python bm25_hybrid.py --query "What do the current API docs say about rollback?"
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import run_method


METHOD_KEY = "01-hybrid-rag"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Hybrid RAG demo.")
    parser.add_argument(
        "--query",
        default="Where does vendor onboarding require security review?",
        help="Question to run against the local mixed-source corpus.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of evidence items to return.")
    args = parser.parse_args()

    result = run_method(METHOD_KEY, args.query, top_k=args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
