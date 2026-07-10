"""Evaluate the local method demo with tiny qrels.

Method: Layout-aware / multimodal PDF RAG

Layout-aware PDF RAG uses document layout, OCR, page images, figures, tables, captions, and multimodal models to retrieve from complex PDFs.

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

from utils.cookbook_core import evaluate

METHOD_KEY = '16-layout-aware-multimodal-pdf-rag'


def main() -> None:
    print(json.dumps(evaluate(METHOD_KEY, top_k=3), indent=2))


if __name__ == "__main__":
    main()
