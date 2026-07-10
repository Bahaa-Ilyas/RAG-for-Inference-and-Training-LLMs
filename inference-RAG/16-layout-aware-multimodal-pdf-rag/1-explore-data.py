"""Explore the local mixed-source corpus.

Method: Layout-aware / multimodal PDF RAG

Layout-aware PDF RAG uses document layout, OCR, page images, figures, tables, captions, and multimodal models to retrieve from complex PDFs.

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
