"""Validate that the Hybrid RAG cookbook is GitHub-ready."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "README.md",
    "RESPONSIBILITY.md",
    "ARCHITECTURE.md",
    "COMPLETE_UNDERSTAND.md",
    "implementation_notes.md",
    "sources.md",
    "architecture.mmd",
    "assets/architecture.mmd",
    "assets/paper_diagram.svg",
    "assets/pipeline.mmd",
    "data/corpus.jsonl",
    "data/README.md",
    "data/queries.jsonl",
    "data/qrels.jsonl",
    "examples/sample_policy.pdf",
    "examples/scanned_page_ocr.txt",
    "examples/sample_webpage.html",
    "examples/sample_table.csv",
    "examples/tool_response.json",
    "docs/enterprise_vendor_msa.md",
    "docs/incident_postmortem_2026-05.md",
    "docs/access_control_matrix.csv",
    "docs/soc2_control_excerpt.txt",
    "docs/procurement_policy_addendum.md",
    "docs/data_retention_policy.html",
    "utils/cookbook_core.py",
    "bm25_hybrid.py",
    ".github/workflows/ci.yml",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "requirements.txt",
]

RUNNABLE_SCRIPTS = [
    "1-explore-data.py",
    "2-build-index.py",
    "3-retrieve.py",
    "4-run-method.py",
    "5-evaluate.py",
    "bm25_hybrid.py",
    "examples/run_example.py",
]


def fail(message: str) -> None:
    raise SystemExit(f"Validation failed: {message}")


def validate_files() -> None:
    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if not path.exists():
            fail(f"missing {relative}")
        if path.is_file() and path.stat().st_size == 0:
            fail(f"empty file {relative}")


def validate_mermaid() -> None:
    text = (ROOT / "architecture.mmd").read_text(encoding="utf-8")
    if "flowchart LR" not in text:
        fail("architecture.mmd must use flowchart LR for left-to-right rendering")


def validate_readme() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lower_text = text.lower()
    required_phrases = [
        "Hybrid RAG From Scratch",
        "End-to-End RAG Responsibility",
        "document ingestion",
        "chunking strategy",
        "Retrieval evaluation",
        "Response quality",
        "Quick Start",
        "Example Data Gallery",
        "Production Mapping",
        "Quality Gates",
        "assets/paper_diagram.svg",
    ]
    for phrase in required_phrases:
        if phrase.lower() not in lower_text:
            fail(f"README.md is missing {phrase!r}")


def run_scripts() -> None:
    for relative in RUNNABLE_SCRIPTS:
        result = subprocess.run(
            [sys.executable, str(ROOT / relative)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            fail(f"{relative} exited with {result.returncode}: {detail[-1000:]}")


def main() -> None:
    validate_files()
    validate_mermaid()
    validate_readme()
    run_scripts()
    print("Hybrid RAG cookbook validation passed.")


if __name__ == "__main__":
    main()
