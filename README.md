# RAG for Inference and Training

RAG for Inference and Training covers the top 20 RAG methods for inference and training workflows. Built to accelerate learning, it explains different RAG types with runnable examples, diagrams, enterprise data, evaluation, and end-to-end pipeline responsibility.

This repository currently publishes one complete, GitHub-ready cookbook module:

- [`inference-RAG/01-hybrid-rag`](inference-RAG/01-hybrid-rag): Hybrid RAG from scratch with BM25, dense-style retrieval, reciprocal-rank fusion, enterprise documents, evaluation, diagrams, and local examples.

The rest of the local `inference-RAG` workspace is intentionally ignored for now. This keeps the first GitHub push focused and clean.

## Current Module

The Hybrid RAG module demonstrates conceptualization and responsibility for an end-to-end RAG pipeline:

1. Document ingestion for PDFs, OCR text, webpages, tables, and tool/API outputs.
2. Chunking and metadata strategy.
3. Sparse and dense-style indexing.
4. Retrieval evaluation with qrels, recall@k, and MRR.
5. Response-quality preparation through evidence bundles, citations, and traceable metadata.

## Quick Start

```bash
cd inference-RAG/01-hybrid-rag
python validate_project.py
python 4-run-method.py --query "What is the liability cap in MSA-2026 for Vendor Atlas?"
python 5-evaluate.py
```

## License

MIT. See [LICENSE](LICENSE).
