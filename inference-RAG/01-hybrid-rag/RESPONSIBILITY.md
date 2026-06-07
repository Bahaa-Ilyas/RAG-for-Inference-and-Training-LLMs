# End-to-End RAG Responsibility

This repository treats RAG as a full production pipeline, not as a single retrieval function.

The owner of the system is responsible for:

1. Document ingestion: collecting PDFs, scanned OCR, webpages, tables, figures, tool outputs, and enterprise records.
2. Parsing and normalization: preserving page numbers, sections, canonical URLs, content hashes, versions, ACLs, and source IDs.
3. Chunking strategy: choosing chunk sizes, overlap, parent-child relationships, and table/figure handling based on evidence needs.
4. Indexing: building sparse BM25 indexes, dense vector indexes, metadata filters, and version-aware fields.
5. Retrieval evaluation: measuring recall@k, MRR, qrels coverage, hard negatives, and failure cases before tuning generation.
6. Context assembly: deduplicating, ordering, compressing, and citing retrieved evidence under the model context budget.
7. Response quality: checking groundedness, citation precision, abstention behavior, freshness, security, and user-facing answer clarity.
8. Operations: monitoring latency, cost, source freshness, parser failures, empty results, and unsupported-answer rates.

For this Hybrid RAG cookbook, BM25 plus dense-style retrieval is only the retrieval baseline. The real engineering responsibility is the complete path from raw source data to evaluated, auditable, grounded responses.
