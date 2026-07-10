# Implementation Notes: Corrective RAG / CRAG

## Evidence classification
Paper-supported and repository-supported by the CRAG authors' public implementation.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Initial retrieval plus evaluator-controlled corrective branch..
5. Add tracing for every retrieval, ranking, generation, and verification step.
6. Compare against a simple hybrid RAG baseline.

## Engineering defaults
- Use stable source IDs and content hashes.
- Store original text plus cleaned text.
- Keep page, URL, section, table, and version metadata on every chunk.
- Batch embeddings and reranker calls.
- Use idempotent ingestion so recrawls and PDF updates are safe.
- Preserve raw evidence for audits.

## Method-specific notes
Retrieve; grade relevance and support; if correct, generate; if ambiguous, refine query or retrieve more; if incorrect, use fallback retrieval or abstain.

## PDF notes
For PDFs, CRAG can reject OCR-noisy chunks, table fragments without headers, or pages that mention terms but do not answer the question. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
CRAG is valuable for webpages because it can detect stale or irrelevant pages and trigger fresh crawl/search under domain and security constraints. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
