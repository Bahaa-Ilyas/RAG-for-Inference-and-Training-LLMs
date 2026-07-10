# Implementation Notes: Version-aware RAG

## Evidence classification
Engineering best practice; official-doc-supported indirectly through metadata filtering and crawler freshness docs. No canonical paper defines the entire pattern.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Metadata-filtered and freshness-aware retrieval over versioned documents..
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
Extract version metadata; store validity windows; detect supersession; apply version filters; retrieve; include version scope in final answer.

## PDF notes
PDF versions need document hash, revision date, effective date, supersedes/superseded-by links, page citations, and section IDs that survive revisions. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
Webpage versions need canonical URL, crawl timestamp, release path, content hash, redirect history, and deprecated-page detection. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
