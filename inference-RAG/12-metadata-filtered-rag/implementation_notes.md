# Implementation Notes: Metadata-filtered RAG / auto-retrieval

## Evidence classification
Framework-doc-supported and official-doc-supported by vector databases and cloud search engines.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Vector, sparse, or hybrid retrieval constrained by structured filters..
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
Extract constraints; validate against schema; apply filters and ACLs; retrieve; relax safe filters only when recall is insufficient; answer with visible scope.

## PDF notes
PDF ingestion should store document ID, title, page, section, author, effective date, table flag, OCR confidence, and access-control fields. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
Web metadata should include canonical URL, domain, crawl timestamp, content hash, language, version, and freshness policy. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
