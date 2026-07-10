# Implementation Notes: Webpage RAG with crawler and freshness control

## Evidence classification
Framework-doc-supported by web crawling/parsing tools and engineering best practice.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Hybrid retrieval over cleaned page chunks plus freshness-aware ranking and recrawl triggers..
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
Discover URLs; fetch/render; clean HTML; canonicalize; hash; chunk; index with crawl metadata; recrawl by schedule/change signal; retrieve with freshness policy.

## PDF notes
Linked PDFs should enter the PDF pipeline with the referring URL, crawl timestamp, and file hash preserved as provenance. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
This method directly covers crawling, sitemap ingestion, robots awareness, HTML cleaning, canonical URLs, freshness, versioning, dynamic pages, duplicates, and prompt injection. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
