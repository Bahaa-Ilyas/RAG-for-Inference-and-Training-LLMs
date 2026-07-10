# Implementation Notes: Table-aware RAG

## Evidence classification
Engineering best practice informed by table QA and document parsing research.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Hybrid text/table retrieval plus optional structured execution over extracted tables..
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
Detect tables; reconstruct schema; index caption/header/rows; retrieve candidate tables; execute or reason over rows; verify units and cite cells.

## PDF notes
PDF table RAG needs layout extraction, header reconstruction, units, captions, footnotes, page numbers, and OCR confidence for scanned tables. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
Web tables should preserve DOM structure, surrounding headings, caption text, URL, crawl timestamp, and repeated header rows. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
