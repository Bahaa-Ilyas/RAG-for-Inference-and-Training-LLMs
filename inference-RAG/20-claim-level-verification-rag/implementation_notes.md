# Implementation Notes: Claim-level verification RAG

## Evidence classification
Paper-supported by FEVER and FActScore; engineering best practice for auditable RAG.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Post-generation or interleaved retrieval per atomic claim..
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
Draft answer; extract atomic claims; retrieve evidence per claim; classify support/contradiction; revise unsupported claims; return cited answer.

## PDF notes
For PDFs, verification should point claims to page-level spans, table cells, captions, or figure references and reject unsupported extrapolation. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
For webpages, verification must consider freshness, canonical URL, source reliability, and prompt-injection filtering for each supporting citation. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
