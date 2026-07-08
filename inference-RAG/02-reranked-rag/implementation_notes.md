# Implementation Notes: Reranked RAG

## Evidence classification
Paper-supported by cross-encoder and late-interaction retrieval literature; framework-doc-supported by reranking APIs.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Any first-stage sparse, dense, or hybrid retriever followed by cross-encoder, late-interaction, or LLM reranking..
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
Retrieve a large candidate set; split or trim candidates for the reranker; score query-document pairs; diversify by source; pass top evidence to the generator.

## PDF notes
Reranking helps PDFs when adjacent chunks have similar text but only one contains the answer, table caption, or required page-specific citation. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
For webpages, reranking should penalize boilerplate, stale copies, and pages whose title or canonical URL does not match the query intent. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
