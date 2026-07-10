# Implementation Notes: Layout-aware / multimodal PDF RAG

## Evidence classification
Framework-doc-supported by parsing tools and engineering best practice; related to document AI research.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Text, layout, table, and image-aware retrieval over page objects and structured elements..
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
Parse pages; run OCR/layout/table detection; create element records; embed text and images; retrieve by modality; assemble page-grounded context.

## PDF notes
This method is designed for normal and scanned PDFs, OCR, layout extraction, tables, figures, captions, equations, page numbers, references, and long scientific papers. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
The same layout principles apply to rendered web pages or dynamic docs captured by Playwright, but canonical URL and crawl metadata remain required. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
