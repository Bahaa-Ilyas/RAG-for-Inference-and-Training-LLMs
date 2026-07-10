# Implementation Notes: Tool-augmented RAG / API RAG

## Evidence classification
Paper-supported by ReAct and Toolformer; framework-doc-supported by agent and tool-calling systems.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Static RAG plus dynamic tool calls selected by a planner or deterministic router..
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
Retrieve policy context; choose required tool; validate arguments; call read-only API first; normalize result; synthesize with source and tool citations.

## PDF notes
PDFs provide procedure, policy, and field definitions; tools provide current values. Cite the PDF for rules and the API result for state. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
Web docs explain API behavior, but live APIs should answer current values; protect against webpage prompt injection that tries to alter tool calls. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
