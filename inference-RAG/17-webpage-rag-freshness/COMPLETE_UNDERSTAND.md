# Complete Understanding: Webpage RAG with crawler and freshness control

## Core idea
Webpage RAG builds a controlled crawler, cleaned HTML corpus, freshness metadata, and recrawl policy so answers reflect current web content. The method should be understood as an evidence-control strategy: it changes what information is eligible to reach the model, how that information is ranked or transformed, and how the final answer is checked.

## Intuition
Naive webpage ingestion duplicates boilerplate, misses dynamic content, ignores canonical URLs, and serves stale pages. The intuition is to add the missing control surface rather than asking the language model to compensate after retrieval has already failed. Better evidence selection usually improves faithfulness more reliably than a larger generator alone.

## Algorithmic view
1. Define the source schema and retrieval objective.
2. Parse documents and webpages into source-grounded records.
3. Build the required indexes or training artifacts.
4. At query or training time, select evidence with Hybrid retrieval over cleaned page chunks plus freshness-aware ranking and recrawl triggers..
5. Apply filtering, reranking, compression, training loss, or verification as required.
6. Emit an answer, model update, evaluation record, or abstention with traceable provenance.

## Mathematical view
Rank can include relevance plus freshness: score = relevance(q,d) + beta*freshness(d) - gamma*staleness_risk(d) - duplicate_penalty. Evidence quality should be measured separately from answer quality because high answer scores can hide weak retrieval.

## Difference from standard RAG
Standard RAG retrieves a few chunks and generates. Webpage RAG with crawler and freshness control changes that by using Hybrid retrieval over cleaned page chunks plus freshness-aware ranking and recrawl triggers. and by explicitly addressing: Naive webpage ingestion duplicates boilerplate, misses dynamic content, ignores canonical URLs, and serves stale pages.

## How it handles PDFs
- text PDFs: extract text with stable page and section metadata before chunking.
- scanned PDFs: run OCR, store confidence, and keep page image references.
- tables: extract as structured rows/cells, not only flattened text.
- figures: store captions, figure numbers, page image crops, and surrounding paragraphs.
- captions: index with both the visual object and nearby body text.
- page numbers: preserve original and normalized page labels.
- sections: chunk by headings where possible and store parent-child relationships.
- long documents: combine section-aware chunking, parent expansion, summaries, and reranking.

## How it handles webpages
- HTML cleaning: remove nav, footer, cookie banners, and repeated templates.
- boilerplate removal: keep title, headings, main content, captions, and tables.
- canonical URLs: normalize URLs and store redirects.
- crawl timestamp: store first_seen, last_seen, and retrieved_at.
- page versioning: hash cleaned content and keep old versions when answers depend on time.
- dynamic content: render with Playwright only for pages that require JavaScript.
- duplicate detection: hash content and collapse near-duplicates.

## Implementation details
Use deterministic document IDs, stable chunk IDs, source offsets, page and URL metadata, embeddings, sparse indexes when useful, rerankers when precision matters, and prompts that force citation to provided evidence. For this method specifically: Discover URLs; fetch/render; clean HTML; canonicalize; hash; chunk; index with crawl metadata; recrawl by schedule/change signal; retrieve with freshness policy.

## Example metadata schema
```json
{
  "source_id": "policy-2026-04",
  "chunk_id": "policy-2026-04:p12:s3:c02",
  "source_type": "pdf|webpage|table|tool",
  "title": "Example source title",
  "page_number": 12,
  "section": "Eligibility",
  "url": "https://example.com/docs/policy",
  "canonical_url": "https://example.com/docs/policy",
  "content_hash": "sha256:...",
  "version": "2026.04",
  "effective_from": "2026-04-01",
  "retrieved_at": "2026-06-06T00:00:00Z",
  "acl": ["group:legal"],
  "bbox": [72, 120, 540, 700],
  "parent_id": "policy-2026-04:p12:s3",
  "modality": "text"
}
```

## Example prompt templates
Retrieval rewrite prompt:
```text
Rewrite the user question for retrieval. Preserve quoted text, document IDs, dates, versions, and filters. Return 3 concise search queries.
```

Generation prompt:
```text
Answer using only the supplied evidence. Cite every factual claim with source_id and page or URL. If evidence is insufficient, say what is missing.
```

Verification prompt:
```text
For each answer claim, mark Supported, Contradicted, or Not found. Quote the minimal supporting source span identifier, not a long passage.
```

Citation prompt:
```text
Attach citations to the sentence they support. Do not cite a source unless the source directly supports the sentence.
```

## Pseudocode
```python
def run_method(query, user_context):
    constraints = extract_constraints(query, user_context)
    candidates = retrieve(query, constraints)
    candidates = method_specific_transform(query, candidates)
    evidence = select_evidence_under_budget(query, candidates)
    draft = generate_answer(query, evidence)
    verdict = verify_claims(draft, evidence)
    if verdict.has_unsupported_claims:
        return revise_or_abstain(draft, verdict, evidence)
    return attach_citations(draft, evidence)
```

## How to combine it with other RAG methods
Combine with hybrid retrieval for recall, reranking for precision, metadata filters for scope, parent-child retrieval for context, contextual compression for token control, and claim-level verification for auditability. For training workflows, combine with synthetic query generation and hard-negative mining.

## Production recommendations
Start with a small labeled evaluation set before tuning. Store every intermediate score and source span. Keep ingestion idempotent. Version indexes and prompts. Add a no-answer path. Run security filters on retrieved webpages and PDFs. Use human review for high-risk domains.

## Common mistakes
- Chunking PDFs without page or section metadata, which breaks citations.
- Treating OCR text as equally reliable without confidence signals.
- Letting generated query rewrites drop dates, versions, or exact IDs.
- Evaluating only final answer style and ignoring retrieval recall.
- Training on synthetic data before verifying source support.
- Failing to separate train, validation, and test documents.

## Debugging checklist
- Does the gold source appear in top 20 retrieval results?
- Did filters remove the correct document?
- Are chunks too small to answer or too large to rank?
- Are duplicate chunks crowding the context?
- Is the reranker improving nDCG or only adding latency?
- Are citations mapped to exact pages, URLs, rows, or spans?
- Are stale or superseded documents being retrieved?
- Can the system abstain when evidence is missing?

## Evaluation strategy
Offline evaluation should include retrieval recall@k, MRR, nDCG, answer faithfulness, citation precision, citation recall, abstention quality, freshness accuracy, and latency/cost. Online evaluation should include user correction rate, unsupported-claim rate, clickthrough on citations, and failure taxonomy reviews.

## Security and safety concerns
Retrieved PDFs and webpages can contain prompt injection, hidden text, poisoned OCR, malicious links, or confidential data. Enforce ACLs before retrieval, sanitize tool instructions from retrieved text, isolate web content from system prompts, and log provenance for audits.

## Open research questions
- How can systems prove citation support without expensive claim verification?
- How should multimodal evidence be ranked against text evidence?
- How can freshness and version correctness be benchmarked reliably?
- How can synthetic training data avoid teaching hallucinated patterns?
- What is the right human review loop for high-stakes RAG?



## Recommended next reading
See [sources.md](sources.md) for primary papers, official docs, repository links, and uncertainty notes.
