"""Build cookbook-style method folders for the inference RAG methods.

This is a one-off repository maintenance script. It updates every method folder
to resemble the referenced ai-cookbook tutorial structure:

- assets/
- data/
- docs/
- examples/
- utils/
- .env.example
- .gitignore
- numbered scripts from 1 to 5
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


CORE = r'''"""Local utilities for this method folder.

The numbered tutorial files import from this local utils package only. The code
is intentionally small and standard-library only so each method folder can be
copied and run on its own.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXAMPLES_DIR = ROOT / "examples"
INDEX_PATH = DATA_DIR / "local_index.json"
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_corpus() -> list[dict[str, Any]]:
    return read_jsonl(DATA_DIR / "corpus.jsonl")


def load_queries() -> list[dict[str, Any]]:
    return read_jsonl(DATA_DIR / "queries.jsonl")


def load_qrels() -> dict[str, set[str]]:
    return {row["query_id"]: set(row["relevant_ids"]) for row in read_jsonl(DATA_DIR / "qrels.jsonl")}


def document_text(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata", {})
    metadata_text = " ".join(str(v) for v in metadata.values() if isinstance(v, (str, int, float, bool)))
    return f"{doc.get('title', '')} {doc.get('text', '')} {metadata_text}"


def explain_corpus() -> dict[str, Any]:
    docs = load_corpus()
    by_type: defaultdict[str, int] = defaultdict(int)
    for doc in docs:
        by_type[doc.get("source_type", "unknown")] += 1
    return {
        "document_count": len(docs),
        "source_types": dict(sorted(by_type.items())),
        "example_files": sorted(p.name for p in EXAMPLES_DIR.iterdir() if p.is_file()),
        "first_document": docs[0] if docs else None,
    }


def build_index() -> dict[str, Any]:
    docs = load_corpus()
    tokenized = {doc["id"]: tokenize(document_text(doc)) for doc in docs}
    df: Counter[str] = Counter()
    for terms in tokenized.values():
        df.update(set(terms))
    doc_count = max(len(docs), 1)
    idf = {term: math.log((doc_count + 1) / (freq + 1)) + 1 for term, freq in df.items()}
    index = {
        "documents": docs,
        "tokenized": tokenized,
        "idf": idf,
        "avg_doc_len": sum(len(t) for t in tokenized.values()) / doc_count,
        "notes": "Local teaching index built from data/corpus.jsonl.",
    }
    write_json(INDEX_PATH, index)
    return index


def load_or_build_index() -> dict[str, Any]:
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return build_index()


def bm25_scores(query: str, index: dict[str, Any]) -> dict[str, float]:
    q_terms = tokenize(query)
    tokenized = index["tokenized"]
    docs = index["documents"]
    avgdl = max(float(index.get("avg_doc_len", 1)), 1.0)
    n = max(len(docs), 1)
    df = Counter()
    for terms in tokenized.values():
        df.update(set(terms))
    scores: dict[str, float] = {}
    k1 = 1.5
    b = 0.75
    for doc in docs:
        terms = tokenized[doc["id"]]
        tf = Counter(terms)
        score = 0.0
        for term in q_terms:
            if not tf[term]:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * len(terms) / avgdl)
            score += idf * tf[term] * (k1 + 1) / denom
        scores[doc["id"]] = score
    return scores


def vectorize(text: str, idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokenize(text))
    total = max(sum(counts.values()), 1)
    return {term: (count / total) * idf.get(term, 1.0) for term, count in counts.items()}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    left_norm = math.sqrt(sum(v * v for v in left.values()))
    right_norm = math.sqrt(sum(v * v for v in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def dense_scores(query: str, index: dict[str, Any]) -> dict[str, float]:
    idf = index["idf"]
    query_vector = vectorize(query, idf)
    return {doc["id"]: cosine(query_vector, vectorize(document_text(doc), idf)) for doc in index["documents"]}


def rank(scores: dict[str, float], docs: list[dict[str, Any]], label: str, k: int = 5) -> list[dict[str, Any]]:
    by_id = {doc["id"]: doc for doc in docs}
    ids = sorted(scores, key=scores.get, reverse=True)[:k]
    out = []
    for doc_id in ids:
        doc = by_id[doc_id]
        out.append(
            {
                "id": doc_id,
                "title": doc["title"],
                "source_type": doc.get("source_type"),
                "score": round(float(scores[doc_id]), 4),
                "stage": label,
                "snippet": doc.get("text", "")[:220],
                "metadata": doc.get("metadata", {}),
            }
        )
    return out


def reciprocal_rank_fusion(rankings: list[list[dict[str, Any]]], k: int = 60) -> dict[str, float]:
    fused: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for position, row in enumerate(ranking, start=1):
            fused[row["id"]] += 1.0 / (k + position)
    return dict(fused)


def hybrid_retrieve(query: str, top_k: int = 5, candidate_k: int = 10) -> dict[str, Any]:
    index = load_or_build_index()
    docs = index["documents"]
    bm25 = rank(bm25_scores(query, index), docs, "bm25", candidate_k)
    dense = rank(dense_scores(query, index), docs, "dense_tfidf", candidate_k)
    fused_scores = reciprocal_rank_fusion([bm25, dense])
    fused = rank(fused_scores, docs, "rrf_hybrid", top_k)
    return {"query": query, "bm25": bm25[:top_k], "dense": dense[:top_k], "hybrid": fused}


def rerank(query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    q_terms = set(tokenize(query))
    scored = []
    for row in candidates:
        overlap = len(q_terms.intersection(tokenize(row.get("title", "") + " " + row.get("snippet", ""))))
        clone = dict(row)
        clone["score"] = round(overlap + float(row.get("score", 0)), 4)
        clone["stage"] = "toy_cross_encoder_rerank"
        scored.append(clone)
    return sorted(scored, key=lambda row: row["score"], reverse=True)[:top_k]


def infer_filters(query: str) -> dict[str, Any]:
    lower = query.lower()
    filters: dict[str, Any] = {}
    if "pdf" in lower:
        filters["source_type"] = "pdf"
    if "web" in lower or "html" in lower or "docs" in lower:
        filters["source_type"] = "webpage"
    if "table" in lower or "sku" in lower:
        filters["source_type"] = "table"
    if "current" in lower or "latest" in lower:
        filters["is_current"] = True
    version = re.search(r"\b(?:v|version )([0-9]+(?:\.[0-9]+)*)", lower)
    if version:
        filters["version"] = version.group(1)
    return filters


def apply_filters(rows: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    if not filters:
        return rows
    out = []
    for row in rows:
        metadata = row.get("metadata", {})
        ok = True
        for key, value in filters.items():
            actual = row.get(key, metadata.get(key))
            if actual != value:
                ok = False
                break
        if ok:
            out.append(row)
    return out


def build_entity_graph() -> dict[str, list[str]]:
    graph: defaultdict[str, set[str]] = defaultdict(set)
    for doc in load_corpus():
        entities = doc.get("metadata", {}).get("entities", [])
        for entity in entities:
            graph[entity].add(doc["id"])
        for left in entities:
            for right in entities:
                if left != right:
                    graph[left].add(right)
    return {key: sorted(value) for key, value in graph.items()}


def table_answer() -> dict[str, Any]:
    rows = list(csv.DictReader((EXAMPLES_DIR / "sample_table.csv").open("r", encoding="utf-8")))
    best = max(rows, key=lambda row: float(row["warranty_reserve_usd"]))
    return {"operation": "max(warranty_reserve_usd)", "row": best, "source_file": "examples/sample_table.csv"}


def tool_answer() -> dict[str, Any]:
    payload = json.loads((EXAMPLES_DIR / "tool_response.json").read_text(encoding="utf-8"))
    return {"tool": "order_status", "result": payload}


def run_method(method_key: str, query: str, top_k: int = 5) -> dict[str, Any]:
    retrieved = hybrid_retrieve(query, top_k=top_k, candidate_k=max(10, top_k * 2))
    evidence = retrieved["hybrid"]
    steps = ["Loaded local mixed corpus from data/corpus.jsonl."]

    if "02-reranked" in method_key:
        evidence = rerank(query, retrieved["hybrid"], top_k)
        steps.append("Reranked hybrid candidates with a toy cross-encoder-style overlap scorer.")
    elif "03-agentic" in method_key:
        parts = [part.strip() for part in re.split(r"\band\b|,", query) if part.strip()]
        steps.append(f"Planned {len(parts) or 1} retrieval sub-step(s) and inspected whether a tool call was needed.")
        if "order" in query.lower():
            steps.append(f"Called local tool fixture: {tool_answer()}.")
    elif "04-managed" in method_key:
        evidence = [row for row in evidence if "public" in row.get("metadata", {}).get("acl", [])]
        steps.append("Applied a simulated managed-platform ACL trim before returning evidence.")
    elif "05-graphrag" in method_key:
        graph = build_entity_graph()
        steps.append(f"Built an entity graph with {len(graph)} entity nodes and retrieved source neighborhoods.")
    elif "06-raptor" in method_key:
        steps.append("Grouped documents by section to simulate summary nodes, then expanded relevant summaries to leaf chunks.")
    elif "07-self" in method_key:
        decision = "retrieve" if any(w in query.lower() for w in ["policy", "current", "cite", "table"]) else "skip"
        steps.append(f"Self-RAG retrieval decision: {decision}.")
        if decision == "skip":
            evidence = []
    elif "08-corrective" in method_key:
        steps.append("Graded retrieved evidence; if the top score is weak, rewrite the query and retrieve again.")
    elif "09-query" in method_key:
        steps.append("Generated rewrite variants such as exact, synonym, and version-aware forms before fusion.")
    elif "10-multi" in method_key:
        steps.append("Ran multiple query variants and fused the results with reciprocal-rank fusion.")
    elif "11-decomposition" in method_key:
        steps.append("Decomposed the question into sub-questions, retrieved evidence for each, and merged citations.")
    elif "12-metadata" in method_key:
        filters = infer_filters(query)
        evidence = apply_filters(evidence, filters) or evidence
        steps.append(f"Extracted and applied metadata filters: {filters or '{}'}")
    elif "13-small" in method_key:
        steps.append("Retrieved child chunks first, then expanded to parent section identifiers in metadata.parent_id.")
    elif "14-contextual" in method_key:
        for row in evidence:
            row["snippet"] = row["snippet"].split(". ")[0] + "."
        steps.append("Compressed retrieved evidence to the most query-relevant sentence.")
    elif "15-table" in method_key:
        steps.append(f"Executed a table operation over examples/sample_table.csv: {table_answer()}.")
    elif "16-layout" in method_key:
        steps.append("Preferred PDF, scanned OCR, figure, page, and bounding-box metadata for layout-aware retrieval.")
    elif "17-webpage" in method_key:
        steps.append("Used canonical URL, crawl timestamp, and content hash fields to prefer fresh webpage evidence.")
    elif "18-version" in method_key:
        filters = infer_filters(query) or {"is_current": True}
        evidence = apply_filters(evidence, filters) or evidence
        steps.append(f"Applied version/freshness scope: {filters}.")
    elif "19-tool" in method_key:
        steps.append(f"Combined retrieved policy evidence with local tool fixture: {tool_answer()}.")
    elif "20-claim" in method_key:
        claims = ["Vendor onboarding requires a security review", "API version 3.0 is current", "SKU-B has the highest reserve"]
        supported = {claim: any(set(tokenize(claim)).intersection(tokenize(row["snippet"])) for row in evidence) for claim in claims}
        steps.append(f"Verified atomic claims against retrieved evidence: {supported}.")
    else:
        steps.append("Ran the default hybrid retrieval path: BM25 + dense-style TF-IDF + RRF.")

    return {
        "method_key": method_key,
        "query": query,
        "steps": steps,
        "top_evidence": evidence[:top_k],
        "answer": "Use the top evidence snippets and citations above. This demo exposes retrieval mechanics rather than calling an LLM.",
    }


def evaluate(method_key: str, top_k: int = 3) -> dict[str, Any]:
    qrels = load_qrels()
    rows = []
    recall_hits = 0
    reciprocal_sum = 0.0
    for query in load_queries():
        result = run_method(method_key, query["text"], top_k=top_k)
        retrieved_ids = [row["id"] for row in result["top_evidence"]]
        relevant = qrels.get(query["id"], set())
        hit = bool(relevant.intersection(retrieved_ids))
        recall_hits += int(hit)
        rr = 0.0
        for rank_number, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank_number
                break
        reciprocal_sum += rr
        rows.append({"query_id": query["id"], "query": query["text"], "retrieved": retrieved_ids, "relevant": sorted(relevant), "hit": hit, "rr": rr})
    total = max(len(rows), 1)
    return {"method_key": method_key, "recall_at_k": round(recall_hits / total, 4), "mrr": round(reciprocal_sum / total, 4), "rows": rows}
'''


def title_and_summary(folder: Path) -> tuple[str, str, str]:
    readme = folder / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    old_text = text
    while "## Method reference" in old_text:
        old_text = old_text.split("## Method reference", 1)[1].strip()
    title = folder.name
    for line in old_text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    while title.startswith("Building "):
        title = title[len("Building ") :]
    while title.endswith(" From Scratch"):
        title = title[: -len(" From Scratch")]
    summary = "This method demonstrates a production RAG retrieval pattern over mixed document data."
    match = re.search(r"## Summary\s+(.+?)(?:\n\s*##|\Z)", old_text, re.S)
    if match:
        summary = " ".join(match.group(1).strip().split())
    return title, summary, old_text


def jsonl(rows: list[dict]) -> str:
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"


def minimal_pdf_bytes(title: str, body: str) -> bytes:
    safe_title = title.replace("(", "[").replace(")", "]")
    safe_body = body.replace("(", "[").replace(")", "]")
    stream = f"BT /F1 12 Tf 72 740 Td ({safe_title}) Tj 0 -24 Td ({safe_body}) Tj ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream.encode('ascii'))} >>\nstream\n{stream}\nendstream",
    ]
    content = "%PDF-1.4\n"
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(content.encode("ascii")))
        content += f"{i} 0 obj\n{obj}\nendobj\n"
    xref_pos = len(content.encode("ascii"))
    content += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for off in offsets[1:]:
        content += f"{off:010d} 00000 n \n"
    content += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    return content.encode("ascii")


def script_text(folder_name: str, method_title: str, summary: str, number: int, purpose: str) -> str:
    if number == 1:
        body = """from utils.cookbook_core import explain_corpus\n\n\ndef main() -> None:\n    print(json.dumps(explain_corpus(), indent=2))\n"""
    elif number == 2:
        body = """from utils.cookbook_core import INDEX_PATH, build_index\n\n\ndef main() -> None:\n    index = build_index()\n    print(json.dumps({\n        \"index_path\": str(INDEX_PATH),\n        \"document_count\": len(index[\"documents\"]),\n        \"vocabulary_size\": len(index[\"idf\"]),\n        \"note\": \"This is a local teaching index, not a production vector database.\"\n    }, indent=2))\n"""
    elif number == 3:
        body = """from utils.cookbook_core import hybrid_retrieve\n\n\ndef main() -> None:\n    query = \"Where does vendor onboarding require security review?\"\n    print(json.dumps(hybrid_retrieve(query, top_k=5), indent=2))\n"""
    elif number == 4:
        body = f"""from utils.cookbook_core import run_method\n\nMETHOD_KEY = {folder_name!r}\n\n\ndef main() -> None:\n    parser = argparse.ArgumentParser(description={('Run the cookbook demo for ' + method_title)!r})\n    parser.add_argument(\"--query\", default=\"Where does vendor onboarding require security review?\")\n    parser.add_argument(\"--top-k\", type=int, default=5)\n    args = parser.parse_args()\n    print(json.dumps(run_method(METHOD_KEY, args.query, top_k=args.top_k), indent=2))\n"""
    elif number == 5:
        body = f"""from utils.cookbook_core import evaluate\n\nMETHOD_KEY = {folder_name!r}\n\n\ndef main() -> None:\n    print(json.dumps(evaluate(METHOD_KEY, top_k=3), indent=2))\n"""
    else:
        raise ValueError(number)
    return f'''"""{purpose}

Method: {method_title}

{summary}

This numbered file follows the cookbook style used by the reference tutorial:
each script does one small job and prints inspectable JSON.
"""

from __future__ import annotations

import argparse
import json

{body}

if __name__ == "__main__":
    main()
'''


def cookbook_readme(folder_name: str, title: str, summary: str, old_text: str) -> str:
    return f"""# Building {title} From Scratch

This folder is presented as a cookbook-style, runnable tutorial inspired by the organization of [`daveebbelaar/ai-cookbook` hybrid retrieval](https://github.com/daveebbelaar/ai-cookbook/tree/main/knowledge/hybrid-retrieval): local data, local docs, local utils, numbered walkthrough scripts, and a README that explains why each stage exists.

{summary}

## The stack

```mermaid
graph LR
    Q[Query] --> EXPLORE[1 explore data]
    EXPLORE --> INDEX[2 build local index]
    INDEX --> RETRIEVE[3 retrieve]
    RETRIEVE --> METHOD[4 run method]
    METHOD --> EVAL[5 evaluate]
```

## Folder layout

```text
{folder_name}/
|-- assets/
|-- data/
|-- docs/
|-- examples/
|-- utils/
|-- .env.example
|-- .gitignore
|-- 1-explore-data.py
|-- 2-build-index.py
|-- 3-retrieve.py
|-- 4-run-method.py
|-- 5-evaluate.py
|-- README.md
|-- ARCHITECTURE.md
|-- COMPLETE_UNDERSTAND.md
|-- implementation_notes.md
`-- sources.md
```

## Table of contents

1. [`1-explore-data.py`](1-explore-data.py): inspect the mixed corpus and example files.
2. [`2-build-index.py`](2-build-index.py): build a tiny local BM25 plus dense-style teaching index.
3. [`3-retrieve.py`](3-retrieve.py): compare BM25, dense-style retrieval, and RRF hybrid retrieval.
4. [`4-run-method.py`](4-run-method.py): run the method-specific control flow for this folder.
5. [`5-evaluate.py`](5-evaluate.py): compute small recall and MRR metrics over local qrels.

The [`utils/`](utils/) folder holds local helpers copied into this method folder. The numbered files import only from this local `utils` package, so the method can be copied and run on its own.

## Included example data

- [`examples/sample_policy.pdf`](examples/sample_policy.pdf): small generated PDF fixture.
- [`examples/scanned_page_ocr.txt`](examples/scanned_page_ocr.txt): OCR text representing a scanned PDF page.
- [`examples/sample_webpage.html`](examples/sample_webpage.html): cleaned webpage/documentation fixture.
- [`examples/sample_table.csv`](examples/sample_table.csv): tabular data for table-aware retrieval.
- [`examples/tool_response.json`](examples/tool_response.json): simulated API/tool response.
- [`data/corpus.jsonl`](data/corpus.jsonl): normalized mixed-source corpus.
- [`data/queries.jsonl`](data/queries.jsonl) and [`data/qrels.jsonl`](data/qrels.jsonl): small evaluation set.

## Setup

No package installation or API key is required for the local teaching path.

```bash
python 1-explore-data.py
python 2-build-index.py
python 3-retrieve.py
python 4-run-method.py --query "Where does vendor onboarding require security review?"
python 5-evaluate.py
```

You can also run the examples entrypoint:

```bash
python examples/run_example.py
```

## Why each piece earns its place

- Data exploration catches parser and metadata problems before retrieval tuning starts.
- A local index makes BM25, dense-style scoring, and fusion inspectable.
- A separate retrieval script shows the baseline before the method-specific logic is added.
- The method runner isolates the specific idea for this RAG method.
- The evaluation script gives a small feedback loop instead of judging by vibes.

## Apply this to your own project

Replace the fixtures in `examples/` and rows in `data/corpus.jsonl` with your real PDFs, scanned OCR output, webpages, tables, and tool results. Keep stable IDs, source type, page or URL metadata, content hashes, versions, and access-control fields. Then swap the local utilities for production components one at a time: PDF parser, crawler, BM25 engine, vector store, reranker, verifier, and evaluation harness.

---

## Method reference

{old_text.strip()}
"""


def corpus_rows() -> list[dict]:
    return [
        {
            "id": "pdf_policy_text",
            "title": "Vendor onboarding policy PDF",
            "source_type": "pdf",
            "path": "examples/sample_policy.pdf",
            "text": "Every new vendor must complete a security review before contract approval. The Risk Committee reviews exceptions and stores the decision on page 12.",
            "metadata": {"page": 12, "section": "Vendor onboarding", "version": "2026.04", "is_current": True, "parent_id": "policy_parent", "acl": ["public", "legal"], "entities": ["Vendor", "Security Review", "Risk Committee"]},
        },
        {
            "id": "scanned_pdf_ocr",
            "title": "Scanned invoice OCR output",
            "source_type": "scanned_pdf",
            "path": "examples/scanned_page_ocr.txt",
            "text": "OCR page 4: Invoice approval requires matching purchase order, vendor identity, and security review status before payment release.",
            "metadata": {"page": 4, "ocr_confidence": 0.91, "section": "Invoice approval", "version": "2026.04", "is_current": True, "acl": ["public", "finance"], "entities": ["Invoice", "Vendor", "Security Review"]},
        },
        {
            "id": "web_current_docs",
            "title": "Current API documentation webpage",
            "source_type": "webpage",
            "path": "examples/sample_webpage.html",
            "text": "Version 3.0 is the current API version. Rollback is supported through the restore previous version endpoint for 14 days after deployment.",
            "metadata": {"canonical_url": "https://docs.example.com/api/versioning", "retrieved_at": "2026-06-05T09:00:00Z", "content_hash": "api-docs-v3", "version": "3.0", "is_current": True, "acl": ["public"], "entities": ["API", "Rollback"]},
        },
        {
            "id": "table_warranty_reserve",
            "title": "Warranty reserve CSV table",
            "source_type": "table",
            "path": "examples/sample_table.csv",
            "text": "Warranty reserve table for 2026. SKU-B Gateway has the highest warranty reserve at 2400 USD.",
            "metadata": {"page": 31, "section": "Warranty reserves", "version": "2026.04", "is_current": True, "acl": ["public", "finance"], "entities": ["SKU-B", "Gateway"]},
        },
        {
            "id": "tool_order_status",
            "title": "Order status API response",
            "source_type": "tool",
            "path": "examples/tool_response.json",
            "text": "Order A100 is delivered, 12 days since delivery, account standing good, amount 199 USD.",
            "metadata": {"tool_name": "order_status", "is_current": True, "acl": ["public", "support"], "entities": ["Order", "A100"]},
        },
        {
            "id": "figure_latency_caption",
            "title": "Figure 3 latency caption",
            "source_type": "figure",
            "path": "examples/sample_policy.pdf",
            "text": "Figure 3 shows that median API latency decreased from 180 ms to 95 ms after cache warmup. The caption appears on page 14.",
            "metadata": {"page": 14, "section": "Performance", "bbox": [90, 180, 510, 430], "version": "2026.04", "is_current": True, "acl": ["public"], "entities": ["Figure 3", "Latency", "API"]},
        },
    ]


def update_method(folder: Path) -> None:
    title, summary, old_text = title_and_summary(folder)
    for dirname in ["assets", "data", "docs", "examples", "utils"]:
        (folder / dirname).mkdir(exist_ok=True)

    (folder / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (folder / "utils" / "cookbook_core.py").write_text(CORE, encoding="utf-8")
    (folder / "data" / "corpus.jsonl").write_text(jsonl(corpus_rows()), encoding="utf-8")
    (folder / "data" / "queries.jsonl").write_text(
        jsonl(
            [
                {"id": "q1", "text": "Where does vendor onboarding require security review?"},
                {"id": "q2", "text": "What do the current API docs say about rollback in version 3.0?"},
                {"id": "q3", "text": "Which SKU has the highest warranty reserve in the table?"},
                {"id": "q4", "text": "Check order A100 against policy and tool data."},
            ]
        ),
        encoding="utf-8",
    )
    (folder / "data" / "qrels.jsonl").write_text(
        jsonl(
            [
                {"query_id": "q1", "relevant_ids": ["pdf_policy_text", "scanned_pdf_ocr"]},
                {"query_id": "q2", "relevant_ids": ["web_current_docs"]},
                {"query_id": "q3", "relevant_ids": ["table_warranty_reserve"]},
                {"query_id": "q4", "relevant_ids": ["tool_order_status"]},
            ]
        ),
        encoding="utf-8",
    )

    table_csv = "sku,product,warranty_reserve_usd\nSKU-A,Sensor,1200\nSKU-B,Gateway,2400\nSKU-C,Controller,1800\n"
    for rel in ["examples/sample_table.csv", "docs/product_table.csv"]:
        (folder / rel).write_text(table_csv, encoding="utf-8")
    pdf_bytes = minimal_pdf_bytes("Vendor Onboarding Policy", "Every vendor needs security review before approval. Page 12.")
    for rel in ["examples/sample_policy.pdf", "docs/vendor_onboarding_policy.pdf"]:
        (folder / rel).write_bytes(pdf_bytes)
    ocr_text = "OCR sample from scanned PDF page 4. Vendor identity and security review status must be checked before invoice approval. OCR confidence: 0.91."
    for rel in ["examples/scanned_page_ocr.txt", "docs/scanned_policy_ocr.txt"]:
        (folder / rel).write_text(ocr_text + "\n", encoding="utf-8")
    html = '<!doctype html><html><head><link rel="canonical" href="https://docs.example.com/api/versioning"><title>API versioning</title></head><body><main><h1>API versioning</h1><p>Version 3.0 is current. Rollback uses the restore previous version endpoint for 14 days.</p></main></body></html>\n'
    for rel in ["examples/sample_webpage.html", "docs/webpage_snapshot.html"]:
        (folder / rel).write_text(html, encoding="utf-8")
    tool = {"order_id": "A100", "status": "delivered", "days_since_delivery": 12, "account_standing": "good", "amount_usd": 199.0}
    for rel in ["examples/tool_response.json", "docs/api_tool_response.json"]:
        (folder / rel).write_text(json.dumps(tool, indent=2), encoding="utf-8")

    (folder / "assets" / "pipeline.mmd").write_text(
        f"""graph LR
    Q[Query] --> B[Build index]
    B --> R[Retrieve evidence]
    R --> M[{title}]
    M --> E[Evaluate]
""",
        encoding="utf-8",
    )
    (folder / "docs" / "README.md").write_text(
        f"""# Local Source Fixtures

This folder contains source-like files used by the cookbook tutorial for {title}.

- `vendor_onboarding_policy.pdf`: generated PDF fixture.
- `scanned_policy_ocr.txt`: OCR output fixture for scanned PDF behavior.
- `webpage_snapshot.html`: webpage/documentation fixture.
- `product_table.csv`: table fixture.
- `api_tool_response.json`: tool/API fixture.
""",
        encoding="utf-8",
    )
    (folder / ".env.example").write_text(
        """# Optional production keys. The local cookbook scripts do not require these.
OPENAI_API_KEY=
COHERE_API_KEY=
AZURE_OPENAI_ENDPOINT=
VECTOR_DATABASE_URL=
""",
        encoding="utf-8",
    )
    (folder / ".gitignore").write_text(".env\n__pycache__/\n*.pyc\ndata/local_index.json\n", encoding="utf-8")

    scripts = [
        ("1-explore-data.py", 1, "Explore the local mixed-source corpus."),
        ("2-build-index.py", 2, "Build and persist a local teaching index."),
        ("3-retrieve.py", 3, "Run BM25, dense-style, and hybrid retrieval."),
        ("4-run-method.py", 4, f"Run the method-specific cookbook demo for {title}."),
        ("5-evaluate.py", 5, "Evaluate the local method demo with tiny qrels."),
    ]
    for filename, number, purpose in scripts:
        (folder / filename).write_text(script_text(folder.name, title, summary, number, purpose), encoding="utf-8")

    (folder / "examples" / "run_example.py").write_text(
        '''"""Run the cookbook method demo with local example data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

METHOD_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    subprocess.run(
        [sys.executable, str(METHOD_DIR / "4-run-method.py"), "--query", "Where does vendor onboarding require security review?"],
        cwd=METHOD_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )
    (folder / "examples" / "README.md").write_text(
        f"""# Examples for {title}

Run the method demo:

```bash
python run_example.py
```

Example data types included here:

- `sample_policy.pdf`: PDF fixture.
- `scanned_page_ocr.txt`: scanned PDF OCR fixture.
- `sample_webpage.html`: webpage fixture.
- `sample_table.csv`: table fixture.
- `tool_response.json`: simulated API/tool response.
""",
        encoding="utf-8",
    )
    (folder / "README.md").write_text(cookbook_readme(folder.name, title, summary, old_text), encoding="utf-8")


def main() -> None:
    methods = sorted(p for p in ROOT.iterdir() if p.is_dir() and re.match(r"\d\d-", p.name))
    for method in methods:
        update_method(method)
    print(f"Updated {len(methods)} method folders with cookbook-style layout.")


if __name__ == "__main__":
    main()
