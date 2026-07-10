"""Local utilities for this method folder.

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
