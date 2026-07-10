"""Pedagogical runnable demo for Decomposition RAG / sub-question RAG.

This file is intentionally self-contained:

1. It imports only Python standard-library modules.
2. It reads only this method folder's `examples/sample_corpus.json` by default.
3. It implements a small teaching version of the inference-time method.
4. It prints JSON so you can inspect each retrieval, ranking, verification, or
   training-data step without needing a vector database or model server.

The implementation is not a production system. It is a readable executable
blueprint for the control flow described in this folder's Markdown files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_RE.split(text or "") if p.strip()]
    return parts or [text.strip()] if text.strip() else []


def load_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def doc_text(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata", {})
    extra = " ".join(
        str(v)
        for k, v in metadata.items()
        if k in {"section", "version", "source_type", "canonical_url", "effective_from"}
    )
    return f"{doc.get('title', '')} {doc.get('text', '')} {extra}"


def add_score(doc: dict[str, Any], score: float, reason: str) -> dict[str, Any]:
    clone = dict(doc)
    clone["score"] = round(float(score), 4)
    clone["reason"] = reason
    return clone


def bm25_scores(query: str, docs: list[dict[str, Any]]) -> dict[str, float]:
    q_terms = tokenize(query)
    doc_terms = {d["id"]: tokenize(doc_text(d)) for d in docs}
    avgdl = sum(len(t) for t in doc_terms.values()) / max(len(docs), 1)
    df: Counter[str] = Counter()
    for terms in doc_terms.values():
        df.update(set(terms))
    scores: dict[str, float] = {}
    k1 = 1.5
    b = 0.75
    n = max(len(docs), 1)
    for doc in docs:
        terms = doc_terms[doc["id"]]
        tf = Counter(terms)
        score = 0.0
        for term in q_terms:
            if tf[term] == 0:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * len(terms) / max(avgdl, 1))
            score += idf * tf[term] * (k1 + 1) / denom
        scores[doc["id"]] = score
    return scores


def idf_map(docs: list[dict[str, Any]]) -> dict[str, float]:
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(tokenize(doc_text(doc))))
    n = max(len(docs), 1)
    return {term: math.log((n + 1) / (freq + 1)) + 1 for term, freq in df.items()}


def vectorize(text: str, idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokenize(text))
    total = max(sum(tf.values()), 1)
    return {term: (count / total) * idf.get(term, 1.0) for term, count in tf.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(term, 0.0) for term, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def dense_scores(query: str, docs: list[dict[str, Any]]) -> dict[str, float]:
    idf = idf_map(docs)
    q_vec = vectorize(query, idf)
    return {doc["id"]: cosine(q_vec, vectorize(doc_text(doc), idf)) for doc in docs}


def rank_from_scores(
    docs: list[dict[str, Any]], scores: dict[str, float], reason: str, top_k: int = 5
) -> list[dict[str, Any]]:
    ranked = sorted(docs, key=lambda d: scores.get(d["id"], 0.0), reverse=True)
    return [add_score(doc, scores.get(doc["id"], 0.0), reason) for doc in ranked[:top_k]]


def reciprocal_rank_fusion(rankings: list[list[dict[str, Any]]], k: int = 60) -> dict[str, float]:
    fused: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc["id"]] += 1.0 / (k + rank)
    return dict(fused)


def hybrid_retrieve(query: str, docs: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """Teach the default hybrid retrieval pattern.

    Production systems usually use BM25 from OpenSearch/Elasticsearch/Azure AI
    Search and dense vectors from a neural embedding model. This demo uses BM25
    plus TF-IDF cosine so the ranking math is visible and runnable locally.
    """

    bm25 = rank_from_scores(docs, bm25_scores(query, docs), "BM25 lexical match", top_k=max(top_k * 3, 10))
    dense = rank_from_scores(docs, dense_scores(query, docs), "TF-IDF dense-style cosine", top_k=max(top_k * 3, 10))
    fused = reciprocal_rank_fusion([bm25, dense])
    by_id = {doc["id"]: doc for doc in docs}
    ranked_ids = sorted(fused, key=fused.get, reverse=True)[:top_k]
    return [add_score(by_id[doc_id], fused[doc_id], "hybrid reciprocal-rank fusion") for doc_id in ranked_ids]


def rerank(query: str, docs: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    q_terms = set(tokenize(query))
    scores = {}
    for doc in docs:
        terms = tokenize(doc_text(doc))
        overlap = len(q_terms.intersection(terms))
        title_bonus = 2 if any(term in tokenize(doc.get("title", "")) for term in q_terms) else 0
        exact_bonus = 2 if query.lower() in doc_text(doc).lower() else 0
        scores[doc["id"]] = overlap + title_bonus + exact_bonus + float(doc.get("score", 0))
    return rank_from_scores(docs, scores, "cross-encoder-style overlap rerank", top_k)


def compress_doc(query: str, doc: dict[str, Any], max_sentences: int = 2) -> dict[str, Any]:
    q_terms = set(tokenize(query))
    scored = []
    for sentence in sentences(doc.get("text", "")):
        score = len(q_terms.intersection(tokenize(sentence)))
        if score:
            scored.append((score, sentence))
    if not scored:
        scored = [(0, sentences(doc.get("text", ""))[0] if sentences(doc.get("text", "")) else "")]
    text = " ".join(sentence for _score, sentence in sorted(scored, reverse=True)[:max_sentences])
    clone = dict(doc)
    clone["text"] = text
    clone["reason"] = f"{doc.get('reason', 'retrieved')} + contextual compression"
    return clone


def evidence_summary(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for doc in docs:
        metadata = doc.get("metadata", {})
        rows.append(
            {
                "id": doc.get("id"),
                "title": doc.get("title"),
                "score": doc.get("score"),
                "reason": doc.get("reason"),
                "source_type": metadata.get("source_type"),
                "page": metadata.get("page"),
                "url": metadata.get("canonical_url") or metadata.get("url"),
                "version": metadata.get("version"),
                "snippet": doc.get("text", "")[:220],
            }
        )
    return rows


def answer_from_evidence(query: str, docs: list[dict[str, Any]], prefix: str) -> str:
    if not docs:
        return f"{prefix}: insufficient evidence for '{query}'."
    citations = ", ".join(
        f"{doc['id']}"
        + (f":p{doc.get('metadata', {}).get('page')}" if doc.get("metadata", {}).get("page") else "")
        for doc in docs[:3]
    )
    best = docs[0]
    return f"{prefix}: based on {citations}, {best.get('text', '').strip()[:260]}"


def infer_filters(query: str) -> dict[str, Any]:
    q = query.lower()
    filters: dict[str, Any] = {}
    if "pdf" in q:
        filters["source_type"] = "pdf"
    if "web" in q or "page" in q or "docs" in q:
        filters["source_type"] = "webpage"
    if "table" in q or "highest" in q or "row" in q:
        filters["source_type"] = "table"
    version = re.search(r"\bv(?:ersion)?\s*([0-9]+(?:\.[0-9]+)*)", q)
    if version:
        filters["version"] = version.group(1)
    if "latest" in q or "current" in q:
        filters["is_current"] = True
    return filters


def apply_filters(docs: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    if not filters:
        return docs
    out = []
    for doc in docs:
        metadata = doc.get("metadata", {})
        ok = True
        for key, value in filters.items():
            if metadata.get(key) != value:
                ok = False
                break
        if ok:
            out.append(doc)
    return out


def capitalized_entities(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", text or ""))


def build_entity_graph(docs: list[dict[str, Any]]) -> dict[str, set[str]]:
    graph: defaultdict[str, set[str]] = defaultdict(set)
    for doc in docs:
        entities = set(doc.get("metadata", {}).get("entities", [])) or capitalized_entities(doc_text(doc))
        for entity in entities:
            graph[entity].add(doc["id"])
        for left in entities:
            for right in entities:
                if left != right:
                    graph[left].add(right)
    return graph


def parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.min
    value = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min


def dedupe_by_hash(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for doc in sorted(
        docs,
        key=lambda d: parse_time(d.get("metadata", {}).get("retrieved_at")),
        reverse=True,
    ):
        text_hash = doc.get("metadata", {}).get("content_hash")
        if not text_hash:
            text_hash = hashlib.sha256(doc_text(doc).encode("utf-8")).hexdigest()
        if text_hash in seen:
            continue
        seen.add(text_hash)
        out.append(doc)
    return out


def call_demo_tool(payload: dict[str, Any], query: str) -> dict[str, Any]:
    tools = payload.get("tools", {})
    if "order" in query.lower() and "order_status" in tools:
        result = tools["order_status"]
        return {"tool": "order_status", "arguments": {"order_id": result.get("order_id")}, "result": result}
    return {"tool": "none", "arguments": {}, "result": "No tool call needed for this query."}


def run_inference_method(method_key: str, payload: dict[str, Any], query: str, top_k: int) -> dict[str, Any]:
    """Run one inference-time method branch.

    Each branch is written as a small lesson:

    1. Start from the same local corpus.
    2. Change exactly the control-flow idea that defines the method.
    3. Return `steps` so the learner can see what happened.
    4. Return `top_evidence` so the learner can inspect ranking behavior.

    The branches are deliberately explicit instead of abstracted behind classes,
    because the goal is to make the method mechanics easy to read.
    """

    docs = payload["documents"]
    steps: list[str] = []

    if method_key == "hybrid":
        evidence = hybrid_retrieve(query, docs, top_k)
        steps.append("Retrieved with BM25 and TF-IDF cosine, then fused rankings with reciprocal rank fusion.")

    elif method_key == "reranked":
        candidates = hybrid_retrieve(query, docs, top_k=max(top_k * 3, 10))
        evidence = rerank(query, candidates, top_k)
        steps.append("Retrieved a broad candidate set and reranked query-document pairs.")

    elif method_key == "agentic":
        subqueries = [part.strip() for part in re.split(r"\band\b|,|\?", query) if part.strip()]
        if not subqueries:
            subqueries = [query]
        collected = []
        for subquery in subqueries[:4]:
            collected.extend(hybrid_retrieve(subquery, docs, 2))
        tool_result = call_demo_tool(payload, query)
        evidence = rerank(query, list({d["id"]: d for d in collected}.values()), top_k)
        steps.append(f"Planned {len(subqueries)} retrieval step(s) and tool action {tool_result['tool']}.")

    elif method_key == "managed_enterprise":
        visible = [d for d in docs if "public" in d.get("metadata", {}).get("acl", [])]
        evidence = hybrid_retrieve(query, visible, top_k)
        steps.append("Applied simulated enterprise ACL trimming before managed hybrid retrieval.")

    elif method_key == "graphrag":
        graph = build_entity_graph(docs)
        q_entities = capitalized_entities(query)
        candidate_ids: set[str] = set()
        for entity in q_entities:
            candidate_ids.update(node for node in graph.get(entity, set()) if node.startswith("doc_") or "_" in node)
        candidates = [d for d in docs if d["id"] in candidate_ids] or docs
        evidence = hybrid_retrieve(query, candidates, top_k)
        steps.append(f"Built an entity graph with {len(graph)} nodes and retrieved related source neighborhoods.")

    elif method_key == "raptor":
        groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for doc in docs:
            groups[doc.get("metadata", {}).get("section", "General")].append(doc)
        summaries = [
            {
                "id": f"summary_{section}",
                "title": f"Summary: {section}",
                "text": " ".join(d["text"] for d in group)[:800],
                "metadata": {"source_type": "summary", "section": section, "children": [d["id"] for d in group]},
            }
            for section, group in groups.items()
        ]
        summary_hits = hybrid_retrieve(query, summaries, min(2, len(summaries)))
        child_ids = {child for s in summary_hits for child in s.get("metadata", {}).get("children", [])}
        evidence = hybrid_retrieve(query, [d for d in docs if d["id"] in child_ids] or docs, top_k)
        steps.append("Retrieved hierarchy summaries first, then expanded to leaf chunks.")

    elif method_key == "self_rag":
        needs_retrieval = any(word in query.lower() for word in ["policy", "current", "version", "cite", "where", "table"])
        evidence = hybrid_retrieve(query, docs, top_k) if needs_retrieval else []
        steps.append(f"Adaptive retrieval decision: {'retrieve' if needs_retrieval else 'skip retrieval'}.")

    elif method_key == "crag":
        first = hybrid_retrieve(query, docs, top_k)
        quality = first[0]["score"] if first else 0.0
        if quality < 0.02:
            rewritten = f"{query} policy current answer"
            evidence = hybrid_retrieve(rewritten, docs, top_k)
            steps.append("Initial evidence was weak, so the corrective branch rewrote the query and retrieved again.")
        else:
            evidence = first
            steps.append("Evidence grader accepted the initial retrieval set.")

    elif method_key == "query_rewriting":
        rewrites = [
            query,
            query.replace("rollback", "restore previous version"),
            f"current policy {query}",
        ]
        collected = []
        for rewrite in rewrites:
            collected.extend(hybrid_retrieve(rewrite, docs, top_k))
        evidence = rerank(query, list({d["id"]: d for d in collected}.values()), top_k)
        steps.append(f"Generated {len(rewrites)} retrieval rewrites and fused their candidates.")

    elif method_key == "multi_query":
        variants = [query, f"{query} eligibility", f"{query} version", f"{query} table"]
        collected = []
        for variant in variants:
            collected.extend(hybrid_retrieve(variant, docs, top_k))
        evidence = rerank(query, list({d["id"]: d for d in collected}.values()), top_k)
        steps.append(f"Ran {len(variants)} parallel query variants and deduplicated results.")

    elif method_key == "decomposition":
        subquestions = [q.strip() for q in re.split(r"\band\b|compare|versus|vs", query, flags=re.I) if q.strip()]
        collected = []
        for subquestion in subquestions or [query]:
            collected.extend(hybrid_retrieve(subquestion, docs, 2))
        evidence = rerank(query, list({d["id"]: d for d in collected}.values()), top_k)
        steps.append(f"Decomposed into {len(subquestions or [query])} sub-question(s).")

    elif method_key == "metadata_filtered":
        filters = infer_filters(query)
        filtered = apply_filters(docs, filters)
        evidence = hybrid_retrieve(query, filtered or docs, top_k)
        steps.append(f"Extracted metadata filters {filters or '{}'} before retrieval.")

    elif method_key == "parent_child":
        child_hits = hybrid_retrieve(query, docs, top_k)
        parent_ids = {d.get("metadata", {}).get("parent_id") for d in child_hits if d.get("metadata", {}).get("parent_id")}
        parents = [d for d in docs if d["id"] in parent_ids]
        evidence = parents + child_hits
        evidence = rerank(query, evidence, top_k)
        steps.append("Retrieved child chunks, then expanded hits to parent sections.")

    elif method_key == "contextual_compression":
        retrieved = hybrid_retrieve(query, docs, top_k)
        evidence = [compress_doc(query, doc) for doc in retrieved]
        steps.append("Compressed retrieved chunks to query-relevant sentences while preserving source IDs.")

    elif method_key == "table_aware":
        tables = [d for d in docs if d.get("metadata", {}).get("source_type") == "table"]
        evidence = hybrid_retrieve(query, tables or docs, top_k)
        if "highest" in query.lower() and evidence:
            rows = evidence[0].get("metadata", {}).get("table_rows", [])
            if rows:
                best = max(rows, key=lambda r: float(r.get("warranty_reserve_usd", 0)))
                steps.append(f"Executed table aggregation: highest warranty reserve is {best}.")
        steps.append("Retrieved table records with headers, rows, units, and table provenance.")

    elif method_key == "layout_pdf":
        pdf_docs = [d for d in docs if d.get("metadata", {}).get("source_type") in {"pdf", "figure", "table"}]
        evidence = hybrid_retrieve(query, pdf_docs or docs, top_k)
        steps.append("Searched layout-aware PDF elements including pages, bounding boxes, captions, and OCR confidence.")

    elif method_key == "webpage_freshness":
        web_docs = [d for d in docs if d.get("metadata", {}).get("source_type") == "webpage"]
        fresh = dedupe_by_hash(web_docs)
        evidence = hybrid_retrieve(query, fresh or docs, top_k)
        steps.append("Deduplicated by content hash and preferred newest crawled canonical pages.")

    elif method_key == "version_aware":
        filters = infer_filters(query)
        if "version" not in filters and filters.get("is_current"):
            filtered = [d for d in docs if d.get("metadata", {}).get("is_current")]
        else:
            filtered = apply_filters(docs, filters)
        evidence = hybrid_retrieve(query, filtered or docs, top_k)
        steps.append(f"Applied version/freshness scope {filters or {'is_current': True}} before retrieval.")

    elif method_key == "tool_augmented":
        policy = hybrid_retrieve(query, docs, top_k)
        tool_result = call_demo_tool(payload, query)
        evidence = policy
        steps.append(f"Retrieved policy context and executed simulated tool call: {tool_result}.")

    elif method_key == "claim_verification":
        evidence = hybrid_retrieve(query, docs, top_k)
        claims = [
            "Vendor onboarding requires a security review.",
            "The current API documentation is version 3.0.",
            "Warranty reserve values can be verified from the table.",
        ]
        support = {
            claim: any(set(tokenize(claim)).intersection(tokenize(doc_text(doc))) for doc in evidence)
            for claim in claims
        }
        steps.append(f"Extracted and verified atomic claims: {support}.")

    else:
        evidence = hybrid_retrieve(query, docs, top_k)
        steps.append("Used the default hybrid demo path.")

    return {
        "method_key": method_key,
        "query": query,
        "steps": steps,
        "top_evidence": evidence_summary(evidence[:top_k]),
        "answer": answer_from_evidence(query, evidence, f"Demo answer for {method_key}"),
    }


def make_synthetic_query(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata", {})
    section = metadata.get("section", "this source")
    title = doc.get("title", "the document")
    return f"What does {title} say about {section}?"


def hard_negatives(query: str, docs: list[dict[str, Any]], positive_id: str, n: int = 2) -> list[str]:
    ranked = hybrid_retrieve(query, [d for d in docs if d["id"] != positive_id], top_k=n)
    return [d["id"] for d in ranked]


def run_training_method(method_key: str, payload: dict[str, Any], query: str, top_k: int) -> dict[str, Any]:
    """Run one training-time RAG method branch.

    The output is not a trained model. It is the artifact that a real training
    pipeline would feed into a trainer: neighbor chunks, positives, hard
    negatives, citation examples, preference pairs, tool traces, or evaluation
    records.
    """

    docs = payload["documents"]
    examples = payload.get("training_examples", [])
    artifacts: list[dict[str, Any]] = []
    steps: list[str] = []

    if method_key == "retro":
        for doc in docs[:top_k]:
            neighbors = [d["id"] for d in hybrid_retrieve(doc.get("text", ""), docs, 3) if d["id"] != doc["id"]]
            artifacts.append({"chunk": doc["id"], "retrieved_neighbors": neighbors})
        steps.append("Created RETRO-style neighbor chunks for language-model training examples.")

    elif method_key == "realm":
        for ex in examples:
            retrieved = hybrid_retrieve(ex["query"], docs, top_k)
            artifacts.append({"query": ex["query"], "positive": ex["positive_id"], "retrieved": [d["id"] for d in retrieved]})
        steps.append("Simulated REALM-style retriever supervision and marginal evidence candidates.")

    elif method_key == "atlas":
        for ex in examples:
            retrieved = hybrid_retrieve(ex["query"], docs, top_k)
            artifacts.append({"input": ex["query"], "fusion_in_decoder_context": [d["id"] for d in retrieved], "target": ex["answer"]})
        steps.append("Prepared Atlas-style question, retrieved passages, and target answer triples.")

    elif method_key == "original_rag_finetuning":
        for ex in examples:
            retrieved = hybrid_retrieve(ex["query"], docs, top_k)
            marginal = sum(d["score"] for d in retrieved)
            artifacts.append({"query": ex["query"], "answer": ex["answer"], "marginal_retrieval_score": round(marginal, 4)})
        steps.append("Computed toy RAG-sequence marginal scores over retrieved passages.")

    elif method_key == "self_rag_training":
        for ex in examples:
            needs_retrieval = "policy" in ex["query"].lower() or "current" in ex["query"].lower()
            artifacts.append({"query": ex["query"], "reflection_tokens": {"retrieve": needs_retrieval, "support": True}, "answer": ex["answer"]})
        steps.append("Generated Self-RAG reflection-token supervision.")

    elif method_key == "rag_grounded_sft":
        for doc in docs[:top_k]:
            artifacts.append({"instruction": make_synthetic_query(doc), "evidence_id": doc["id"], "response": f"Use source {doc['id']}: {doc['text'][:160]}"})
        steps.append("Created grounded SFT examples with explicit evidence IDs.")

    elif method_key == "grounded_preference":
        for ex in examples:
            artifacts.append({"prompt": ex["query"], "chosen": f"{ex['answer']} [cite:{ex['positive_id']}]", "rejected": "A plausible but uncited answer."})
        steps.append("Created chosen/rejected groundedness preference pairs.")

    elif method_key == "retriever_hard_negative":
        for ex in examples:
            artifacts.append({"query": ex["query"], "positive": ex["positive_id"], "hard_negatives": hard_negatives(ex["query"], docs, ex["positive_id"])})
        steps.append("Mined hard negatives with the current hybrid retriever.")

    elif method_key == "reranker_finetuning":
        for ex in examples:
            negatives = hard_negatives(ex["query"], docs, ex["positive_id"])
            artifacts.append({"query": ex["query"], "positive_pair": [ex["query"], ex["positive_id"]], "negative_pairs": [[ex["query"], n] for n in negatives]})
        steps.append("Prepared pairwise reranker fine-tuning examples.")

    elif method_key == "citation_instruction":
        for ex in examples:
            artifacts.append({"instruction": ex["query"], "answer_with_citation": f"{ex['answer']} [source:{ex['positive_id']}]", "citation_required": True})
        steps.append("Created citation-supervised instruction examples.")

    elif method_key == "synthetic_query":
        for doc in docs[:top_k]:
            artifacts.append({"synthetic_query": make_synthetic_query(doc), "positive_id": doc["id"]})
        steps.append("Generated document-grounded synthetic query-positive pairs.")

    elif method_key == "multihop_synthetic":
        graph = build_entity_graph(docs)
        entities = list(graph)[:3]
        artifacts.append({"question": f"How are {', '.join(entities)} connected?", "evidence_path": entities, "source_ids": sorted({x for e in entities for x in graph[e] if '_' in x})[:top_k]})
        steps.append("Generated a multi-hop question from entity graph connections.")

    elif method_key == "graphrag_training":
        graph = build_entity_graph(docs)
        for entity, neighbors in list(graph.items())[:top_k]:
            artifacts.append({"entity": entity, "neighbors": sorted(neighbors)[:5], "training_prompt": f"Summarize evidence about {entity} with citations."})
        steps.append("Created graph-neighborhood training prompts.")

    elif method_key == "raptor_training":
        groups: defaultdict[str, list[str]] = defaultdict(list)
        for doc in docs:
            groups[doc.get("metadata", {}).get("section", "General")].append(doc["id"])
        for section, children in groups.items():
            artifacts.append({"summary_node": section, "children": children, "question": f"What is the main point of {section}?"})
        steps.append("Created level-aware hierarchy examples from section groups.")

    elif method_key == "table_synthetic":
        for doc in docs:
            rows = doc.get("metadata", {}).get("table_rows", [])
            if rows:
                best = max(rows, key=lambda r: float(r.get("warranty_reserve_usd", 0)))
                artifacts.append({"question": "Which SKU has the highest warranty reserve?", "answer": best, "table_id": doc["id"]})
        steps.append("Generated table QA examples with executable answers.")

    elif method_key == "web_freshness_training":
        web_docs = [d for d in docs if d.get("metadata", {}).get("source_type") == "webpage"]
        for doc in web_docs:
            artifacts.append({"url": doc.get("metadata", {}).get("canonical_url"), "is_current": doc.get("metadata", {}).get("is_current"), "retrieved_at": doc.get("metadata", {}).get("retrieved_at")})
        steps.append("Built current-vs-stale webpage training labels.")

    elif method_key == "version_aware_training":
        for doc in docs:
            version = doc.get("metadata", {}).get("version")
            if version:
                artifacts.append({"query": f"What is true in version {version}?", "positive_id": doc["id"], "version": version, "is_current": doc.get("metadata", {}).get("is_current")})
        steps.append("Created version-scoped positives and stale negatives.")

    elif method_key == "tool_use_training":
        artifacts.append({"trace": [{"action": "retrieve_policy", "query": query}, {"action": "call_tool", "tool": "order_status", "args": {"order_id": "A100"}}, {"action": "answer_with_policy_and_tool_result"}]})
        steps.append("Generated a safe tool-use trace with retrieval before API use.")

    elif method_key == "evaluation_bootstrapping":
        for ex in examples:
            artifacts.append({"query": ex["query"], "expected_source": ex["positive_id"], "rubric": "answer must cite expected_source and abstain if not retrieved"})
        steps.append("Bootstrapped a small RAG evaluation set.")

    elif method_key == "continual_memory":
        current_ids = {d["id"] for d in docs}
        for update in payload.get("updates", []):
            artifacts.append({"operation": "upsert" if update["id"] not in current_ids else "replace", "id": update["id"], "requires_reindex": True})
        steps.append("Produced a continual-memory update plan with reindex flags.")

    else:
        for doc in docs[:top_k]:
            artifacts.append({"query": make_synthetic_query(doc), "positive_id": doc["id"]})
        steps.append("Created generic training artifacts.")

    return {
        "method_key": method_key,
        "query": query,
        "steps": steps,
        "artifacts": artifacts[: max(top_k, 5)],
        "notes": "This is a runnable teaching demo. Replace the toy scorer/data with production parsers, indexes, models, and evaluators.",
    }


def run_cli(method: dict[str, Any]) -> None:
    parser = argparse.ArgumentParser(description=method["description"])
    parser.add_argument("--corpus", default=method.get("default_corpus", str(Path("examples") / "sample_corpus.json")))
    parser.add_argument("--query", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()

    payload = load_payload(args.corpus)
    query = args.query or payload.get("query") or "What evidence is relevant?"
    if args.explain:
        result = {
            "method": method["name"],
            "script": method["script"],
            "description": method["description"],
            "teaching_steps": method.get("teaching_steps", []),
            "pedagogical_simplifications": [
                "Uses a tiny JSON corpus instead of a production document store.",
                "Uses BM25 and TF-IDF cosine instead of external search engines or embedding APIs.",
                "Uses deterministic toy scoring so the control flow is inspectable.",
                "Returns JSON traces instead of hiding intermediate retrieval decisions.",
            ],
            "how_to_run": f"python {method['script']} --corpus examples/sample_corpus.json --query \"{query}\"",
        }
    elif method["mode"] == "training":
        result = run_training_method(method["key"], payload, query, args.top_k)
        result["method"] = method["name"]
    else:
        result = run_inference_method(method["key"], payload, query, args.top_k)
        result["method"] = method["name"]
    print(json.dumps(result, indent=2, ensure_ascii=False))



SCRIPT_DIR = Path(__file__).resolve().parent


METHOD = {
    "name": 'Decomposition RAG / sub-question RAG',
    "key": 'decomposition',
    "mode": 'inference',
    "script": 'decomposition_rag.py',
    "description": 'Split a complex question into sub-questions and retrieve evidence per part.',
    "default_corpus": str(SCRIPT_DIR / "examples" / "sample_corpus.json"),
    "teaching_steps": [
        "Open examples/sample_corpus.json and inspect the documents, metadata, training examples, and tool payload.",
        "Run this script with --explain to read the method intent and simplifications.",
        "Run examples/run_example.py to execute the local sample query.",
        "Read the JSON `steps` field first; it explains the method control flow.",
        "Then inspect `top_evidence` or `artifacts` to see what the method selected or produced.",
        "Replace the toy scorer with the production component described in this folder's ARCHITECTURE.md.",
    ],
}


if __name__ == "__main__":
    run_cli(METHOD)
