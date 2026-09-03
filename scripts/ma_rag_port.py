"""Faithful, isolated MA-RAG-style multi-round controller for TRACE-o1 experiments.

This file intentionally does not modify the existing Search-o1 / TRACE-o1 pipeline.
It ports the core loop from NJU-RL/MA-RAG into a reusable controller:

  sample N candidates -> stop on consensus -> extract conflict queries -> retrieve
  -> rank previous answers -> next round with retrieved documents and history.

Stage-aligned conflict routing is deliberately NOT implemented here; it belongs in
later ablations so this module can serve as the vanilla MA-RAG baseline.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


QUERY_RE = re.compile(r"\[Query\s*\d*\](.*?)(?=\n|$)", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>", re.IGNORECASE)
FINAL_CHOICE_RE = re.compile(r"(?:final\s+answer|answer)\s*[:：]?\s*\(?([A-Za-z])\)?", re.IGNORECASE)


@dataclass
class Candidate:
    text: str
    answer: str
    # For the entropy variant this field stores mean token entropy, i.e. an
    # uncertainty score: lower entropy means higher confidence.
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundRecord:
    round_id: int
    candidates: List[Candidate]
    predictions: List[str]
    consensus: bool
    consensus_answer: str
    disagreement: float
    conflict_queries: List[str] = field(default_factory=list)
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    ranked_candidate_indices: List[int] = field(default_factory=list)
    entropy_order_mode: str = "none"


@dataclass
class MARAGResult:
    final_answer: str
    final_text: str
    stopped_reason: str
    rounds: List[RoundRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "final_text": self.final_text,
            "stopped_reason": self.stopped_reason,
            "rounds": [asdict(r) for r in self.rounds],
        }


class MARAGController:
    """Vanilla conflict-to-consensus controller.

    Required callbacks
    ------------------
    generate_fn(prompt, n, round_id) -> list[str | dict]
        Generate independent solver answers. A dict may contain ``text``,
        ``confidence`` (mean token entropy in the entropy variant), and metadata.
    retrieve_fn(query, top_k) -> list[dict]
        Retrieve evidence for one conflict-derived query.

    Optional callbacks
    ------------------
    query_generate_fn(question, options, answers) -> str | list[str]
        If omitted, a deterministic lexical fallback is used. For a faithful
        MA-RAG experiment, pass an LLM-backed implementation using
        ``build_conflict_query_prompt``.
    confidence_fn(text) -> float
        Fallback uncertainty callback. Lower values should mean higher confidence.

    Entropy order modes
    -------------------
    official_code
        Match the released ``ma_rag_entropy.py`` implementation: previous answers
        are ordered from HIGH mean token entropy to LOW mean token entropy.
    low_first
        Follow the semantic interpretation stated in the upstream prompt: lower
        entropy = higher confidence, so previous answers are ordered LOW to HIGH.
    none
        Preserve candidate generation order.
    """

    VALID_ENTROPY_ORDERS = {"official_code", "low_first", "none"}

    def __init__(
        self,
        generate_fn: Callable[[str, int, int], Sequence[Any]],
        retrieve_fn: Callable[[str, int], Sequence[Dict[str, Any]]],
        query_generate_fn: Optional[Callable[[str, str, Sequence[str]], Any]] = None,
        confidence_fn: Optional[Callable[[str], float]] = None,
        num_workers: int = 5,
        num_rounds: int = 5,
        queries_per_conflict: int = 4,
        docs_per_query: int = 2,
        entropy_order_mode: str = "official_code",
    ) -> None:
        if num_workers < 2:
            raise ValueError("num_workers must be >= 2 for conflict detection")
        if num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")
        if entropy_order_mode not in self.VALID_ENTROPY_ORDERS:
            raise ValueError(f"entropy_order_mode must be one of {sorted(self.VALID_ENTROPY_ORDERS)}")
        self.generate_fn = generate_fn
        self.retrieve_fn = retrieve_fn
        self.query_generate_fn = query_generate_fn
        self.confidence_fn = confidence_fn
        self.num_workers = num_workers
        self.num_rounds = num_rounds
        self.queries_per_conflict = queries_per_conflict
        self.docs_per_query = docs_per_query
        self.entropy_order_mode = entropy_order_mode

    @staticmethod
    def extract_answer(text: str) -> str:
        text = text or ""
        tagged = ANSWER_TAG_RE.findall(text)
        if tagged:
            return tagged[-1].upper()
        boxed = BOXED_RE.findall(text)
        if boxed:
            return boxed[-1].strip()
        final = FINAL_CHOICE_RE.findall(text)
        if final:
            return final[-1].upper()
        return ""

    @staticmethod
    def disagreement(predictions: Sequence[str]) -> float:
        valid = [p for p in predictions if p]
        if not valid:
            return 1.0
        majority = Counter(valid).most_common(1)[0][1]
        return 1.0 - majority / len(valid)

    @staticmethod
    def consensus(predictions: Sequence[str]) -> Tuple[bool, str]:
        valid = [p for p in predictions if p]
        if not valid:
            return False, ""
        unique = set(valid)
        return len(unique) == 1 and len(valid) == len(predictions), valid[0] if len(unique) == 1 else ""

    @staticmethod
    def _dedupe_queries(queries: Sequence[str], limit: int) -> List[str]:
        out: List[str] = []
        seen = set()
        for q in queries:
            q = re.sub(r"\s+", " ", (q or "").strip())
            key = q.lower()
            if q and key not in seen:
                seen.add(key)
                out.append(q)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def parse_queries(raw: Any, limit: int = 4) -> List[str]:
        if isinstance(raw, (list, tuple)):
            return MARAGController._dedupe_queries([str(x) for x in raw], limit)
        text = str(raw or "")
        matches = [m.strip() for m in QUERY_RE.findall(text) if m.strip()]
        if not matches:
            matches = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
        return MARAGController._dedupe_queries(matches, limit)

    @staticmethod
    def build_initial_prompt(question: str, options: str = "") -> str:
        option_block = f"\n### Options\n{options}\n" if options else ""
        return (
            "Below is a question.\n"
            f"### Question\n{question}\n{option_block}\n"
            "Analyze the question carefully. End with a concise final answer; for multiple-choice "
            "questions use <answer>X</answer>."
        )

    @staticmethod
    def build_round_prompt(question: str, options: str, documents: str, answers: Sequence[str]) -> str:
        option_block = f"\n### Options\n{options}\n" if options else ""
        answer_block = "\n\n".join(
            f"{i}. Previous assistant answer:\n{answer}" for i, answer in enumerate(answers, start=1)
        )
        return (
            "Below is a question.\n"
            f"### Question\n{question}\n{option_block}"
            f"\n### Documents\n{documents or 'null'}\n\n"
            "Several previous answers may be incorrect. Compare their reasoning, use the retrieved "
            "documents to resolve their disagreements, and answer the question again.\n\n"
            f"{answer_block}\n\n"
            "End with a concise final answer; for multiple-choice questions use <answer>X</answer>."
        )

    @staticmethod
    def build_conflict_query_prompt(question: str, options: str, answers: Sequence[str]) -> str:
        option_block = f"\n### Options\n{options}\n" if options else ""
        answer_block = "\n\n".join(
            f"{i}. Assistant answer:\n{answer}" for i, answer in enumerate(answers, start=1)
        )
        return (
            "You are a query-generation expert. Identify contradictions, ambiguities, and core dispute "
            "points among the candidate answers, then generate 1-4 precise search queries that would "
            "verify those disputed points.\n\n"
            f"### Question\n{question}\n{option_block}\n{answer_block}\n\n"
            "Output only:\n[Query 1] ...\n[Query 2] ..."
        )

    @staticmethod
    def combine_documents(documents: Sequence[Dict[str, Any]]) -> str:
        chunks: List[str] = []
        for i, doc in enumerate(documents, start=1):
            title = str(doc.get("title", "")).strip()
            content = str(doc.get("content", doc.get("snippet", doc.get("text", "")))).strip()
            url = str(doc.get("url", "")).strip()
            header = f"Document [{i}]"
            if title:
                header += f" (Title: {title})"
            if url:
                header += f" [URL: {url}]"
            chunks.append(f"{header}\n{content}")
        return "\n\n".join(chunks)

    def _to_candidate(self, obj: Any) -> Candidate:
        if isinstance(obj, str):
            text, metadata, confidence = obj, {}, None
        elif isinstance(obj, dict):
            text = str(obj.get("text", obj.get("response", "")))
            confidence = obj.get("confidence")
            metadata = {k: v for k, v in obj.items() if k not in {"text", "response", "confidence"}}
        else:
            text, metadata, confidence = str(obj), {}, None
        if confidence is None and self.confidence_fn is not None:
            confidence = float(self.confidence_fn(text))
        return Candidate(text=text, answer=self.extract_answer(text), confidence=confidence, metadata=metadata)

    def _rank_candidates(self, candidates: Sequence[Candidate]) -> List[Tuple[int, Candidate]]:
        indexed = list(enumerate(candidates))
        if self.entropy_order_mode == "none" or not any(c.confidence is not None for c in candidates):
            return indexed

        if self.entropy_order_mode == "official_code":
            # Released MA-RAG code uses np.argsort(entropies)[::-1]: high -> low.
            return sorted(
                indexed,
                key=lambda x: -math.inf if x[1].confidence is None else x[1].confidence,
                reverse=True,
            )

        # Prompt semantics: low entropy = high confidence.
        return sorted(
            indexed,
            key=lambda x: math.inf if x[1].confidence is None else x[1].confidence,
        )

    def _rank_answers(self, candidates: Sequence[Candidate]) -> Tuple[List[str], List[int]]:
        ranked = self._rank_candidates(candidates)
        return [c.text for _, c in ranked], [idx for idx, _ in ranked]

    def _fallback_queries(self, question: str, candidates: Sequence[Candidate]) -> List[str]:
        answers = [c.answer for c in candidates if c.answer]
        suffix = " vs ".join(dict.fromkeys(answers)) if answers else "candidate disagreement"
        return [f"{question} {suffix}"]

    def _generate_queries(self, question: str, options: str, candidates: Sequence[Candidate]) -> List[str]:
        if self.query_generate_fn is None:
            return self._fallback_queries(question, candidates)[: self.queries_per_conflict]
        raw = self.query_generate_fn(question, options, [c.text for c in candidates])
        return self.parse_queries(raw, self.queries_per_conflict)

    def _retrieve(self, queries: Sequence[str]) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        seen = set()
        for query in queries:
            for doc in self.retrieve_fn(query, self.docs_per_query) or []:
                key = str(doc.get("id", doc.get("url", json.dumps(doc, sort_keys=True, ensure_ascii=False))))
                if key in seen:
                    continue
                seen.add(key)
                copied = dict(doc)
                copied.setdefault("query", query)
                docs.append(copied)
        return docs

    def run(self, question: str, options: str = "") -> MARAGResult:
        rounds: List[RoundRecord] = []
        previous_answers: List[str] = []
        documents: List[Dict[str, Any]] = []
        last_candidates: List[Candidate] = []

        for round_id in range(1, self.num_rounds + 1):
            if round_id == 1:
                prompt = self.build_initial_prompt(question, options)
            else:
                prompt = self.build_round_prompt(
                    question,
                    options,
                    self.combine_documents(documents),
                    previous_answers,
                )

            raw_candidates = self.generate_fn(prompt, self.num_workers, round_id)
            candidates = [self._to_candidate(x) for x in raw_candidates]
            predictions = [c.answer for c in candidates]
            converged, consensus_answer = self.consensus(predictions)
            record = RoundRecord(
                round_id=round_id,
                candidates=candidates,
                predictions=predictions,
                consensus=converged,
                consensus_answer=consensus_answer,
                disagreement=self.disagreement(predictions),
                entropy_order_mode=self.entropy_order_mode,
            )
            rounds.append(record)
            last_candidates = candidates

            if converged:
                winner = candidates[0]
                return MARAGResult(winner.answer, winner.text, "consensus", rounds)

            if round_id == self.num_rounds:
                break

            queries = self._generate_queries(question, options, candidates)
            documents = self._retrieve(queries)
            record.conflict_queries = queries
            record.retrieved_documents = documents
            previous_answers, record.ranked_candidate_indices = self._rank_answers(candidates)

        valid_answers = [c.answer for c in last_candidates if c.answer]
        majority = Counter(valid_answers).most_common(1)[0][0] if valid_answers else ""
        winner = next((c for c in last_candidates if c.answer == majority), last_candidates[0] if last_candidates else Candidate("", ""))
        return MARAGResult(majority, winner.text, "round_budget", rounds)
