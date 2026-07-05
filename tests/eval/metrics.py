"""Retrieval evaluation metrics and gold-set loading (backend-agnostic)."""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

GOLD_PATH = Path(__file__).parent / "gold_queries.yaml"

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


@dataclass(frozen=True)
class GoldQuery:
    id: str
    question: str
    expected_any: List[str]


def load_gold(path: Path = GOLD_PATH) -> List[GoldQuery]:
    data = yaml.safe_load(path.read_text())
    return [
        GoldQuery(id=q["id"], question=q["question"], expected_any=q["expected_any"])
        for q in data["queries"]
    ]


def first_hit_rank(results: List[Dict], expected_any: List[str]) -> Optional[int]:
    """0-indexed rank of the first result matching any expected substring, else None."""
    needles = [_normalize(e) for e in expected_any]
    for rank, doc in enumerate(results):
        content = _normalize(doc["content"])
        if any(needle in content for needle in needles):
            return rank
    return None


@dataclass
class QueryResult:
    id: str
    rank: Optional[int]  # 0-indexed first-hit rank, None if not retrieved


@dataclass
class EvalReport:
    per_query: List[QueryResult]

    def recall_at(self, k: int) -> float:
        hits = sum(1 for r in self.per_query if r.rank is not None and r.rank < k)
        return hits / len(self.per_query)

    def mrr(self) -> float:
        total = 0.0
        for r in self.per_query:
            if r.rank is not None:
                total += 1.0 / (r.rank + 1)
        return total / len(self.per_query)

    def format(self, name: str, ks=(1, 3, 5, 10, 12)) -> str:
        lines = [f"=== {name} ==="]
        for r in self.per_query:
            rank = "MISS" if r.rank is None else f"#{r.rank}"
            lines.append(f"  {r.id:<24} {rank}")
        recalls = "  ".join(f"R@{k}={self.recall_at(k):.2f}" for k in ks)
        lines.append(f"  {recalls}  MRR={self.mrr():.3f}")
        return "\n".join(lines)


def evaluate(
    gold: List[GoldQuery],
    retrieve: Callable[[str], List[Dict]],
) -> EvalReport:
    """Run `retrieve` for each gold query and score the first-hit rank."""
    results = [
        QueryResult(id=g.id, rank=first_hit_rank(retrieve(g.question), g.expected_any))
        for g in gold
    ]
    return EvalReport(per_query=results)
