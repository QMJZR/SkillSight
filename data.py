"""Data loading helpers for SR-Agents-style SkillSight evaluation."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


def skill_text(skill: dict) -> str:
    parts: List[str] = []
    for key in ("name", "description", "content"):
        value = skill.get(key)
        if value:
            parts.append(str(value))
    return "\n".join(parts)


def load_instances(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("instances", "data", "examples", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Cannot parse instances from {path}")


def get_question(inst: dict) -> str:
    for key in ("question", "query", "problem", "prompt", "instruction"):
        value = inst.get(key)
        if value:
            return str(value)
    return ""


def get_gold(inst: dict) -> List[str]:
    gold = inst.get("skill_annotations") or []
    return [str(x) for x in gold]


@dataclass
class QueryExample:
    dataset: str
    query: str
    gold_ids: List[str]
    instance_index: int


def collect_queries(dataset: str, instances: Sequence[dict]) -> List[QueryExample]:
    examples: List[QueryExample] = []
    for i, inst in enumerate(instances):
        q = get_question(inst)
        gold = get_gold(inst)
        if q and gold:
            examples.append(QueryExample(dataset=dataset, query=q, gold_ids=gold, instance_index=i))
    return examples


# --- SkillRouter-Eval-Core loaders -------------------------------------------
# The "all" pool is the deduplicated union of the easy and hard skill tiers
# (hard is a superset of easy); queries/gold come from tasks.jsonl + relevance.json.


def skillrouter_skill_text(skill: dict) -> str:
    return "\n".join(str(skill[k]) for k in ("name", "description", "body") if skill.get(k))


def load_skillrouter_corpus(root: Path, tiers: Sequence[str] = ("easy", "hard")) -> Tuple[List[str], List[str]]:
    """Load the deduplicated union of the given SkillRouter tiers."""
    texts: List[str] = []
    ids: List[str] = []
    seen: set[str] = set()
    for tier in tiers:
        shards = sorted((root / tier).glob("part-*.jsonl.gz"))
        for shard in shards:
            with gzip.open(shard, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    s = json.loads(line)
                    sid = str(s["skill_id"])
                    if sid in seen:
                        continue
                    seen.add(sid)
                    ids.append(sid)
                    texts.append(skillrouter_skill_text(s))
    return texts, ids


def load_skillrouter_examples(
    root: Path,
    exclude_generic_only: bool = True,
    include_excluded: bool = False,
) -> List[QueryExample]:
    tasks = [json.loads(l) for l in (root / "tasks.jsonl").read_text().splitlines() if l.strip()]
    rel = json.loads((root / "relevance.json").read_text())
    examples: List[QueryExample] = []
    for i, t in enumerate(tasks):
        tid = t["task_id"]
        if t.get("excluded", False) and not include_excluded:
            continue
        r = rel.get(tid)
        if r is None:
            continue
        if exclude_generic_only and r.get("task_type") == "generic_only":
            continue
        gold = [str(x) for x in r.get("gt_skill_ids", [])]
        if not gold or not t.get("instruction_text"):
            continue
        examples.append(
            QueryExample(
                dataset="skillrouter_core",
                query=str(t["instruction_text"]),
                gold_ids=gold,
                instance_index=i,
            )
        )
    return examples
