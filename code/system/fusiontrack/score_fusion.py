from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

from fusiontrack.event_segments import event_segments_from_frame_scores, normalize_frame_event_scores


EPSILON = 1e-6


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def robust_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    med = median(values)
    mad = median([abs(value - med) for value in values])
    if mad <= EPSILON:
        ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]))
        denom = max(len(ordered) - 1, 1)
        return {sample_id: idx / denom for idx, (sample_id, _) in enumerate(ordered)}
    return {
        sample_id: max(0.0, (value - med) / (1.4826 * mad + EPSILON))
        for sample_id, value in scores.items()
    }


def _prefix_components(prefix: str, components: dict[str, Any]) -> dict[str, float]:
    prefixed = {}
    for key, value in components.items():
        try:
            prefixed[f"{prefix}_{key}"] = float(value)
        except (TypeError, ValueError):
            continue
    return prefixed


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _event_score(record: dict[str, Any] | None) -> float:
    if record is None:
        return 0.0
    return _coerce_float(record.get("event_score", 0.0), 0.0)


def _source_event_segments(source: str, record: dict[str, Any] | None) -> list[dict[str, Any]]:
    if record is None:
        return []
    segments = record.get("event_segments", [])
    normalized = []
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            item = dict(segment)
            item["source"] = source
            normalized.append(item)
    if normalized:
        return normalized
    return event_segments_from_frame_scores(record.get("frame_event_scores", []), source=source)


def _source_frame_event_scores(source: str, record: dict[str, Any] | None) -> list[dict[str, Any]]:
    if record is None:
        return []
    return normalize_frame_event_scores(record.get("frame_event_scores", []), source=source)


def fuse_score_records(
    individual_jsonl: str | Path,
    group_jsonl: str | Path,
    output_jsonl: str | Path,
    output_csv: str | Path,
    alpha: float = 0.65,
) -> dict[str, Any]:
    individual_jsonl = Path(individual_jsonl)
    group_jsonl = Path(group_jsonl)
    output_jsonl = Path(output_jsonl)
    output_csv = Path(output_csv)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    individual_records = _load_jsonl(individual_jsonl)
    group_records = _load_jsonl(group_jsonl)
    individual_by_id = {record["sample_id"]: record for record in individual_records}
    group_by_id = {record["sample_id"]: record for record in group_records}
    individual_norm = robust_normalize(
        {sample_id: float(record["score"]) for sample_id, record in individual_by_id.items()}
    )
    group_norm = robust_normalize(
        {sample_id: float(record["score"]) for sample_id, record in group_by_id.items()}
    )

    fused_records = []
    for sample_id in sorted(set(individual_by_id) | set(group_by_id)):
        individual = individual_by_id.get(sample_id)
        group = group_by_id.get(sample_id)
        base = individual or group
        assert base is not None
        used_sources = []
        if individual is not None and group is not None:
            score = alpha * individual_norm[sample_id] + (1.0 - alpha) * group_norm[sample_id]
            used_sources = ["individual", "group"]
        elif individual is not None:
            score = individual_norm[sample_id]
            used_sources = ["individual"]
        else:
            score = group_norm[sample_id]
            used_sources = ["group"]

        component_scores = {}
        if individual is not None:
            component_scores.update(_prefix_components("individual", individual.get("component_scores", {})))
        if group is not None:
            component_scores.update(_prefix_components("group", group.get("component_scores", {})))

        s_ind = individual_norm.get(sample_id, 0.0) if individual is not None else 0.0
        s_grp = group_norm.get(sample_id, 0.0) if group is not None else 0.0
        s_event = max(_event_score(individual), _event_score(group))
        component_scores.update(
            {
                "S_ind": float(s_ind),
                "S_grp": float(s_grp),
                "S_event": float(s_event),
                "S_fused": float(score),
            }
        )
        event_segments = (
            _source_event_segments("individual", individual)
            + _source_event_segments("group", group)
        )
        frame_event_scores = (
            _source_frame_event_scores("individual", individual)
            + _source_frame_event_scores("group", group)
        )
        frame_event_scores.sort(key=lambda item: (int(item["frame"]), item["source"]))

        fused_records.append(
            {
                "sample_id": sample_id,
                "sequence": base["sequence"],
                "track_id": base["track_id"],
                "category_id": base.get("category_id"),
                "category_name": base.get("category_name"),
                "source": "fusion",
                "score": float(score),
                "event_score": float(s_event),
                "event_segments": event_segments,
                "frame_event_scores": frame_event_scores,
                "component_scores": component_scores,
                "metadata": {
                    "alpha": alpha,
                    "used_sources": used_sources,
                    "individual_raw_score": None if individual is None else float(individual["score"]),
                    "group_raw_score": None if group is None else float(group["score"]),
                },
            }
        )

    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in fused_records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_id",
            "sequence",
            "track_id",
            "category_id",
            "category_name",
            "score",
            "used_sources",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in fused_records:
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "sequence": record["sequence"],
                    "track_id": record["track_id"],
                    "category_id": record["category_id"],
                    "category_name": record["category_name"],
                    "score": record["score"],
                    "used_sources": "|".join(record["metadata"]["used_sources"]),
                }
            )

    return {
        "individual_jsonl": str(individual_jsonl),
        "group_jsonl": str(group_jsonl),
        "output_jsonl": str(output_jsonl),
        "output_csv": str(output_csv),
        "num_fused_scores": len(fused_records),
    }
