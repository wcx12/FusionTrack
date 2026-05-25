from __future__ import annotations

from typing import Any, Iterable

from protocol.schemas import build_sample_id


SEQUENCE_FIELDS = ("sequence", "sequence_name", "seq", "video", "video_id")
TRACK_FIELDS = ("track_id", "object_id", "target_id", "id")
SAMPLE_FIELDS = ("sample_id",)
WINDOW_FIELDS = ("window_id", "window", "segment_id", "event_id")
LABEL_FIELDS = ("label", "is_anomaly", "anomaly", "is_abnormal", "target")
TYPE_FIELDS = ("anomaly_type", "type", "label_type", "event_type")
FRAME_START_FIELDS = ("frame_start", "start_frame", "start", "begin_frame")
FRAME_END_FIELDS = ("frame_end", "end_frame", "end", "stop_frame")
TRUE_VALUES = {"1", "true", "yes", "y", "anomaly", "abnormal", "positive", "pos"}
FALSE_VALUES = {"0", "false", "no", "n", "normal", "negative", "neg"}


def normalize_real_label_rows(rows: Iterable[dict[str, Any]], level: str) -> list[dict[str, Any]]:
    if level not in {"individual", "group"}:
        raise ValueError("level must be one of: individual, group")
    normalized = [_normalize_row(row, level=level, row_index=index) for index, row in enumerate(rows, start=1)]
    return normalized


def _normalize_row(row: dict[str, Any], level: str, row_index: int) -> dict[str, Any]:
    sequence = _optional_text(_first_present(row, SEQUENCE_FIELDS))
    track_id = _optional_text(_first_present(row, TRACK_FIELDS))
    sample_id = _optional_text(_first_present(row, SAMPLE_FIELDS))
    if not sample_id:
        if not sequence or not track_id:
            raise ValueError(f"real label row {row_index} requires sample_id or sequence + track_id")
        sample_id = build_sample_id(sequence, track_id)

    label_value = _first_present(row, LABEL_FIELDS)
    label = _binary_label(label_value, row_index=row_index)
    anomaly_type = _optional_text(_first_present(row, TYPE_FIELDS)) or ("real_anomaly" if label else "normal")
    output: dict[str, Any] = {
        "sample_id": sample_id,
        "sequence": sequence or _sequence_from_sample_id(sample_id),
        "track_id": track_id or _track_from_sample_id(sample_id),
        "label": label,
        "anomaly_type": anomaly_type,
        "metadata": {"source": "real_label"},
    }

    frame_start = _optional_int(_first_present(row, FRAME_START_FIELDS), field="frame_start", row_index=row_index)
    frame_end = _optional_int(_first_present(row, FRAME_END_FIELDS), field="frame_end", row_index=row_index)
    if frame_start is not None:
        output["frame_start"] = frame_start
    if frame_end is not None:
        output["frame_end"] = frame_end

    if level == "group":
        window_id = _optional_text(_first_present(row, WINDOW_FIELDS))
        if not window_id:
            raise ValueError(f"real group label row {row_index} requires window_id")
        output["window_id"] = window_id
        output["metadata"]["window_id"] = window_id
    return output


def _first_present(row: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return None


def _binary_label(value: Any, row_index: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if value in (None, ""):
        raise ValueError(f"real label row {row_index} requires a label field")
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return 1
    if text in FALSE_VALUES:
        return 0
    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(f"real label row {row_index} has unsupported label value: {value!r}") from exc
    if numeric in (0.0, 1.0):
        return int(numeric)
    raise ValueError(f"real label row {row_index} has unsupported label value: {value!r}")


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value: Any, field: str, row_index: int) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"real label row {row_index} field {field!r} must be an integer") from exc


def _sequence_from_sample_id(sample_id: str) -> str:
    return sample_id.split(":", 1)[0] if ":" in sample_id else ""


def _track_from_sample_id(sample_id: str) -> str:
    return sample_id.split(":", 1)[1] if ":" in sample_id else sample_id
