"""Per-chunk durations for the V3.9 Sampler (Experimental).

The Sampler's Chunk Seconds stays the length of every chunk. A Chunk Timings
bundle gives selected chunks a different length. Resolution returns ``None``
when no chunk differs from Chunk Seconds; every consumer then takes its
existing uniform path, so a run without an override generates and hashes
exactly as before. Storage identity follows docs/CHUNK_TIMINGS_CONTRACT_V1.md.

Nothing in this module imports torch or ComfyUI.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
import re
from typing import Any, Iterator

from ..constants import CHUNK_SECONDS_MAX, CHUNK_SECONDS_MIN, FPS, MAX_CHUNKS
from .reference_routing import parse_chunk_selector


CHUNK_TIMINGS_TYPE = "H3_CONTINUUM_CHUNK_TIMINGS"
CHUNK_TIMINGS_CONTRACT_VERSION = 1
# Same tolerance the runtime already uses to compare chunk_seconds values.
SECONDS_TOLERANCE = 1e-6

_SECONDS = re.compile(
    r"^(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:s|sec|secs|second|seconds)?$",
    re.IGNORECASE,
)
_KEY_SEPARATOR = re.compile(r"[:=]")

_ACTIVE: ContextVar["ChunkTimings | None"] = ContextVar(
    "h3_continuum_chunk_timings", default=None,
)


class ChunkTimingsError(ValueError):
    """Chunk Timings text, a resolved schedule or a stored record is invalid."""


@dataclass(frozen=True, slots=True)
class ChunkTimingRow:
    chunks: tuple[int, ...]
    seconds: float


@dataclass(frozen=True, slots=True)
class ChunkTimings:
    """Validated overrides in the order written; later rows win."""

    rows: tuple[ChunkTimingRow, ...] = ()


def _seconds(text: str, *, where: str) -> float:
    match = _SECONDS.match(text.strip())
    if match is None:
        raise ChunkTimingsError(f"Chunk Timings {where}: {text.strip()!r} is not a number of seconds")
    return _checked_seconds(float(match.group("value")), where=where)


def _checked_seconds(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChunkTimingsError(f"Chunk Timings {where}: seconds must be a number")
    result = float(value)
    if not math.isfinite(result) or not CHUNK_SECONDS_MIN <= result <= CHUNK_SECONDS_MAX:
        raise ChunkTimingsError(
            f"Chunk Timings {where}: {result:g}s is outside "
            f"{CHUNK_SECONDS_MIN:g}-{CHUNK_SECONDS_MAX:g}s"
        )
    return round(result, 6)


def _selected_chunks(selector: str, *, where: str) -> tuple[int, ...]:
    parsed = parse_chunk_selector(selector.strip(), total_chunks=MAX_CHUNKS)
    if not parsed.valid:
        details = "; ".join(issue.message for issue in parsed.issues)
        raise ChunkTimingsError(
            f"Chunk Timings {where}: chunk selection is invalid ({details}). "
            "Use all, or chunk numbers and ranges such as 2 or 1,3-5."
        )
    return tuple(parsed.logical_chunks)


def parse_chunk_timings(text: Any) -> ChunkTimings:
    """Parse a list of seconds (``5, 8, 6``) or ``chunks: seconds`` lines.

    A list gives chunk 1 its first value, chunk 2 its second, and so on.
    Lines such as ``2: 8`` or ``4-6: 10`` set only the chunks they name; a
    later line wins where two lines name the same chunk. Lines and list items
    may also be separated by ``;``. Empty text sets nothing.
    """

    if not isinstance(text, str):
        raise ChunkTimingsError("Chunk Timings must be text")
    entries = [
        (line_number, part.strip())
        for line_number, line in enumerate(text.splitlines(), start=1)
        for part in line.split(";")
        if part.strip()
    ]
    keyed = [entry for entry in entries if _KEY_SEPARATOR.search(entry[1])]
    if keyed and len(keyed) != len(entries):
        raise ChunkTimingsError(
            "Chunk Timings mixes a list of seconds with 'chunks: seconds' lines. "
            "Use one form, for example '5, 8, 6' or '2: 8'."
        )
    rows: list[ChunkTimingRow] = []
    if keyed:
        for line_number, entry in entries:
            selector, value = _KEY_SEPARATOR.split(entry, maxsplit=1)
            where = f"line {line_number}"
            rows.append(ChunkTimingRow(
                _selected_chunks(selector, where=where),
                _seconds(value, where=where),
            ))
        return ChunkTimings(tuple(rows))
    values = [
        token.strip()
        for _, entry in entries
        for token in entry.split(",")
    ]
    if any(not token for token in values):
        raise ChunkTimingsError("Chunk Timings list contains an empty item")
    if len(values) > MAX_CHUNKS:
        raise ChunkTimingsError(f"Chunk Timings lists {len(values)} chunks; the limit is {MAX_CHUNKS}")
    for number, token in enumerate(values, start=1):
        rows.append(ChunkTimingRow((number,), _seconds(token, where=f"chunk {number}")))
    return ChunkTimings(tuple(rows))


def validate_chunk_timings(timings: Any) -> ChunkTimings:
    if not isinstance(timings, ChunkTimings) or not isinstance(timings.rows, tuple):
        raise ChunkTimingsError("Chunk Timings input must come from H3 Continuum Chunk Timings")
    for number, row in enumerate(timings.rows, start=1):
        if not isinstance(row, ChunkTimingRow) or not isinstance(row.chunks, tuple):
            raise ChunkTimingsError(f"Chunk Timings row {number} is invalid")
        if any(isinstance(chunk, bool) or not isinstance(chunk, int)
               or not 1 <= chunk <= MAX_CHUNKS for chunk in row.chunks):
            raise ChunkTimingsError(f"Chunk Timings row {number} names an invalid chunk")
        _checked_seconds(row.seconds, where=f"row {number}")
    return timings


def resolve_chunk_durations(
    timings: ChunkTimings | None, *, chunks: int, chunk_seconds: float,
) -> tuple[float, ...] | None:
    """Return one duration per chunk, or ``None`` when none differs.

    Chunks beyond ``chunks`` are ignored until the run is that long. A value
    equal to Chunk Seconds is no override, so the result stays ``None``.
    """

    if timings is None:
        return None
    timings = validate_chunk_timings(timings)
    count = int(chunks)
    base = float(chunk_seconds)
    values = [base] * count
    for row in timings.rows:
        seconds = base if abs(row.seconds - base) <= SECONDS_TOLERANCE else row.seconds
        for chunk in row.chunks:
            if chunk <= count:
                values[chunk - 1] = seconds
    if all(value == base for value in values):
        return None
    return tuple(values)


def validate_chunk_durations(
    values: Any, *, chunks: int, chunk_seconds: float,
) -> tuple[float, ...]:
    """Validate a resolved schedule carried by a prompt plan."""

    count = int(chunks)
    base = float(chunk_seconds)
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ChunkTimingsError("chunk durations do not match the configured chunk count")
    result = tuple(
        _checked_seconds(value, where=f"chunk {number}")
        for number, value in enumerate(values, start=1)
    )
    if all(abs(value - base) <= SECONDS_TOLERANCE for value in result):
        raise ChunkTimingsError("chunk durations set no override; omit them instead")
    return tuple(base if abs(value - base) <= SECONDS_TOLERANCE else value for value in result)


def frames_through(
    count: int, *, chunk_seconds: float, durations: tuple[float, ...] | list[float] | None = None,
) -> int:
    """Visible frames the first ``count`` chunks should reach together.

    Without an override in that prefix this is the established
    ``count * chunk_seconds * FPS`` expression, evaluated in the same order.
    """

    count = int(count)
    base = float(chunk_seconds)
    prefix = None if durations is None else [float(value) for value in durations[:count]]
    if prefix is None or all(value == base for value in prefix):
        return int(round(count * base * FPS))
    return int(round(math.fsum(prefix) * FPS))


def chunk_seconds_at(
    index: int, *, chunk_seconds: float, durations: tuple[float, ...] | list[float] | None = None,
) -> float:
    """Duration of the zero-based chunk ``index``."""

    if durations is None:
        return float(chunk_seconds)
    return float(durations[int(index)])


def total_seconds(
    chunks: int, *, chunk_seconds: float, durations: tuple[float, ...] | list[float] | None = None,
) -> float:
    if durations is None:
        return int(chunks) * float(chunk_seconds)
    return math.fsum(float(value) for value in durations[: int(chunks)])


def duration_overrides(
    durations: tuple[float, ...] | list[float] | None, *, chunk_seconds: float,
) -> list[dict[str, Any]]:
    """Storage records: one per chunk whose length differs from Chunk Seconds."""

    if durations is None:
        return []
    base = float(chunk_seconds)
    return [
        {"chunk_number": number, "duration_seconds": float(value)}
        for number, value in enumerate(durations, start=1)
        if float(value) != base
    ]


def validate_duration_overrides(
    records: Any, *, chunk_count: int, chunk_seconds: float,
) -> list[dict[str, Any]]:
    """Check stored override records before they enter or leave a contract."""

    if not isinstance(records, list) or not records:
        raise ChunkTimingsError("chunk duration overrides must be a non-empty list")
    base = float(chunk_seconds)
    previous = 0
    result = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"chunk_number", "duration_seconds"}:
            raise ChunkTimingsError("chunk duration override record is malformed")
        number = record["chunk_number"]
        if isinstance(number, bool) or not isinstance(number, int):
            raise ChunkTimingsError("chunk duration override chunk number is invalid")
        if not previous < number <= int(chunk_count):
            raise ChunkTimingsError("chunk duration overrides are out of order or out of range")
        seconds = _checked_seconds(record["duration_seconds"], where=f"chunk {number}")
        if seconds != float(record["duration_seconds"]) or seconds == base:
            raise ChunkTimingsError(f"chunk {number} duration override is not canonical")
        previous = number
        result.append({"chunk_number": number, "duration_seconds": seconds})
    return result


def format_chunk_durations(durations: tuple[float, ...] | list[float], *, chunk_seconds: float) -> str:
    """One Status line, with runs of equal lengths collapsed."""

    groups: list[tuple[int, int, float]] = []
    for number, value in enumerate(durations, start=1):
        value = float(value)
        if groups and groups[-1][2] == value and groups[-1][1] == number - 1:
            groups[-1] = (groups[-1][0], number, value)
        else:
            groups.append((number, number, value))
    parts = [
        f"{start}: {value:g}s" if start == stop else f"{start}-{stop}: {value:g}s"
        for start, stop, value in groups
    ]
    total = math.fsum(float(value) for value in durations)
    return (
        "Chunk timings (Experimental): " + ", ".join(parts)
        + f"; total {total:g}s. Chunks without an override use Chunk Seconds "
        f"{float(chunk_seconds):g}s."
    )


@contextmanager
def chunk_timings_scope(timings: ChunkTimings | None) -> Iterator[ChunkTimings | None]:
    """Expose ``timings`` to the nested Sampler layers of one execution."""

    validated = None if timings is None else validate_chunk_timings(timings)
    token = _ACTIVE.set(validated)
    try:
        yield validated
    finally:
        _ACTIVE.reset(token)


def active_chunk_timings() -> ChunkTimings | None:
    return _ACTIVE.get()


def active_chunk_durations(*, chunks: int, chunk_seconds: float) -> tuple[float, ...] | None:
    return resolve_chunk_durations(_ACTIVE.get(), chunks=chunks, chunk_seconds=chunk_seconds)
