"""Pure LoRA Plan data, per-group schedule and storage contract.

A LoRA Plan lists model-only LoRAs with a chunk selection each. It is compiled
to one ordered LoRA set per physical sampling group. Nothing in this module
loads a file, touches a MODEL, or imports ComfyUI.

Storage identity follows docs/LORA_PLAN_CONTRACT_V1.md: only groups that
actually receive a LoRA produce a record, so a plan that selects no chunk, and
every chunk a plan leaves untouched, keep the contract of a run without a plan.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from ..branch_provenance import BranchProvenanceError, physical_groups
from ..constants import MAX_CHUNKS
from .reference_routing import parse_chunk_selector


LORA_PLAN_TYPE = "H3_CONTINUUM_LORA_PLAN"
LORA_PLAN_CONTRACT_VERSION = 1
LORA_NONE = "None"
_HEX = frozenset("0123456789abcdef")


class LoraPlanError(ValueError):
    """A LoRA Plan, its schedule or a stored LoRA group contract is invalid."""


@dataclass(frozen=True, slots=True)
class LoraPlanRow:
    lora_name: str
    strength: float
    chunks: str


@dataclass(frozen=True, slots=True)
class LoraPlan:
    rows: tuple[LoraPlanRow, ...] = ()


@dataclass(frozen=True, slots=True)
class LoraUse:
    """One LoRA applied to one physical group, in plan order."""

    lora_name: str
    strength: float


@dataclass(frozen=True, slots=True)
class LoraGroupRoute:
    physical_group: int
    logical_chunks: tuple[int, ...]
    loras: tuple[LoraUse, ...]


@dataclass(frozen=True, slots=True)
class LoraSchedule:
    total_chunks: int
    terminal_merge_enabled: bool
    groups: tuple[LoraGroupRoute, ...]

    @property
    def active_groups(self) -> tuple[LoraGroupRoute, ...]:
        return tuple(group for group in self.groups if group.loras)


def _strength(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LoraPlanError("LoRA Plan strength must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise LoraPlanError("LoRA Plan strength must be finite")
    return result


def _selected_chunks(selector: Any, *, row_number: int) -> tuple[int, ...]:
    parsed = parse_chunk_selector(selector, total_chunks=MAX_CHUNKS)
    if not parsed.valid:
        details = "; ".join(issue.message for issue in parsed.issues)
        raise LoraPlanError(
            f"LoRA Plan row {row_number} chunk selection is invalid: {details}. "
            "Use all, off, or chunk numbers and ranges such as 1,3-5."
        )
    return parsed.logical_chunks


def append_lora_plan_row(
    previous: LoraPlan | None, *, lora_name: Any, strength: Any, chunks: Any,
) -> LoraPlan:
    """Return ``previous`` plus one validated row; never mutate ``previous``."""

    if previous is None:
        rows: tuple[LoraPlanRow, ...] = ()
    elif isinstance(previous, LoraPlan):
        rows = previous.rows
    else:
        raise LoraPlanError("LoRA Plan input must come from H3 Continuum LoRA Plan")
    if not isinstance(lora_name, str) or not lora_name.strip():
        raise LoraPlanError("LoRA Plan needs a LoRA name or None")
    if not isinstance(chunks, str):
        raise LoraPlanError("LoRA Plan chunk selection must be text")
    row = LoraPlanRow(lora_name, _strength(strength), chunks)
    _selected_chunks(row.chunks, row_number=len(rows) + 1)
    return LoraPlan((*rows, row))


def validate_lora_plan(plan: Any) -> LoraPlan:
    if not isinstance(plan, LoraPlan) or not isinstance(plan.rows, tuple):
        raise LoraPlanError("LoRA Plan input must come from H3 Continuum LoRA Plan")
    for number, row in enumerate(plan.rows, start=1):
        if not isinstance(row, LoraPlanRow) or not isinstance(row.lora_name, str):
            raise LoraPlanError(f"LoRA Plan row {number} is invalid")
        _strength(row.strength)
        _selected_chunks(row.chunks, row_number=number)
    return plan


def _row_is_active(row: LoraPlanRow) -> bool:
    # Core's LoRA loader returns the MODEL unchanged at strength 0.
    return row.lora_name != LORA_NONE and float(row.strength) != 0.0


def compile_lora_schedule(
    plan: LoraPlan, *, total_chunks: int, terminal_merge_enabled: bool,
) -> LoraSchedule:
    """Resolve the plan to one ordered LoRA set per physical group."""

    validate_lora_plan(plan)
    if isinstance(total_chunks, bool) or not isinstance(total_chunks, int) or total_chunks < 1:
        raise LoraPlanError("LoRA Plan needs a positive chunk count")
    try:
        groups = physical_groups(
            chunks=total_chunks, terminal_merge_enabled=bool(terminal_merge_enabled),
        )
    except BranchProvenanceError as exc:
        raise LoraPlanError(f"LoRA Plan physical groups are invalid: {exc}") from exc
    by_chunk: dict[int, list[LoraUse]] = {n: [] for n in range(1, total_chunks + 1)}
    for number, row in enumerate(plan.rows, start=1):
        if not _row_is_active(row):
            continue
        # Selections beyond the configured Chunks wait for a longer run.
        for chunk in _selected_chunks(row.chunks, row_number=number):
            if chunk <= total_chunks:
                by_chunk[chunk].append(LoraUse(row.lora_name, float(row.strength)))
    routes = []
    for group in groups:
        chunks = tuple(range(group.start, group.end + 1))
        member_sets = [tuple(by_chunk[chunk]) for chunk in chunks]
        if any(member != member_sets[0] for member in member_sets[1:]):
            raise LoraPlanError(
                f"LoRA Plan: chunks {chunks[0]} and {chunks[-1]} are sampled together "
                "as one group when First and Last Image are both connected. "
                "Give both chunks the same LoRAs."
            )
        routes.append(LoraGroupRoute(int(group.physical_group), chunks, member_sets[0]))
    return LoraSchedule(total_chunks, bool(terminal_merge_enabled), tuple(routes))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def lora_group_record(
    *, physical_group: int, logical_chunks: tuple[int, ...],
    loras: tuple[tuple[str, float], ...],
) -> dict[str, Any]:
    """Storage record for one group; ``loras`` is (file SHA-256, strength) in order."""

    if not loras:
        raise LoraPlanError("a LoRA group contract needs at least one LoRA")
    descriptor = {
        "version": LORA_PLAN_CONTRACT_VERSION,
        "physical_group": int(physical_group),
        "logical_chunks": [int(chunk) for chunk in logical_chunks],
        "loras": [
            {"sha256": str(sha256), "strength_model": _strength(strength)}
            for sha256, strength in loras
        ],
    }
    record = {"descriptor": descriptor, "sha256": canonical_sha256(descriptor)}
    return record


def validate_lora_group_contracts(
    records: Any, *, chunk_count: int, terminal_merge_enabled: bool,
) -> list[dict[str, Any]]:
    """Check stored or freshly built records against the run's group topology."""

    if not isinstance(records, list) or not records:
        raise LoraPlanError("LoRA group contracts must be a non-empty list")
    try:
        allowed = {
            int(group.physical_group): tuple(range(group.start, group.end + 1))
            for group in physical_groups(
                chunks=int(chunk_count), terminal_merge_enabled=bool(terminal_merge_enabled),
            )
        }
    except (BranchProvenanceError, TypeError, ValueError) as exc:
        raise LoraPlanError(f"LoRA group topology is invalid: {exc}") from exc
    previous_group = 0
    for record in records:
        if not isinstance(record, dict) or set(record) != {"descriptor", "sha256"}:
            raise LoraPlanError("LoRA group contract record is malformed")
        descriptor = record["descriptor"]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "version", "physical_group", "logical_chunks", "loras",
        }:
            raise LoraPlanError("LoRA group descriptor is malformed")
        if descriptor["version"] != LORA_PLAN_CONTRACT_VERSION:
            raise LoraPlanError("LoRA group contract version is incompatible")
        group = descriptor["physical_group"]
        if type(group) is not int or group <= previous_group:
            raise LoraPlanError("LoRA group contracts are not in ascending group order")
        previous_group = group
        if tuple(descriptor["logical_chunks"]) != allowed.get(group):
            raise LoraPlanError("LoRA group contract does not match the run's physical groups")
        loras = descriptor["loras"]
        if not isinstance(loras, list) or not loras:
            raise LoraPlanError("LoRA group contract lists no LoRA")
        for item in loras:
            if (not isinstance(item, dict) or set(item) != {"sha256", "strength_model"}
                    or not _valid_sha(item["sha256"])
                    or type(item["strength_model"]) is not float
                    or not math.isfinite(item["strength_model"])
                    or item["strength_model"] == 0.0):
                raise LoraPlanError("LoRA group contract entry is malformed")
        if record["sha256"] != canonical_sha256(descriptor):
            raise LoraPlanError("LoRA group contract SHA-256 is invalid")
    return records


def lora_group_sha_by_chunk(records: list[dict[str, Any]]) -> dict[int, str]:
    return {
        int(chunk): str(record["sha256"])
        for record in records
        for chunk in record["descriptor"]["logical_chunks"]
    }
