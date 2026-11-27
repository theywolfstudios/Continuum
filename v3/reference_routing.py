"""Pure V3.9 Reference selector parsing and group schedule planning.

V3.9 Custom consumes this plan through the Reference runtime and storage
contracts. Planning here remains free of Sampling and persistence side effects.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

from ..branch_provenance import BranchProvenanceError, physical_groups
from ..constants import MAX_REFERENCE_SLOTS, MAX_REFERENCES_PER_CHUNK


# R1-R9 are the original fixed slots. A Custom schedule must state each of
# them, and legacy "All" means exactly these nine.
REFERENCE_SLOT_IDS = tuple(f"R{index}" for index in range(1, 10))
# Every fixed identity. Extended slots (R10 and above) are optional in a
# schedule: an absent selector is "off", so a run that does not use them
# compiles exactly as it did when only nine slots existed.
ALL_REFERENCE_SLOT_IDS = tuple(f"R{index}" for index in range(1, MAX_REFERENCE_SLOTS + 1))
EXTENDED_REFERENCE_SLOT_IDS = ALL_REFERENCE_SLOT_IDS[len(REFERENCE_SLOT_IDS):]
REFERENCE_SLOT_ORDER = {slot_id: index for index, slot_id in enumerate(ALL_REFERENCE_SLOT_IDS)}
# Private selector value: "the chunks whose prompt names this slot's @R tag".
# It is resolved to an explicit chunk list before a schedule is compiled.
TAGGED_SELECTOR = "tagged"

REFERENCE_TAG = re.compile(r"(?<![A-Za-z0-9_])@R([1-9][0-9]?)(?![A-Za-z0-9_])")

_INTEGER_TOKEN = re.compile(r"^[0-9]+$")
_RANGE_TOKEN = re.compile(r"^([0-9]+)\s*-\s*([0-9]+)$")


@dataclass(frozen=True, slots=True)
class RoutingIssue:
    """A classified parser/planner issue; it does not itself stop execution."""

    code: str
    message: str
    slot_id: str | None = None
    selector: str | None = None


@dataclass(frozen=True, slots=True)
class ChunkSelectorResult:
    """Parsed selector with all/off kept distinct from explicit chunk lists."""

    kind: str
    logical_chunks: tuple[int, ...]
    issues: tuple[RoutingIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return self.kind != "invalid" and not self.issues


@dataclass(frozen=True, slots=True)
class LogicalReferenceRoute:
    """Reference identities selected for one project-wide logical chunk."""

    logical_chunk: int
    reference_slot_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PhysicalReferenceRoute:
    """Routing facts for one authoritative atomic physical sampling group."""

    physical_group: int
    logical_chunks: tuple[int, ...]
    terminal_atomic: bool
    logical_reference_slot_ids: tuple[tuple[str, ...], ...]
    status: str
    reference_slot_ids: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class ReferenceRoutingSchedule:
    """Full logical schedule plus its physical-group compatibility view."""

    mode: str
    legacy_fallback: bool
    total_chunks: int | None
    terminal_merge_enabled: bool | None
    logical_routes: tuple[LogicalReferenceRoute, ...] = ()
    physical_routes: tuple[PhysicalReferenceRoute, ...] = ()
    issues: tuple[RoutingIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.issues


def _invalid_selector(code: str, message: str, selector: str | None) -> ChunkSelectorResult:
    return ChunkSelectorResult(
        kind="invalid",
        logical_chunks=(),
        issues=(RoutingIssue(code=code, message=message, selector=selector),),
    )


def parse_chunk_selector(selector: Any, *, total_chunks: int) -> ChunkSelectorResult:
    """Parse all/off, a one-based chunk, inclusive ranges, and their combinations.

    Invalid values are returned as structured issues so callers can inspect
    them without introducing a new runtime execution stop.
    """

    if isinstance(total_chunks, bool) or not isinstance(total_chunks, int) or total_chunks < 1:
        return _invalid_selector(
            "invalid_chunk_count",
            "total_chunks must be a positive integer",
            selector if isinstance(selector, str) else None,
        )
    if not isinstance(selector, str):
        return _invalid_selector(
            "selector_not_string",
            "selector must be a string",
            None,
        )

    text = selector.strip()
    normalized = text.casefold()
    if not normalized:
        return _invalid_selector("selector_empty", "selector is empty", selector)
    if normalized == "all":
        return ChunkSelectorResult("all", tuple(range(1, total_chunks + 1)))
    if normalized == "off":
        return ChunkSelectorResult("off", ())

    selected: set[int] = set()
    issues: list[RoutingIssue] = []
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            issues.append(
                RoutingIssue(
                    "empty_selector_item",
                    "selector contains an empty comma-separated item",
                    selector=selector,
                )
            )
            continue

        if _INTEGER_TOKEN.fullmatch(token):
            start = end = int(token)
        else:
            match = _RANGE_TOKEN.fullmatch(token)
            if match is None:
                issues.append(
                    RoutingIssue(
                        "invalid_selector_item",
                        f"invalid selector item: {token!r}",
                        selector=selector,
                    )
                )
                continue
            start, end = int(match.group(1)), int(match.group(2))
            if start > end:
                issues.append(
                    RoutingIssue(
                        "reversed_range",
                        f"range start {start} exceeds end {end}",
                        selector=selector,
                    )
                )
                continue

        if start < 1 or end > total_chunks:
            issues.append(
                RoutingIssue(
                    "chunk_out_of_range",
                    f"selector item {token!r} is outside 1..{total_chunks}",
                    selector=selector,
                )
            )
            continue

        values = set(range(start, end + 1))
        if selected.intersection(values):
            issues.append(
                RoutingIssue(
                    "duplicate_chunk",
                    f"selector item {token!r} overlaps an earlier item",
                    selector=selector,
                )
            )
            continue
        selected.update(values)

    if issues:
        return ChunkSelectorResult("invalid", (), tuple(issues))
    return ChunkSelectorResult("chunks", tuple(sorted(selected)))


def tagged_reference_slot_ids(prompt: Any) -> tuple[str, ...]:
    """Fixed slots named by @R tags in one prompt, in fixed slot order."""

    found = {f"R{number}" for number in REFERENCE_TAG.findall(str(prompt))}
    return tuple(slot_id for slot_id in ALL_REFERENCE_SLOT_IDS if slot_id in found)


def resolve_tagged_selectors(
    selectors_by_slot: Any,
    *,
    prompts: Any,
    total_chunks: int,
    terminal_merge_enabled: bool,
) -> Any:
    """Replace each ``tagged`` selector with the chunks that name its @R tag.

    A terminal atomic physical group is sampled once from a prompt holding
    both members' text, so both members receive the union of their tags. The
    mapping is returned unchanged when no slot is tagged.
    """

    if not isinstance(selectors_by_slot, Mapping):
        return selectors_by_slot
    tagged = [
        slot_id for slot_id, selector in selectors_by_slot.items()
        if isinstance(selector, str) and selector.strip().casefold() == TAGGED_SELECTOR
    ]
    if not tagged:
        return selectors_by_slot
    if isinstance(total_chunks, bool) or not isinstance(total_chunks, int) or total_chunks < 1:
        raise ValueError("total_chunks must be a positive integer")
    prompt_list = list(prompts) if isinstance(prompts, (list, tuple)) else None
    if prompt_list is None or len(prompt_list) != total_chunks:
        raise ValueError("Tagged Reference routing needs one prompt per chunk")
    try:
        groups = physical_groups(
            chunks=total_chunks, terminal_merge_enabled=terminal_merge_enabled,
        )
    except BranchProvenanceError as exc:
        raise ValueError(str(exc)) from exc
    chunks_by_slot: dict[str, list[int]] = {slot_id: [] for slot_id in tagged}
    for group in groups:
        members = range(group.start, group.end + 1)
        named: set[str] = set()
        for chunk in members:
            named.update(tagged_reference_slot_ids(prompt_list[chunk - 1]))
        for slot_id in tagged:
            if slot_id in named:
                chunks_by_slot[slot_id].extend(members)
    resolved = dict(selectors_by_slot)
    for slot_id in tagged:
        resolved[slot_id] = ",".join(map(str, chunks_by_slot[slot_id])) or "off"
    return resolved


def reference_limit_issues(
    schedule: "ReferenceRoutingSchedule", connected_slot_ids: Any,
) -> tuple[RoutingIssue, ...]:
    """Physical groups whose connected selection exceeds the per-chunk limit."""

    connected = set(connected_slot_ids)
    issues: list[RoutingIssue] = []
    for route in schedule.physical_routes:
        active = tuple(
            slot_id for slot_id in (route.reference_slot_ids or ()) if slot_id in connected
        )
        if len(active) <= MAX_REFERENCES_PER_CHUNK:
            continue
        label = (
            f"Chunk {route.logical_chunks[0]}" if len(route.logical_chunks) == 1 else
            "Chunks " + "-".join(str(route.logical_chunks[index]) for index in (0, -1))
        )
        issues.append(RoutingIssue(
            "too_many_references",
            f"{label} would receive {len(active)} Reference Images ({', '.join(active)}); "
            f"one chunk accepts at most {MAX_REFERENCES_PER_CHUNK}",
        ))
    return tuple(issues)


def _invalid_schedule(
    *,
    mode: str,
    legacy_fallback: bool,
    total_chunks: int | None,
    terminal_merge_enabled: bool | None,
    issues: tuple[RoutingIssue, ...],
) -> ReferenceRoutingSchedule:
    return ReferenceRoutingSchedule(
        mode=mode,
        legacy_fallback=legacy_fallback,
        total_chunks=total_chunks,
        terminal_merge_enabled=terminal_merge_enabled,
        issues=issues,
    )


def compile_reference_routing_schedule(
    *,
    total_chunks: int,
    terminal_merge_enabled: bool,
    mode: Any = None,
    selectors_by_slot: Any = None,
) -> ReferenceRoutingSchedule:
    """Compile routing over full logical chunk numbers and physical groups.

    An absent mode preserves legacy All behavior. Custom mode requires an
    explicit selector for each fixed R1-R9 slot; explicit all-off is valid and
    remains distinct from an absent mode. Extended slots (R10 and above) are
    optional and default to off. Terminal atomic groups with
    different member routes are surfaced as conflicts without auto-correction.
    """

    legacy_fallback = mode is None
    normalized_mode = "all" if legacy_fallback else None
    issues: list[RoutingIssue] = []

    if isinstance(total_chunks, bool) or not isinstance(total_chunks, int) or total_chunks < 1:
        issues.append(RoutingIssue("invalid_chunk_count", "total_chunks must be a positive integer"))
    if type(terminal_merge_enabled) is not bool:
        issues.append(
            RoutingIssue("invalid_terminal_merge", "terminal_merge_enabled must be a boolean")
        )

    if not legacy_fallback:
        if not isinstance(mode, str):
            issues.append(RoutingIssue("invalid_mode", "mode must be All or Custom"))
        else:
            token = mode.strip().casefold()
            if token == "all":
                normalized_mode = "all"
            elif token == "custom":
                normalized_mode = "custom"
            else:
                issues.append(RoutingIssue("invalid_mode", "mode must be All or Custom"))

    if issues:
        return _invalid_schedule(
            mode=normalized_mode or "invalid",
            legacy_fallback=legacy_fallback,
            total_chunks=total_chunks if isinstance(total_chunks, int) and not isinstance(total_chunks, bool) else None,
            terminal_merge_enabled=terminal_merge_enabled if type(terminal_merge_enabled) is bool else None,
            issues=tuple(issues),
        )

    slot_selectors: dict[str, ChunkSelectorResult] = {}
    if normalized_mode == "custom":
        if not isinstance(selectors_by_slot, Mapping):
            issues.append(
                RoutingIssue(
                    "selectors_required",
                    "Custom mode requires an explicit selector map for R1-R9",
                )
            )
        else:
            for slot_id in selectors_by_slot:
                if slot_id not in REFERENCE_SLOT_ORDER:
                    issues.append(
                        RoutingIssue(
                            "unknown_reference_slot",
                            f"unknown Reference slot: {slot_id!r}",
                            slot_id=str(slot_id),
                        )
                    )
            for slot_id in ALL_REFERENCE_SLOT_IDS:
                if slot_id not in selectors_by_slot:
                    if slot_id in EXTENDED_REFERENCE_SLOT_IDS:
                        continue
                    issues.append(
                        RoutingIssue(
                            "missing_selector",
                            f"Custom mode has no selector for {slot_id}",
                            slot_id=slot_id,
                        )
                    )
                    continue
                parsed = parse_chunk_selector(
                    selectors_by_slot[slot_id], total_chunks=total_chunks
                )
                if not parsed.valid:
                    issues.extend(
                        RoutingIssue(
                            issue.code,
                            issue.message,
                            slot_id=slot_id,
                            selector=issue.selector,
                        )
                        for issue in parsed.issues
                    )
                else:
                    slot_selectors[slot_id] = parsed

    if issues:
        return _invalid_schedule(
            mode=normalized_mode or "invalid",
            legacy_fallback=legacy_fallback,
            total_chunks=total_chunks,
            terminal_merge_enabled=terminal_merge_enabled,
            issues=tuple(issues),
        )

    logical_routes: list[LogicalReferenceRoute] = []
    for logical_chunk in range(1, total_chunks + 1):
        if normalized_mode == "all":
            selected_slots = REFERENCE_SLOT_IDS
        else:
            selected_slots = tuple(
                slot_id
                for slot_id in ALL_REFERENCE_SLOT_IDS
                if slot_id in slot_selectors
                and logical_chunk in slot_selectors[slot_id].logical_chunks
            )
        logical_routes.append(
            LogicalReferenceRoute(logical_chunk, tuple(selected_slots))
        )

    try:
        groups = physical_groups(
            chunks=total_chunks,
            terminal_merge_enabled=terminal_merge_enabled,
        )
    except BranchProvenanceError as exc:
        return _invalid_schedule(
            mode=normalized_mode or "invalid",
            legacy_fallback=legacy_fallback,
            total_chunks=total_chunks,
            terminal_merge_enabled=terminal_merge_enabled,
            issues=(RoutingIssue("invalid_physical_groups", str(exc)),),
        )

    routes_by_chunk = {route.logical_chunk: route.reference_slot_ids for route in logical_routes}
    physical_routes: list[PhysicalReferenceRoute] = []
    for group in groups:
        chunks = tuple(range(group.start, group.end + 1))
        member_routes = tuple(routes_by_chunk[chunk] for chunk in chunks)
        first_route = member_routes[0]
        consistent = all(route == first_route for route in member_routes[1:])
        if not consistent:
            group_status = "conflict"
            group_slots = None
        elif not first_route:
            group_status = "empty"
            group_slots = ()
        else:
            group_status = "consistent"
            group_slots = first_route
        physical_routes.append(
            PhysicalReferenceRoute(
                physical_group=int(group.physical_group),
                logical_chunks=chunks,
                terminal_atomic=group.start != group.end,
                logical_reference_slot_ids=member_routes,
                status=group_status,
                reference_slot_ids=group_slots,
            )
        )

    return ReferenceRoutingSchedule(
        mode=normalized_mode or "invalid",
        legacy_fallback=legacy_fallback,
        total_chunks=total_chunks,
        terminal_merge_enabled=terminal_merge_enabled,
        logical_routes=tuple(logical_routes),
        physical_routes=tuple(physical_routes),
    )
