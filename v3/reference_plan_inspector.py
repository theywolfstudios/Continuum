"""Read-only V3.9 projection of the Reference contract used by this Queue.

This module never selects images or changes a route. Pending groups are shown
as settings only; accepted groups use the frozen/observed RR-R5 descriptor.
"""

from __future__ import annotations

from typing import Any

from .reference_routing import EXTENDED_REFERENCE_SLOT_IDS
from .reference_runtime import ReferenceRoutingRuntime


def project_reference_plan(runtime: ReferenceRoutingRuntime) -> dict[str, Any]:
    if not isinstance(runtime, ReferenceRoutingRuntime):
        raise TypeError("Reference Plan Inspector requires the active routing runtime")
    accepted = int(runtime.accepted_chunks)
    generated = set(runtime.generated_groups)
    groups: list[dict[str, Any]] = []
    for route in runtime.schedule.physical_routes:
        group = int(route.physical_group)
        complete = route.logical_chunks[-1] <= accepted
        item: dict[str, Any] = {
            "physical_group": group,
            "logical_chunks": list(route.logical_chunks),
            "requested_slots": list(route.reference_slot_ids or ()),
            "status": (
                "pending" if not complete else
                "generated" if group in generated else "reused"
            ),
        }
        if complete:
            contract = runtime.group_contract_for(group)
            item["contract_sha256"] = contract["sha256"]
            item["descriptor"] = contract["descriptor"]
            item["warnings"] = list(runtime.group_warnings.get(group, ()))
        groups.append(item)
    plan = {
        "version": 1,
        "mode": runtime.schedule.mode,
        "accepted_chunks": accepted,
        "connected_slots": list(runtime.inputs.connected_source_slot_ids),
        "groups": groups,
    }
    # Extended slots make an unassigned image easy to miss, most of all with
    # Tagged chunks. Runs that connect only R1-R9 keep their previous report.
    routed = {slot for route in runtime.schedule.physical_routes
              for slot in (route.reference_slot_ids or ())}
    extended = set(EXTENDED_REFERENCE_SLOT_IDS)
    if any(slot in extended for slot in plan["connected_slots"]):
        plan["unused_slots"] = [slot for slot in plan["connected_slots"] if slot not in routed]
    return plan


def format_reference_plan(plan: dict[str, Any]) -> str:
    lines = [
        "Reference Plan Inspector (last Queue, runtime verified)",
        f"Mode: {plan['mode']}; accepted chunks: {plan['accepted_chunks']}",
        "Connected: " + (", ".join(plan["connected_slots"]) or "none"),
    ]
    if plan.get("unused_slots"):
        lines.append("Not used by any chunk: " + ", ".join(plan["unused_slots"]))
    for group in plan["groups"]:
        label = ",".join(str(chunk) for chunk in group["logical_chunks"])
        head = f"Group {group['physical_group']} [C{label}] {group['status']}"
        descriptor = group.get("descriptor")
        if descriptor is None:
            lines.append(head + " — configured " + (
                ", ".join(group["requested_slots"]) or "no References"
            ))
            continue
        mapping = ", ".join(
            f"{ref['source_slot_id']}=<Picture {ref['picture_number']}>"
            for ref in descriptor["references"]
        ) or "no References"
        lines.append(head + " — " + mapping + f"; contract={group['contract_sha256'][:12]}")
        for warning in group["warnings"]:
            lines.append("  Warning: " + warning)
    return "\n".join(lines)
