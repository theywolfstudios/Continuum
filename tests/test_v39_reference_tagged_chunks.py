"""Extended Reference slots, Tagged chunks routing and the per-chunk limit."""

from __future__ import annotations

from pathlib import Path
import re

import pytest
import torch

from ComfyUI_H3_Continuum_Join.constants import (
    MAX_REFERENCE_SLOTS,
    MAX_REFERENCES_PER_CHUNK,
    PROMPT_FORMAT_TIMELINE,
)
from ComfyUI_H3_Continuum_Join.reference import (
    REFERENCE_SIZE_MATCH_OUTPUT,
    ReferenceConditioningError,
    ReferenceImageBundle,
    prepare_reference_assets,
    resolve_reference_image_inputs,
)
from ComfyUI_H3_Continuum_Join.v2 import nodes as v2_nodes
from ComfyUI_H3_Continuum_Join.v2 import sequence
from ComfyUI_H3_Continuum_Join.v2.nodes import H3ContinuumSamplerV2
from ComfyUI_H3_Continuum_Join.v3 import driving_nodes
from ComfyUI_H3_Continuum_Join.v3.reference_images_v39 import (
    H3ContinuumReferenceImagesV39,
    REFERENCE_USE_OPTIONS,
    ReferenceImagesV39Bundle,
    effective_reference_inputs,
)
from ComfyUI_H3_Continuum_Join.v3.reference_plan_inspector import (
    format_reference_plan,
    project_reference_plan,
)
from ComfyUI_H3_Continuum_Join.v3.reference_routing import (
    ALL_REFERENCE_SLOT_IDS,
    EXTENDED_REFERENCE_SLOT_IDS,
    REFERENCE_SLOT_IDS,
    TAGGED_SELECTOR,
    compile_reference_routing_schedule,
    reference_limit_issues,
    resolve_tagged_selectors,
    tagged_reference_slot_ids,
)
from ComfyUI_H3_Continuum_Join.v3.reference_runtime import (
    ReferenceInputSet,
    ReferenceRoutingRequest,
    ReferenceRoutingRuntime,
    ReferenceRuntimeContractError,
)
from ComfyUI_H3_Continuum_Join.v3.reference_storage_contract import (
    build_reference_storage_plan,
    validate_reference_group_contracts,
)

from . import test_v39_rr_r4_sequence_integration as seq


ROOT = Path(__file__).resolve().parents[1]
TAGGED = "Tagged chunks"


def _image(value: float):
    return torch.full((1, 64, 64, 3), value, dtype=torch.float32)


def _value(slot: int) -> float:
    """A distinct, exactly representable level for each slot's test image."""
    return slot / 32.0


def _helper(reference_use, slots, **widgets):
    values = {f"reference_image_{slot}": _image(_value(slot)) for slot in slots}
    (bundle,) = H3ContinuumReferenceImagesV39().pack(
        reference_use=reference_use, **values, **widgets,
    )
    return bundle


def _selectors(**overrides):
    selectors = {slot_id: "off" for slot_id in REFERENCE_SLOT_IDS}
    selectors.update(overrides)
    return selectors


def _schedule(selectors, *, chunks, terminal=False):
    schedule = compile_reference_routing_schedule(
        total_chunks=chunks, terminal_merge_enabled=terminal,
        mode="Custom", selectors_by_slot=selectors,
    )
    assert schedule.valid, schedule.issues
    return schedule


def _routing_runtime(slots, selectors, *, chunks, terminal=False):
    images = [None] * MAX_REFERENCE_SLOTS
    for slot in slots:
        images[slot - 1] = _image(_value(slot))
    return ReferenceRoutingRuntime(
        schedule=_schedule(selectors, chunks=chunks, terminal=terminal),
        inputs=ReferenceInputSet(tuple(images), 64, 64, REFERENCE_SIZE_MATCH_OUTPUT),
    )


# --- limits and slot identities ------------------------------------------------


def test_slot_constants_keep_the_original_nine_first():
    assert MAX_REFERENCES_PER_CHUNK == 9 == len(REFERENCE_SLOT_IDS)
    assert MAX_REFERENCE_SLOTS > MAX_REFERENCES_PER_CHUNK
    assert ALL_REFERENCE_SLOT_IDS[:9] == REFERENCE_SLOT_IDS
    assert ALL_REFERENCE_SLOT_IDS == tuple(f"R{n}" for n in range(1, MAX_REFERENCE_SLOTS + 1))
    assert EXTENDED_REFERENCE_SLOT_IDS == ALL_REFERENCE_SLOT_IDS[9:]


def test_frontend_slot_limits_match_the_backend_constants():
    source = (ROOT / "web" / "reference_images_v39.js").read_text(encoding="utf-8")
    assert re.findall(r"^const SLOTS = (\d+);$", source, re.M) == [str(MAX_REFERENCE_SLOTS)]
    assert re.findall(r"^const PER_CHUNK_LIMIT = (\d+);$", source, re.M) == [
        str(MAX_REFERENCES_PER_CHUNK)
    ]
    assert f'const TAGGED = "{TAGGED}";' in source
    assert REFERENCE_USE_OPTIONS == ("All chunks", "Per chunk", TAGGED)


# --- unused extended slots change nothing ---------------------------------------


def test_nine_slot_request_is_unchanged_when_no_extended_slot_is_connected():
    for mode in ("All chunks", "Per chunk"):
        bundle = _helper(mode, (1, 4, 9), reference_r1_chunks="1,2",
                         reference_r4_chunks="2", reference_r9_chunks="all")
        direct, extra, selectors = effective_reference_inputs(bundle, chunks=3)
        assert tuple(selectors) == REFERENCE_SLOT_IDS
        assert len(direct) == 3 and len(extra.images) == 6
        # The legacy bundle shape still resolves on the unrouted path.
        assert len(resolve_reference_image_inputs(*direct, None, None, extra)) == 9


def test_explicit_off_for_extended_slots_compiles_to_the_same_schedule():
    base = _selectors(R1="1", R9="2-3")
    explicit = {**base, **{slot_id: "off" for slot_id in EXTENDED_REFERENCE_SLOT_IDS}}
    assert _schedule(base, chunks=3) == _schedule(explicit, chunks=3)


def test_nine_slot_storage_descriptors_do_not_depend_on_the_slot_count():
    prompts = ["@R1 walks", "@R4 waits", "nobody"]
    selectors = _selectors(R1="1", R4="2")
    short = ReferenceRoutingRuntime(
        schedule=_schedule(selectors, chunks=3),
        inputs=ReferenceInputSet(
            (_image(_value(1)), None, None, _image(_value(4)), *(None,) * 5),
            64, 64, REFERENCE_SIZE_MATCH_OUTPUT,
        ),
    )
    padded = _routing_runtime((1, 4), selectors, chunks=3)
    plans = [
        build_reference_storage_plan(
            runtime=runtime, prompts=prompts, first_frame_hash="none", last_frame_hash="none",
        )
        for runtime in (short, padded)
    ]
    assert plans[0].descriptor_jsons == plans[1].descriptor_jsons
    assert plans[0].routing_config_sha256 == plans[1].routing_config_sha256


def test_legacy_unrouted_path_still_rejects_an_extended_bundle():
    extended = ReferenceImageBundle((None,) * (MAX_REFERENCE_SLOTS - 3))
    with pytest.raises(ReferenceConditioningError, match="slots 4–9"):
        resolve_reference_image_inputs(None, None, None, None, None, extended)
    with pytest.raises(ReferenceConditioningError, match="slots 4–9"):
        prepare_reference_assets(
            reference_image_1=_image(0.5), reference_image_2=None,
            output_width=64, output_height=64, size_mode=REFERENCE_SIZE_MATCH_OUTPUT,
            image_references=extended,
        )
    oversized = ReferenceImageBundle((None,) * (MAX_REFERENCE_SLOTS - 2))
    with pytest.raises(ReferenceConditioningError):
        resolve_reference_image_inputs(None, None, None, None, None, oversized,
                                       allow_extended=True)


# --- extended slots ---------------------------------------------------------------


def test_extended_slot_keeps_its_fixed_number_through_the_private_request():
    bundle = _helper("Per chunk", (2, 12), reference_r2_chunks="1",
                     reference_r12_chunks="2-3")
    direct, extra, selectors = effective_reference_inputs(bundle, chunks=3)
    assert tuple(selectors) == (*REFERENCE_SLOT_IDS, "R12")
    assert selectors["R2"] == "1" and selectors["R12"] == "2,3"
    images = resolve_reference_image_inputs(*direct, None, None, extra, allow_extended=True)
    assert len(images) == MAX_REFERENCE_SLOTS
    assert [index for index, image in enumerate(images, start=1) if image is not None] == [2, 12]
    inputs = ReferenceInputSet.from_inputs(
        reference_image_1=direct[0], reference_image_2=direct[1], reference_image_3=direct[2],
        image_references=extra, output_width=64, output_height=64,
        size_mode=REFERENCE_SIZE_MATCH_OUTPUT,
    )
    assert inputs.connected_source_slot_ids == ("R2", "R12")
    assets = inputs.prepare_selected(("R12",))
    assert assets.source_slot_ids == ("R12",)
    assert float(assets.images[0].mean()) == pytest.approx(_value(12))


def test_blank_saved_value_in_an_extended_slot_means_off():
    # A workflow saved with nine slots can restore "" into the next widget.
    bundle = _helper("Per chunk", (10,), reference_r10_chunks="")
    assert bundle.selectors[9] == "off"
    assert effective_reference_inputs(bundle, chunks=2)[2]["R10"] == "off"
    with pytest.raises(ValueError, match="invalid V3.9 Reference chunk selection"):
        _helper("Per chunk", (1,), reference_r1_chunks="")


def test_schedule_and_storage_accept_extended_slots_in_fixed_order():
    runtime = _routing_runtime(
        (3, 12, 24), _selectors(R3="1-2", R12="2", R24="3"), chunks=3,
    )
    assert [route.reference_slot_ids for route in runtime.schedule.physical_routes] == [
        ("R3",), ("R3", "R12"), ("R24",),
    ]
    plan = build_reference_storage_plan(
        runtime=runtime, prompts=["@R3", "@R3 with @R12", "@R24"],
        first_frame_hash="none", last_frame_hash="none",
    )
    records = validate_reference_group_contracts(
        plan.group_contracts, chunk_count=3, terminal_merge_enabled=False,
    )
    assert [
        [(ref["source_slot_id"], ref["picture_number"]) for ref in record["descriptor"]["references"]]
        for record in records
    ] == [[("R3", 1)], [("R3", 1), ("R12", 2)], [("R24", 1)]]
    assert not compile_reference_routing_schedule(
        total_chunks=3, terminal_merge_enabled=False, mode="Custom",
        selectors_by_slot=_selectors(**{f"R{MAX_REFERENCE_SLOTS + 1}": "1"}),
    ).valid


# --- tagged chunks ----------------------------------------------------------------


def test_tag_scan_uses_whole_slot_numbers_only():
    prompt = "@R1 and @R12, not X@R3, @R4suffix, @R05, @R99 or @R123; @R12 again"
    assert tagged_reference_slot_ids(prompt) == ("R1", "R12")
    assert tagged_reference_slot_ids("no tags") == ()


def test_tagged_selectors_follow_each_chunk_prompt():
    prompts = ["@R1 enters", "@R12 answers @R1", "an empty street", "@R12 leaves"]
    selectors = _selectors(R1=TAGGED_SELECTOR, R2=TAGGED_SELECTOR, R5="3", R12=TAGGED_SELECTOR)
    resolved = resolve_tagged_selectors(
        selectors, prompts=prompts, total_chunks=4, terminal_merge_enabled=False,
    )
    assert resolved == {**selectors, "R1": "1,2", "R2": "off", "R12": "2,4"}
    assert [route.reference_slot_ids for route in _schedule(resolved, chunks=4).logical_routes] == [
        ("R1",), ("R1", "R12"), ("R5",), ("R12",),
    ]


def test_selectors_without_a_tagged_slot_pass_through_untouched():
    selectors = _selectors(R1="1", R9="all")
    assert resolve_tagged_selectors(
        selectors, prompts=None, total_chunks=2, terminal_merge_enabled=False,
    ) is selectors
    with pytest.raises(ValueError, match="one prompt per chunk"):
        resolve_tagged_selectors(
            _selectors(R1=TAGGED_SELECTOR), prompts=["only one"], total_chunks=2,
            terminal_merge_enabled=False,
        )


def test_tagged_terminal_pair_shares_the_union_instead_of_conflicting():
    prompts = ["@R1 starts", "@R2 turns", "@R3 ends"]
    resolved = resolve_tagged_selectors(
        _selectors(R1=TAGGED_SELECTOR, R2=TAGGED_SELECTOR, R3=TAGGED_SELECTOR), prompts=prompts,
        total_chunks=3, terminal_merge_enabled=True,
    )
    assert (resolved["R1"], resolved["R2"], resolved["R3"]) == ("1", "2,3", "2,3")
    schedule = _schedule(resolved, chunks=3, terminal=True)
    assert [route.status for route in schedule.physical_routes] == ["consistent", "consistent"]
    assert schedule.physical_routes[1].reference_slot_ids == ("R2", "R3")


def test_helper_emits_the_tagged_selector_for_connected_images_only():
    bundle = _helper(TAGGED, (1, 12), reference_r1_chunks="3", reference_r12_chunks="not parsed")
    _, extra, selectors = effective_reference_inputs(bundle, chunks=3)
    assert selectors == {**_selectors(R1=TAGGED_SELECTOR), "R12": TAGGED_SELECTOR}
    assert len(extra.images) == MAX_REFERENCE_SLOTS - 3
    # Saved Per chunk ticks survive a visit to Tagged chunks.
    assert bundle.selectors[0] == "3" and bundle.selectors[11] == "not parsed"
    with pytest.raises(ValueError, match="Reference Use must be"):
        H3ContinuumReferenceImagesV39().pack(reference_use="Tagged")


def test_each_tagged_chunk_encodes_and_presents_only_its_own_images():
    slots = tuple(range(1, 13))
    prompts = [
        " ".join(f"@R{slot}" for slot in range(1, 7)),
        " ".join(f"@R{slot}" for slot in range(7, 13)),
        "a quiet transition",
    ]
    resolved = resolve_tagged_selectors(
        {**_selectors(), **{f"R{slot}": TAGGED_SELECTOR for slot in slots}},
        prompts=prompts, total_chunks=3, terminal_merge_enabled=False,
    )
    runtime = _routing_runtime(slots, resolved, chunks=3)
    runtime.check_physical_contract(chunks=3, terminal_merge_enabled=False)
    vae, clip = seq._VAE(), seq._Clip()
    groups = [
        runtime.prepare_group(
            physical_group=group, prompt=prompts[group - 1], clip=clip, video_vae=vae,
            first_image=None, last_image=None,
        )
        for group in (1, 2, 3)
    ]
    assert groups[0].selected_assets.source_slot_ids == tuple(f"R{n}" for n in range(1, 7))
    assert groups[1].selected_assets.source_slot_ids == tuple(f"R{n}" for n in range(7, 13))
    assert groups[2].selected_assets is None
    # Twelve images are connected; no chunk was given more than six.
    assert groups[0].effective_prompt == " ".join(f"<Picture {n}>" for n in range(1, 7))
    assert groups[1].effective_prompt == " ".join(f"<Picture {n}>" for n in range(1, 7))
    assert [len(group.conditioning[0][1].get("minimax_refs") or []) for group in groups] == [6, 6, 0]
    assert vae.values == pytest.approx([_value(slot) for slot in slots])
    assert all(not group.warnings for group in groups)


# --- per-chunk limit --------------------------------------------------------------


def test_more_than_nine_images_in_one_chunk_stops_before_encoding():
    slots = tuple(range(1, 12))
    runtime = _routing_runtime(
        slots, {**_selectors(), **{f"R{slot}": "all" for slot in slots}}, chunks=2,
    )
    issues = reference_limit_issues(runtime.schedule, runtime.inputs.connected_source_slot_ids)
    assert [issue.code for issue in issues] == ["too_many_references"] * 2
    with pytest.raises(ReferenceRuntimeContractError) as raised:
        runtime.check_physical_contract(chunks=2, terminal_merge_enabled=False)
    message = str(raised.value)
    assert "Chunk 1 would receive 11 Reference Images" in message
    assert "at most 9" in message and "Tagged chunks or Per chunk" in message


def test_limit_counts_connected_images_per_physical_group():
    # Nine in one chunk is the existing envelope and stays allowed.
    nine = _routing_runtime(
        range(1, 10), _selectors(**{f"R{slot}": "all" for slot in range(1, 10)}), chunks=1,
    )
    nine.check_physical_contract(chunks=1, terminal_merge_enabled=False)
    # Eighteen connected images, nine per chunk.
    selectors = {**_selectors(**{f"R{slot}": "1" for slot in range(1, 10)}),
                 **{f"R{slot}": "2" for slot in range(10, 19)}}
    spread = _routing_runtime(range(1, 19), selectors, chunks=2)
    spread.check_physical_contract(chunks=2, terminal_merge_enabled=False)
    # A routed but unconnected slot is not an image and is not counted.
    routed = {**_selectors(**{f"R{slot}": "1" for slot in range(1, 10)}), "R10": "1"}
    unconnected = _routing_runtime(range(1, 10), routed, chunks=1)
    unconnected.check_physical_contract(chunks=1, terminal_merge_enabled=False)
    merged = _routing_runtime(
        range(1, 11), {**_selectors(), **{f"R{slot}": "2-3" for slot in range(1, 11)}},
        chunks=3, terminal=True,
    )
    with pytest.raises(ReferenceRuntimeContractError, match="Chunks 2-3 would receive 10"):
        merged.check_physical_contract(chunks=3, terminal_merge_enabled=True)


# --- Sampler wiring ---------------------------------------------------------------


def test_v39_sampler_forwards_tagged_selectors_and_extended_images(monkeypatch):
    calls = []

    def fake_v38_run(_self, **kwargs):
        calls.append(kwargs)
        return ("unchanged",)

    monkeypatch.setattr(driving_nodes.H3ContinuumSamplerV38, "run", fake_v38_run)
    bundle = _helper(TAGGED, (1, 12))
    result = driving_nodes.H3ContinuumSamplerV39().run(
        reference_images=bundle, chunks=2, reference_r12_chunks="stale API key",
    )
    assert result == ("unchanged",)
    (kwargs,) = calls
    assert "reference_r12_chunks" not in kwargs
    assert kwargs["_reference_routing_settings"] == {**_selectors(R1=TAGGED_SELECTOR), "R12": TAGGED_SELECTOR}
    assert len(kwargs["image_references"].images) == MAX_REFERENCE_SLOTS - 3


def _run_v2_node(monkeypatch, *, bundle, script, chunks, first=None, last=None, runtime=None):
    """Drive the V2 node that owns the prompt plan, down to fake sampling."""
    if runtime is None:
        runtime = seq._runtime(monkeypatch, first=first, last=last)
    direct, extra, selectors = effective_reference_inputs(bundle, chunks=chunks)
    request = ReferenceRoutingRequest(
        selectors_by_slot=selectors,
        inputs=ReferenceInputSet.from_inputs(
            reference_image_1=direct[0], reference_image_2=direct[1],
            reference_image_3=direct[2], image_references=extra,
            output_width=64, output_height=64, size_mode=REFERENCE_SIZE_MATCH_OUTPUT,
        ),
    )
    outputs = H3ContinuumSamplerV2().run(
        model=runtime.model, clip=runtime.clip, video_vae=runtime.vae, audio_vae=None,
        sampler=object(), sigmas=torch.tensor([1.0, 0.0]),
        prompt_mode=PROMPT_FORMAT_TIMELINE, prompt_script=script,
        chunks=chunks, chunk_seconds=5.0, width=64, height=64,
        continuity=seq.V2_CONTINUITY_OPTIONS[1], base_seed=42,
        audio_continuity=True, exact_total_duration=False,
        diagnostics=seq.DIAGNOSTICS_BASIC, reroll_from_chunk=0, reroll_nonce=0,
        strict_compatibility=False, debug=False,
        first_frame=first, last_frame=last, latent_only=True,
        reference_routing_request=request,
    )
    return runtime, request, outputs


def test_tagged_chunks_reach_sampling_with_only_the_tagged_images(monkeypatch):
    script = (
        "[Chunk 1]\n@R1 greets @R12\n\n"
        "[Chunk 2]\nthe street is empty\n\n"
        "[Chunk 3]\n@R12 walks away"
    )
    bundle = _helper(TAGGED, (1, 2, 12))
    runtime, request, _outputs = _run_v2_node(
        monkeypatch, bundle=bundle, script=script, chunks=3,
    )
    assert [route.reference_slot_ids for route in request.runtime.schedule.physical_routes] == [
        ("R1", "R12"), (), ("R12",),
    ]
    # R2 is connected but never tagged: it is never encoded or presented.
    assert [sample["refs"] for sample in runtime.samples] == [
        pytest.approx([_value(1), _value(12)]), [], pytest.approx([_value(12)]),
    ]
    assert sorted(set(runtime.vae.values)) == pytest.approx([_value(1), _value(12)])
    assert "<Picture 1> greets <Picture 2>" in runtime.samples[0]["prompt"]
    assert "<Picture 1> walks away" in runtime.samples[2]["prompt"]
    report = format_reference_plan(project_reference_plan(request.runtime))
    assert "Connected: R1, R2, R12" in report
    assert "Not used by any chunk: R2" in report


def test_all_chunks_with_too_many_images_fails_before_any_vae_or_sampling(monkeypatch):
    bundle = _helper("All chunks", range(1, 11))
    runtime = seq._runtime(monkeypatch)
    with pytest.raises(Exception, match="would receive 10 Reference Images") as raised:
        _run_v2_node(monkeypatch, bundle=bundle, runtime=runtime, chunks=2,
                     script="[Chunk 1]\none\n\n[Chunk 2]\ntwo")
    assert "Tagged chunks or Per chunk" in str(raised.value)
    assert runtime.vae.values == [] and runtime.clip.calls == [] and runtime.samples == []
