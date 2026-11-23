"""The public Chunks limit is one shared constant; legacy limits stay at 16."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import pytest

from ComfyUI_H3_Continuum_Join.branch_provenance import physical_groups
from ComfyUI_H3_Continuum_Join.constants import MAX_CHUNKS, PROMPT_FORMAT_TIMELINE
from ComfyUI_H3_Continuum_Join.run_storage import RunStorageController
from ComfyUI_H3_Continuum_Join.v2.nodes import H3ContinuumSamplerV2
from ComfyUI_H3_Continuum_Join.v2.prompts import (
    PromptPlanError,
    make_prompt_plan,
    validate_prompt_plan,
)
from ComfyUI_H3_Continuum_Join.v2.seeds import derive_chunk_seed
from ComfyUI_H3_Continuum_Join.v3.driving_nodes import (
    H3ContinuumSamplerV38,
    H3ContinuumSamplerV39,
)
from ComfyUI_H3_Continuum_Join.v3.easy_nodes import (
    EASY_MAX_CHUNKS,
    resolve_easy_duration,
)
from ComfyUI_H3_Continuum_Join.v3.node_facade import (
    execute_v38_runtime_request,
    make_runtime_request,
)
from ComfyUI_H3_Continuum_Join.v3.nodes import (
    REGENERATE_OPTIONS,
    _regenerate_from_value,
)
from ComfyUI_H3_Continuum_Join.v3.reference_images_v39 import (
    H3ContinuumReferenceImagesV39,
    effective_reference_inputs,
)
from ComfyUI_H3_Continuum_Join.v3.reference_routing import (
    compile_reference_routing_schedule,
)
from ComfyUI_H3_Continuum_Join.v3.reliability_v38 import long_run_decode_advisory
from ComfyUI_H3_Continuum_Join.v3.resolution import (
    H3_SIZE_SOURCE_LEGACY,
    H3_SIZE_SOURCE_MANUAL,
)
from ComfyUI_H3_Continuum_Join.v3.review_control import GENERATION_MODE_FULL_RUN

from . import test_v38_review_run_storage as review


ROOT = Path(__file__).resolve().parents[1]


def test_limit_value():
    assert MAX_CHUNKS == 64


@pytest.mark.parametrize("node_class", [H3ContinuumSamplerV38, H3ContinuumSamplerV39])
def test_public_sampler_schema_uses_the_shared_limit(node_class):
    required = node_class.INPUT_TYPES()["required"]
    assert required["chunks"][1]["min"] == 1
    assert required["chunks"][1]["max"] == MAX_CHUNKS
    assert required["chunks"][1]["default"] == 3
    assert required["take_group"][1]["max"] == MAX_CHUNKS
    assert required["reroll_from_chunk"][0] == REGENERATE_OPTIONS


def test_regenerate_options_cover_every_selectable_chunk():
    assert REGENERATE_OPTIONS[0] == "Auto"
    assert REGENERATE_OPTIONS[1:] == tuple(
        f"Chunk {index}" for index in range(1, MAX_CHUNKS + 1)
    )
    assert _regenerate_from_value(f"Chunk {MAX_CHUNKS}", chunks=MAX_CHUNKS) == MAX_CHUNKS
    with pytest.raises(ValueError):
        _regenerate_from_value("Chunk 41", chunks=40)


@pytest.mark.parametrize("node_class", [H3ContinuumSamplerV38, H3ContinuumSamplerV39])
@pytest.mark.parametrize("chunk", [1, 16, 17, 40, MAX_CHUNKS])
def test_regenerate_from_any_selectable_chunk_is_never_served_from_cache(node_class, chunk):
    value = node_class.IS_CHANGED(
        generation_mode=GENERATION_MODE_FULL_RUN,
        run_storage="Save + Auto Resume",
        reroll_from_chunk=f"Chunk {chunk}",
        reroll_nonce=0,
        chunks=MAX_CHUNKS,
    )
    assert math.isnan(value)


def test_frontend_limits_match_the_backend_constant():
    project = (ROOT / "web" / "project_id.js").read_text(encoding="utf-8")
    reference = (ROOT / "web" / "reference_images_v39.js").read_text(encoding="utf-8")
    assert re.findall(r"^const MAX_CHUNKS = (\d+);$", project, re.M) == [str(MAX_CHUNKS)]
    assert re.findall(r"^const MAX = (\d+);$", reference, re.M) == [str(MAX_CHUNKS)]
    assert "Math.min(MAX_CHUNKS," in project
    assert "1–16" not in reference


def test_prompt_plan_accepts_the_limit_and_rejects_above_it():
    script = "[Chunk 1]\nfirst\n\n[Chunk 64]\nlast"
    plan = make_prompt_plan(
        mode=PROMPT_FORMAT_TIMELINE, script=script, chunks=MAX_CHUNKS, chunk_seconds=5.0
    )
    assert validate_prompt_plan(plan) is plan
    assert plan["chunks"] == MAX_CHUNKS == len(plan["prompts"]) == len(plan["hashes"])
    assert plan["prompts"][0] == "first"
    assert plan["prompts"][MAX_CHUNKS - 1] == "last"
    with pytest.raises(PromptPlanError, match=f"between 1 and {MAX_CHUNKS}"):
        make_prompt_plan(
            mode=PROMPT_FORMAT_TIMELINE, script=script,
            chunks=MAX_CHUNKS + 1, chunk_seconds=5.0,
        )
    oversized = dict(plan, chunks=MAX_CHUNKS + 1)
    with pytest.raises(PromptPlanError):
        validate_prompt_plan(oversized)


def test_planning_primitives_are_not_bounded_by_the_former_limit():
    groups = physical_groups(chunks=MAX_CHUNKS, terminal_merge_enabled=True)
    assert len(groups) == MAX_CHUNKS - 1
    assert (groups[-1].start, groups[-1].end) == (MAX_CHUNKS - 1, MAX_CHUNKS)
    assert len({derive_chunk_seed(1234, index) for index in range(MAX_CHUNKS)}) == MAX_CHUNKS


def test_reference_selectors_reach_the_limit_without_changing_short_runs():
    helper = H3ContinuumReferenceImagesV39()
    bundle, = helper.pack(
        reference_use="Per chunk", reference_r1_chunks="2,17,40,64",
        reference_image_1=object(),
    )
    assert effective_reference_inputs(bundle, chunks=MAX_CHUNKS)[2]["R1"] == "2,17,40,64"
    assert effective_reference_inputs(bundle, chunks=40)[2]["R1"] == "2,17,40"
    assert effective_reference_inputs(bundle, chunks=16)[2]["R1"] == "2"
    with pytest.raises(ValueError):
        effective_reference_inputs(bundle, chunks=MAX_CHUNKS + 1)
    with pytest.raises(ValueError, match="invalid V3.9 Reference chunk selection"):
        helper.pack(reference_use="Per chunk", reference_r1_chunks="65",
                    reference_image_1=object())

    # A saved "all" must route a short run exactly like its explicit spelling.
    saved_all, = helper.pack(reference_use="Per chunk", reference_r1_chunks="all",
                             reference_image_1=object())
    explicit, = helper.pack(reference_use="Per chunk", reference_r1_chunks="1,2,3",
                            reference_image_1=object())
    assert (effective_reference_inputs(saved_all, chunks=3)[2]
            == effective_reference_inputs(explicit, chunks=3)[2])

    schedule = compile_reference_routing_schedule(
        total_chunks=MAX_CHUNKS, terminal_merge_enabled=False, mode="Custom",
        selectors_by_slot=effective_reference_inputs(bundle, chunks=MAX_CHUNKS)[2],
    )
    assert schedule.valid
    routed = [route.logical_chunk for route in schedule.logical_routes
              if route.reference_slot_ids == ("R1",)]
    assert routed == [2, 17, 40, 64]


def test_extending_a_sixteen_chunk_run_keeps_its_accepted_prefix(tmp_path):
    source = review._persist(
        tmp_path, review._contract(chunks=16), prefix=16, review_unit=None,
        updated_utc="2026-11-22T00:00:00+00:00",
    )
    before = {str(path): path.read_bytes()
              for path in source.run_root.rglob("*") if path.is_file()}
    destination = RunStorageController("phase-d-review")
    destination.run_root = source.run_root
    destination.revisions_root = source.revisions_root
    destination.prompts = [f"prompt {index}" for index in range(1, 25)]
    imported = destination._find_plan_import(review._contract(chunks=24))
    assert imported is not None
    assert imported["validated_prefix_count"] == 16
    assert imported["status"] == "review_ready"
    assert [record["sequence_index"] for record in imported["records"]] == list(range(16))
    assert imported["revision_id"] == source.revision_id
    assert before == {str(path): path.read_bytes()
                      for path in source.run_root.rglob("*") if path.is_file()}


def test_legacy_schemas_and_easy_duration_split_keep_their_own_limit():
    legacy = H3ContinuumSamplerV2.INPUT_TYPES()
    assert legacy["required"]["chunks"][1]["max"] == 16
    assert "clip_16_prompt" in legacy["optional"]
    assert "clip_17_prompt" not in legacy["optional"]
    # Raising this would re-split every Easy duration above 160 seconds.
    assert EASY_MAX_CHUNKS == 16
    plan = resolve_easy_duration(200)
    assert (plan.chunks, plan.chunk_seconds) == (16, 12.5)


def test_long_run_advisory_is_silent_up_to_the_former_limit():
    for chunks in (1, 3, 16):
        assert long_run_decode_advisory(
            chunks=chunks, chunk_seconds=30.0, output_width=1344, output_height=768,
        ) is None
    text = long_run_decode_advisory(
        chunks=32, chunk_seconds=10.0, output_width=1344, output_height=768,
    )
    # 32 x 10 s x 24 fps = 7680 frames; 7680 x 1344 x 768 x 3 x 4 bytes = 88.6 GiB.
    assert text.startswith("Long run: 32 chunks x 10s is about 7680 frames.")
    assert "1344 x 768" in text and "88.6 GiB" in text
    assert "\n" not in text


@dataclass(frozen=True)
class _Resolution:
    width: int = 1344
    height: int = 768
    actual_mp: float = 1.03
    aspect_source: str = "Manual Width / Height"
    warnings: tuple[str, ...] = ()


def _run_facade(*, chunks, size_source, advisory=long_run_decode_advisory, output=None):
    output = (object(), object(), object(), "base", object()) if output is None else output
    request = make_runtime_request(
        runtime_kwargs={},
        fixed_runtime_kwargs={},
        resolution=_Resolution(),
        size_source=size_source,
        diagnostics_mode="Off",
        diagnostics_inputs={"chunks": chunks, "chunk_seconds": 10.0},
    )
    result = execute_v38_runtime_request(
        request,
        runtime_adapter=lambda **kwargs: output,
        legacy_size_source=H3_SIZE_SOURCE_LEGACY,
        diagnostics_off="Off",
        diagnostics_full="Detailed Report",
        reference_video_size_default="Efficient",
        build_diagnostics=lambda **kwargs: pytest.fail("must not run"),
        append_status=lambda *args, **kwargs: pytest.fail("must not run"),
        long_run_advisory=advisory,
    )
    return output, result


def test_facade_leaves_short_run_status_untouched():
    output, result = _run_facade(chunks=16, size_source=H3_SIZE_SOURCE_LEGACY)
    assert result is output


def test_facade_appends_the_advisory_after_generation_for_long_runs():
    output, result = _run_facade(chunks=24, size_source=H3_SIZE_SOURCE_MANUAL)
    assert result[:3] == output[:3] and result[4] is output[4]
    lines = result[3].split("\n")
    assert lines[0] == "base"
    assert lines[1].startswith("Resolution: 1344 x 768")
    assert lines[2].startswith("Long run: 24 chunks x 10s is about 5760 frames.")
    assert len(lines) == 3


def test_facade_advisory_failure_never_discards_a_finished_generation():
    def broken(**kwargs):
        raise RuntimeError("advisory failed")

    output, result = _run_facade(
        chunks=24, size_source=H3_SIZE_SOURCE_LEGACY, advisory=broken,
    )
    assert result is output
    # A linked or missing Chunks value is not a number at this boundary.
    output, result = _run_facade(chunks=None, size_source=H3_SIZE_SOURCE_LEGACY)
    assert result is output
