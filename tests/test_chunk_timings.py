"""Chunk Timings: parsing, prompt plans, storage identity, runtime and surface."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from ComfyUI_H3_Continuum_Join import nodes as public_nodes
from ComfyUI_H3_Continuum_Join import run_storage as storage
from ComfyUI_H3_Continuum_Join.constants import (
    DIAGNOSTICS_BASIC,
    MAX_CHUNKS,
    PROMPT_MODE_FIXED,
    PROMPT_MODE_TIMELINE,
    V2_CONTINUITY_OPTIONS,
)
from ComfyUI_H3_Continuum_Join.hardening import validate_assembly_plan_contract
from ComfyUI_H3_Continuum_Join.run_storage import RunStorageController, RunStorageError
from ComfyUI_H3_Continuum_Join.v2 import nodes as v2_nodes
from ComfyUI_H3_Continuum_Join.v2 import sequence
from ComfyUI_H3_Continuum_Join.v2.h3_builder import IdentityAssets
from ComfyUI_H3_Continuum_Join.v2.prompts import (
    PromptPlanError,
    build_sampler_prompt_plan,
    make_prompt_plan,
    plan_chunk_durations,
    prompt_plan_report,
    validate_prompt_plan,
)
from ComfyUI_H3_Continuum_Join.v3 import driving_nodes
from ComfyUI_H3_Continuum_Join.v3 import nodes as v3_nodes
from ComfyUI_H3_Continuum_Join.v3.chunk_timings import (
    CHUNK_TIMINGS_TYPE,
    ChunkTimings,
    ChunkTimingsError,
    active_chunk_durations,
    active_chunk_timings,
    chunk_timings_scope,
    duration_overrides,
    format_chunk_durations,
    frames_through,
    parse_chunk_timings,
    resolve_chunk_durations,
    validate_duration_overrides,
)
from ComfyUI_H3_Continuum_Join.v3.chunk_timings_nodes import H3ContinuumChunkTimings
from ComfyUI_H3_Continuum_Join.v3.plan import (
    AssemblyPlanError,
    prepare_physical_decode_entries,
)

from . import test_preview_reuse_diagnostics as diagnostics
from . import test_v2_sequence_orchestration as orchestration
from . import test_v31_run_storage as signatures
from . import test_v38_review_run_storage as review


def _resolve(text, *, chunks=4, chunk_seconds=5.0):
    return resolve_chunk_durations(
        parse_chunk_timings(text), chunks=chunks, chunk_seconds=chunk_seconds,
    )


# --- parsing and resolution -------------------------------------------------


def test_a_list_sets_chunks_in_order_and_the_rest_keep_chunk_seconds():
    assert _resolve("5, 8, 6.5") == (5.0, 8.0, 6.5, 5.0)
    assert _resolve("5\n8;6.5s") == (5.0, 8.0, 6.5, 5.0)
    assert _resolve("10, 10, 10, 10, 10, 10", chunks=2) == (10.0, 10.0)


def test_lines_set_only_the_chunks_they_name_and_later_lines_win():
    assert _resolve("2: 8\n4 = 10 sec") == (5.0, 8.0, 5.0, 10.0)
    assert _resolve("all: 6\n3: 9", chunks=3) == (6.0, 6.0, 9.0)
    assert _resolve("1-3: 7; 2: 12") == (7.0, 12.0, 7.0, 5.0)
    assert _resolve("9: 12") is None, "chunks above the run's length are ignored"


def test_nothing_that_differs_from_chunk_seconds_resolves_to_none():
    assert parse_chunk_timings("") == ChunkTimings(())
    assert parse_chunk_timings("  \n ; ") == ChunkTimings(())
    assert resolve_chunk_durations(None, chunks=3, chunk_seconds=5.0) is None
    assert _resolve("") is None
    assert _resolve("5, 5, 5, 5") is None
    assert _resolve("2: 5.0000001") is None, "within the existing 1e-6 tolerance"
    assert _resolve("all: 8", chunk_seconds=8.0) is None


@pytest.mark.parametrize(("text", "message"), [
    ("5, 3", "outside 4-30s"),
    ("2: 31", "outside 4-30s"),
    ("5, x", "not a number"),
    ("5,,8", "empty item"),
    ("5 8", "not a number"),
    ("5\n2: 8", "mixes a list"),
    ("0: 8", "chunk selection is invalid"),
    ("2-: 8", "chunk selection is invalid"),
    (", ".join(["5"] * (MAX_CHUNKS + 1)), "the limit is"),
    ("nan", "not a number"),
])
def test_invalid_text_is_rejected_when_the_helper_runs(text, message):
    with pytest.raises(ChunkTimingsError, match=message):
        parse_chunk_timings(text)


def test_only_helper_output_is_accepted():
    with pytest.raises(ChunkTimingsError, match="must come from"):
        resolve_chunk_durations("5, 8", chunks=2, chunk_seconds=5.0)
    with pytest.raises(ChunkTimingsError, match="must be text"):
        parse_chunk_timings(5)


def test_frames_through_keeps_the_uniform_expression_without_an_override():
    for seconds in (4.0, 5.0, 5.1, 7.3, 10.0, 29.9):
        for count in range(1, 9):
            expected = int(round(count * seconds * 24.0))
            assert frames_through(count, chunk_seconds=seconds) == expected
            assert frames_through(
                count, chunk_seconds=seconds, durations=[seconds] * 8,
            ) == expected
    assert frames_through(2, chunk_seconds=5.0, durations=(5.0, 8.0, 5.0)) == 312
    assert frames_through(1, chunk_seconds=5.0, durations=(5.0, 8.0, 5.0)) == 120
    assert frames_through(3, chunk_seconds=5.0, durations=(5.0, 8.0, 5.5)) == 444


def test_status_line_collapses_runs_of_equal_lengths():
    line = format_chunk_durations((5.0, 5.0, 8.0, 5.0, 5.0, 5.0), chunk_seconds=5.0)
    assert line.startswith("Chunk timings (Experimental): 1-2: 5s, 3: 8s, 4-6: 5s; total 33s.")


# --- prompt plan --------------------------------------------------------------


def test_a_plan_without_timings_is_unchanged():
    plan = make_prompt_plan(mode=PROMPT_MODE_FIXED, script="shot", chunks=3, chunk_seconds=5.0)
    assert "chunk_durations" not in plan
    assert plan_chunk_durations(plan) is None
    assert make_prompt_plan(mode=PROMPT_MODE_FIXED, script="shot", chunks=3,
                            chunk_seconds=5.0, chunk_durations=None) == plan
    assert "5.000s per chunk" in prompt_plan_report(plan)


def test_a_plan_carries_and_reports_its_timings():
    plan = make_prompt_plan(mode=PROMPT_MODE_FIXED, script="shot", chunks=3,
                            chunk_seconds=5.0, chunk_durations=(5.0, 8.0, 5.0))
    assert plan["chunk_durations"] == [5.0, 8.0, 5.0]
    assert plan_chunk_durations(validate_prompt_plan(plan)) == (5.0, 8.0, 5.0)
    report = prompt_plan_report(plan)
    assert "per chunk" not in report.splitlines()[0]
    assert "Chunk timings (Experimental): 1: 5s, 2: 8s, 3: 5s; total 18s." in report


@pytest.mark.parametrize("durations", [[5.0, 8.0], [5.0, 5.0, 5.0], [5.0, 2.0, 5.0], "5,8,5"])
def test_a_plan_with_malformed_timings_is_rejected(durations):
    plan = make_prompt_plan(mode=PROMPT_MODE_FIXED, script="shot", chunks=3, chunk_seconds=5.0)
    plan["chunk_durations"] = durations
    with pytest.raises(PromptPlanError):
        validate_prompt_plan(plan)


def test_timeline_sections_follow_the_custom_chunk_boundaries():
    script = "[0-5s]\nwalk in\n[5-13s]\nlong talk\n[13-18s]\nleave"
    uniform = make_prompt_plan(mode=PROMPT_MODE_TIMELINE, script=script, chunks=3, chunk_seconds=5.0)
    timed = make_prompt_plan(mode=PROMPT_MODE_TIMELINE, script=script, chunks=3,
                             chunk_seconds=5.0, chunk_durations=(5.0, 8.0, 5.0))
    assert uniform["prompts"] == ["walk in", "long talk", "long talk"]
    assert timed["prompts"] == ["walk in", "long talk", "leave"]
    beyond = make_prompt_plan(mode=PROMPT_MODE_TIMELINE, script="[0-5s]\na\n[16-18s]\nb",
                              chunks=3, chunk_seconds=5.0, chunk_durations=(5.0, 8.0, 5.0))
    assert beyond["prompts"][-1] == "b", "18 s total keeps a section past 15 s usable"


def test_sampler_plan_takes_the_timings_and_adapts_a_connected_plan():
    built = build_sampler_prompt_plan(
        prompt_mode="Auto", prompt_script="shot", sequence_prompt=None, prompt_plan=None,
        chunks=3, chunk_seconds=5.0, chunk_durations=(5.0, 8.0, 5.0),
    )
    assert built["chunk_durations"] == [5.0, 8.0, 5.0]
    connected = make_prompt_plan(mode=PROMPT_MODE_FIXED, script="shot", chunks=3, chunk_seconds=5.0)
    adapted = build_sampler_prompt_plan(
        prompt_mode="Auto", prompt_script="", sequence_prompt=None, prompt_plan=connected,
        chunks=3, chunk_seconds=5.0, chunk_durations=(5.0, 8.0, 5.0),
    )
    assert adapted["chunk_durations"] == [5.0, 8.0, 5.0]
    assert "adapted connected prompt plan to Sampler chunk settings" in adapted["notes"]
    removed = build_sampler_prompt_plan(
        prompt_mode="Auto", prompt_script="", sequence_prompt=None, prompt_plan=adapted,
        chunks=3, chunk_seconds=5.0,
    )
    assert "chunk_durations" not in removed
    same = build_sampler_prompt_plan(
        prompt_mode="Auto", prompt_script="", sequence_prompt=None, prompt_plan=connected,
        chunks=3, chunk_seconds=5.0,
    )
    assert same == connected


def test_v2_orchestration_builds_the_plan_from_the_sampler_scope(monkeypatch):
    captured = []
    monkeypatch.setattr(v2_nodes, "run_sequence", lambda **kwargs: captured.append(kwargs) or "ok")
    arguments = dict(
        model=object(), clip=object(), video_vae=object(), audio_vae=None, sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]), prompt_mode="Fixed", prompt_script="Prompt",
        chunks=3, chunk_seconds=5.0, width=384, height=384,
        continuity="Balanced — 22 frames", base_seed=1, audio_continuity=True,
        exact_total_duration=False, diagnostics="Off", reroll_from_chunk=0, reroll_nonce=0,
        strict_compatibility=False, debug=False,
    )
    v2_nodes.H3ContinuumSamplerV2().run(**arguments)
    with chunk_timings_scope(parse_chunk_timings("3: 9")):
        v2_nodes.H3ContinuumSamplerV2().run(**arguments)
    assert "chunk_durations" not in captured[0]["prompt_plan"]
    assert captured[1]["prompt_plan"]["chunk_durations"] == [5.0, 5.0, 9.0]


# --- storage identity ---------------------------------------------------------


def _contract(durations=None, *, chunks=4, **kwargs):
    original = storage.build_sampling_contract

    def build(**values):
        if durations is not None:
            values["prompt_plan"] = {**values["prompt_plan"], "chunk_durations": list(durations)}
        return original(**values)

    signatures.build_sampling_contract, saved = build, signatures.build_sampling_contract
    try:
        contract, safe, _ = signatures._sampling_contract(
            chunks=chunks, reroll_from_chunk=0, **kwargs,
        )
    finally:
        signatures.build_sampling_contract = saved
    assert safe
    return contract


def test_a_run_without_timings_keeps_a_byte_identical_contract():
    torch.manual_seed(7)
    model, clip, vae = signatures._Model(), signatures._Clip(), signatures._VideoVAE()
    base = _contract(model=model, clip=clip, video_vae=vae)
    torch.manual_seed(7)
    again = _contract(model=model, clip=clip, video_vae=vae)
    assert storage._canonical(again) == storage._canonical(base)
    assert "chunk_duration_overrides" not in base
    assert all("duration_seconds" not in item for item in base["chunk_contracts"])
    assert storage.SAMPLING_CONTRACT_VERSION == 6


def test_only_chunks_with_a_different_length_change_their_contract_hash():
    torch.manual_seed(7)
    model, clip, vae = signatures._Model(), signatures._Clip(), signatures._VideoVAE()
    base = _contract(model=model, clip=clip, video_vae=vae)
    timed = _contract((5.0, 5.0, 8.0, 5.0), model=model, clip=clip, video_vae=vae)
    assert storage._hash(timed["global"]) == storage._hash(base["global"])
    assert timed["global"]["chunk_seconds"] == 5.0
    assert timed["nonce_lineage_sha256"] != base["nonce_lineage_sha256"]
    same = [a == b for a, b in zip(timed["chunk_contract_hashes"], base["chunk_contract_hashes"])]
    assert same == [True, True, False, True]
    assert timed["chunk_contracts"][2]["duration_seconds"] == 8.0
    assert timed["chunk_duration_overrides"] == [{"chunk_number": 3, "duration_seconds": 8.0}]
    longer = _contract((5.0, 5.0, 9.0, 5.0), model=model, clip=clip, video_vae=vae)
    assert len({c["chunk_contract_hashes"][2] for c in (base, timed, longer)}) == 3


def test_invalid_timings_cannot_enter_a_sampling_contract():
    with pytest.raises(RunStorageError, match="Chunk Timings contract is invalid"):
        _contract((5.0, 5.0, 2.0, 5.0))
    with pytest.raises(RunStorageError, match="Chunk Timings contract is invalid"):
        _contract((5.0, 8.0))


def test_override_records_are_canonical():
    assert duration_overrides(None, chunk_seconds=5.0) == []
    records = duration_overrides((5.0, 8.0, 5.0, 6.5), chunk_seconds=5.0)
    assert validate_duration_overrides(records, chunk_count=4, chunk_seconds=5.0) == records
    for bad in ([], [{"chunk_number": 2, "duration_seconds": 5.0}],
                [{"chunk_number": 3, "duration_seconds": 8.0}, {"chunk_number": 2, "duration_seconds": 8.0}],
                [{"chunk_number": 5, "duration_seconds": 8.0}],
                [{"chunk_number": 2, "duration_seconds": 8.0, "extra": 1}]):
        with pytest.raises(ChunkTimingsError):
            validate_duration_overrides(bad, chunk_count=4, chunk_seconds=5.0)


def _review_contract(durations=None, *, chunks=4):
    contract = review._contract(chunks=chunks)
    if durations is None:
        return contract
    plain = {key: contract[key] for key in (
        "global", "chunk_count", "prompt_mode", "prompt_hashes", "last_frame_hash",
    )}
    plain["chunk_duration_overrides"] = duration_overrides(durations, chunk_seconds=5.0)
    plain["nonce_lineage_sha256"] = storage._nonce_lineage_hash(plain)
    return storage._apply_reroll_branch_contract(
        plain, boundary=0, requested_nonce=0, effective_nonce=0,
    )


def _import(tmp_path, source_contract, current_contract, *, prefix):
    source = review._persist(
        tmp_path, source_contract, prefix=prefix, review_unit=(prefix, prefix),
        updated_utc="2026-11-28T00:00:00+00:00",
    )
    before = {str(p): p.read_bytes() for p in source.run_root.rglob("*") if p.is_file()}
    destination = RunStorageController("phase-d-review")
    destination.run_root = source.run_root
    destination.revisions_root = source.revisions_root
    destination.prompts = [
        f"prompt {index}" for index in range(1, current_contract["chunk_count"] + 1)
    ]
    imported = destination._find_plan_import(current_contract)
    assert before == {str(p): p.read_bytes() for p in source.run_root.rglob("*") if p.is_file()}
    return 0 if imported is None else imported["validated_prefix_count"]


def test_lengthening_a_later_chunk_keeps_every_chunk_before_it(tmp_path):
    later = _review_contract((5.0, 5.0, 5.0, 8.0))
    assert _import(tmp_path, _review_contract(), later, prefix=3) == 3


def test_changing_an_accepted_chunk_regenerates_from_that_chunk(tmp_path):
    earlier = _review_contract((5.0, 8.0, 5.0, 5.0))
    assert _import(tmp_path / "added", _review_contract(), earlier, prefix=3) == 1
    source = _review_contract((5.0, 5.0, 8.0, 5.0))
    changed = _review_contract((5.0, 5.0, 9.0, 5.0))
    assert _import(tmp_path / "changed", source, changed, prefix=3) == 2
    assert _import(tmp_path / "removed", source, _review_contract(), prefix=3) == 2


def test_stored_manifest_with_inconsistent_duration_records_is_rejected(tmp_path):
    contract = _review_contract((5.0, 5.0, 8.0, 5.0))
    source = review._persist(tmp_path, contract, prefix=3, review_unit=(3, 3),
                             updated_utc="2026-11-28T00:00:00+00:00")
    manifest = copy.deepcopy(source.manifest)
    storage._manifest_sampling_identity(manifest)
    manifest["contract"]["chunk_contracts"][2]["duration_seconds"] = 9.0
    with pytest.raises(RunStorageError, match="duration/chunk contracts disagree"):
        storage._manifest_sampling_identity(manifest)
    manifest = copy.deepcopy(source.manifest)
    manifest["contract"]["chunk_duration_overrides"] = [{"chunk_number": 3, "duration_seconds": 5.0}]
    with pytest.raises(RunStorageError, match="chunk duration contract is invalid"):
        storage._manifest_sampling_identity(manifest)


def _review_queue(path, kwargs, *, new_position=None, previous=()):
    controller = diagnostics.controller_at(path)
    session = controller.prepare(**kwargs)
    reused = [] if session is None else list(session["chunks"])
    if new_position is not None:
        assert len(reused) == new_position
        entry = review._entry(new_position, controller.contract)
        controller.commit_group((entry,), positions=(new_position,))
        controller.mark_review_group(
            start=new_position + 1, end=new_position + 1, physical_group=new_position + 1,
        )
        controller.finalize(
            session={"session_id": f"queue-{new_position}", "chunks": [*previous, entry]},
            report="CPU fixture", review_pause_metadata=controller.review_pause_metadata(),
        )
        return controller, [*previous, entry]
    return controller, reused


def _with_durations(kwargs, durations):
    result = dict(kwargs)
    result["prompt_plan"] = {**kwargs["prompt_plan"], "chunk_durations": list(durations)}
    return result


def test_real_run_storage_keeps_the_prefix_when_a_later_chunk_is_lengthened(tmp_path):
    torch.manual_seed(7)
    kwargs = diagnostics.prepare_kwargs(signatures._Model())
    _, entries = _review_queue(tmp_path, kwargs, new_position=0)
    plain, entries = _review_queue(tmp_path, kwargs, new_position=1, previous=entries)
    assert "chunk_duration_overrides" not in plain.contract
    plain_files = {str(p): p.read_bytes() for p in plain.revision_root.rglob("*") if p.is_file()}

    timed, reused = _review_queue(tmp_path, _with_durations(kwargs, (5.0, 5.0, 8.0)))
    assert len(reused) == 2 and timed.reused_count == 2
    assert timed.contract["chunk_duration_overrides"] == [{"chunk_number": 3, "duration_seconds": 8.0}]
    assert timed.revision_id != plain.revision_id
    assert storage._hash(timed.contract["global"]) == storage._hash(plain.contract["global"])
    assert plain_files == {
        str(p): p.read_bytes() for p in plain.revision_root.rglob("*") if p.is_file()
    }, "the Take generated without timings is not rewritten"


def test_real_run_storage_regenerates_from_a_changed_chunk_and_resumes_its_own_run(tmp_path):
    torch.manual_seed(7)
    kwargs = diagnostics.prepare_kwargs(signatures._Model())
    timed = _with_durations(kwargs, (5.0, 8.0, 5.0))
    _, entries = _review_queue(tmp_path, timed, new_position=0)
    first, entries = _review_queue(tmp_path, timed, new_position=1, previous=entries)
    again, reused = _review_queue(tmp_path, timed)
    assert len(reused) == 2 and again.contract_sha256 == first.contract_sha256
    _, reused = _review_queue(tmp_path, _with_durations(kwargs, (5.0, 9.0, 5.0)))
    assert len(reused) == 1
    _, reused = _review_queue(tmp_path, kwargs)
    assert len(reused) == 1, "removing the timings keeps only the chunk before the change"


# --- sequence runtime -----------------------------------------------------------


def _run_three_chunks(monkeypatch, durations, *, exact=True, session=None):
    width, height = 96, 64
    fake_model = SimpleNamespace(
        name="base", model=SimpleNamespace(diffusion_model=SimpleNamespace()),
        model_options={}, wrappers={}, model_dtype=lambda: torch.bfloat16, model_size=lambda: 0,
    )
    assets = IdentityAssets(None, None, None, None, "none")
    sampled = []
    monkeypatch.setattr(sequence, "check_comfy_h3_runtime", lambda: [])
    monkeypatch.setattr(sequence, "clone_model_for_chunk", lambda model, **kwargs: SimpleNamespace(
        name=model.name, model=model.model, model_options={"transformer_options": {}},
        wrappers=model.wrappers, model_dtype=model.model_dtype, model_size=model.model_size,
    ))
    monkeypatch.setattr(sequence, "prepare_identity_assets", lambda *a, **k: assets)
    monkeypatch.setattr(sequence, "encode_identity_latents", lambda *a, **k: assets)
    monkeypatch.setattr(sequence, "empty_h3_latent", orchestration._latent)
    monkeypatch.setattr(sequence, "accelerator_summary", lambda model: "accelerators")

    def sample_chunk(**kwargs):
        sampled.append(kwargs["latent"]["samples"])
        return kwargs["latent"]

    def decode_sequence(**kwargs):
        frames = sum(int(entry["plan"]["net_frames"]) for entry in kwargs["entries"])
        return (torch.zeros(frames, 2, 2, 3),
                {"waveform": torch.zeros(1, 2, round(frames / 24 * 32000)), "sample_rate": 32000}, [])

    monkeypatch.setattr(sequence, "sample_chunk", sample_chunk)
    monkeypatch.setattr(sequence, "decode_sequence", decode_sequence)
    plan = make_prompt_plan(mode=PROMPT_MODE_FIXED, script="continuous shot", chunks=3,
                            chunk_seconds=5.0, chunk_durations=durations)
    result = sequence.run_sequence(
        model=fake_model, clip=orchestration.FakeClip([]), video_vae=object(),
        audio_vae=object(), sampler=object(), sigmas=torch.tensor([1.0, 0.0]),
        first_frame=None, last_frame=None, prompt_plan=plan, width=width, height=height,
        continuity=V2_CONTINUITY_OPTIONS[1], base_seed=42, audio_continuity=True,
        exact_total_duration=exact, diagnostics_mode=DIAGNOSTICS_BASIC,
        reroll_from_chunk=0, reroll_nonce=0, strict_compatibility=True, debug=False,
        session=session,
    )
    return result, sampled


def _cumulative_net(session):
    total, result = 0, []
    for entry in session["chunks"]:
        total += int(entry["plan"]["net_frames"])
        result.append(total)
    return result


def test_sequence_samples_each_chunk_at_its_own_length(monkeypatch):
    uniform, uniform_latents = _run_three_chunks(monkeypatch, None)
    timed, timed_latents = _run_three_chunks(monkeypatch, (5.0, 8.0, 5.0))
    images, _, _, session, report = timed
    plain_session = uniform[3]
    assert [e["plan"]["requested_extend_seconds"] for e in plain_session["chunks"]] == [5.0] * 3
    assert [e["plan"]["requested_extend_seconds"] for e in session["chunks"]] == [5.0, 8.0, 5.0]
    assert session["chunks"][0]["plan"] == plain_session["chunks"][0]["plan"]
    assert torch.equal(timed_latents[0].parts[0], uniform_latents[0].parts[0])
    assert timed_latents[1].parts[0].shape[2] > uniform_latents[1].parts[0].shape[2]
    plain_cumulative, timed_cumulative = _cumulative_net(plain_session), _cumulative_net(session)
    # Each chunk aims at its cumulative target and lands within one step of
    # the H3 17k+5 frame grid, exactly as uniform chunks already do.
    assert plain_cumulative == [124, 243, 362]
    assert timed_cumulative == [124, 311, 430]
    for target, reached in zip((120, 312, 432), timed_cumulative):
        assert abs(reached - target) < 17
    assert uniform[0].shape[0] == 360 and images.shape[0] == 432, "exact total length"
    assert session["chunk_seconds"] == 5.0
    assert session["settings"]["chunk_durations"] == [5.0, 8.0, 5.0]
    assert "chunk_durations" not in plain_session["settings"]
    assert "Chunk timings (Experimental): 1: 5s, 2: 8s, 3: 5s; total 18s." in report
    assert "Chunk timings" not in uniform[-1]


def test_a_saved_session_is_reused_up_to_the_first_changed_length(monkeypatch):
    plain = _run_three_chunks(monkeypatch, None, exact=False)[0][3]
    reused = _run_three_chunks(monkeypatch, (5.0, 5.0, 8.0), exact=False, session=plain)[0]
    assert "session reuse stopped before chunk 3: chunk length changed" in reused[-1]
    assert [bool(e.get("reused")) for e in reused[3]["chunks"]] == [True, True, False]
    timed = reused[3]
    again = _run_three_chunks(monkeypatch, (5.0, 5.0, 8.0), exact=False, session=timed)[0]
    assert all(bool(e.get("reused")) for e in again[3]["chunks"])
    back = _run_three_chunks(monkeypatch, None, exact=False, session=timed)[0]
    assert [bool(e.get("reused")) for e in back[3]["chunks"]] == [True, True, False]


def test_session_timing_limit_handles_absent_short_and_malformed_records():
    limit = sequence._session_timing_limit
    plain = {"settings": {}}
    assert limit(plain, chunks=3, chunk_seconds=5.0, chunk_durations=None) == 3
    assert limit(plain, chunks=3, chunk_seconds=5.0, chunk_durations=(5.0, 8.0, 5.0)) == 1
    timed = {"settings": {"chunk_durations": [5.0, 8.0]}}
    assert limit(timed, chunks=3, chunk_seconds=5.0, chunk_durations=(5.0, 8.0, 5.0)) == 3
    assert limit(timed, chunks=3, chunk_seconds=5.0, chunk_durations=None) == 1
    assert limit({"settings": {"chunk_durations": "5,8"}}, chunks=2,
                 chunk_seconds=5.0, chunk_durations=None) == 0


def test_terminal_merge_needs_the_final_two_chunks_at_five_seconds():
    common = dict(multi_chunk_flf=True, chunks=3, chunk_seconds=5.0,
                  prompt_hashes=["a", "b", "c"], timeline_video_source=None)
    assert sequence._terminal_flf_merge_enabled(**common)
    assert sequence._terminal_flf_merge_enabled(**common, chunk_durations=(8.0, 5.0, 5.0))
    assert not sequence._terminal_flf_merge_enabled(**common, chunk_durations=(5.0, 8.0, 5.0))
    assert not sequence._terminal_flf_merge_enabled(**common, chunk_durations=(5.0, 5.0, 6.0))


def test_the_legacy_timeline_video_input_is_refused_with_timings(monkeypatch):
    plan = make_prompt_plan(mode=PROMPT_MODE_FIXED, script="shot", chunks=2,
                            chunk_seconds=5.0, chunk_durations=(5.0, 8.0))
    with pytest.raises(sequence.SequenceRuntimeError, match="legacy Timeline Video"):
        sequence._run_runtime_internal(
            model=object(), clip=object(), video_vae=object(), audio_vae=None,
            sampler=object(), sigmas=torch.tensor([1.0, 0.0]), first_frame=None,
            last_frame=None, prompt_plan=plan, width=96, height=64,
            continuity=V2_CONTINUITY_OPTIONS[1], base_seed=1, audio_continuity=True,
            exact_total_duration=False, diagnostics_mode=DIAGNOSTICS_BASIC,
            reroll_from_chunk=0, reroll_nonce=0, strict_compatibility=False, debug=False,
            timeline_video_source=object(),
        )


# --- assembly plan --------------------------------------------------------------


def test_assembly_target_is_the_sum_of_chunk_lengths(monkeypatch):
    session = _run_three_chunks(monkeypatch, (5.0, 8.0, 5.0), exact=False)[0][3]
    entries = session["chunks"]
    _, uniform = prepare_physical_decode_entries(
        entries, chunk_seconds=5.0, preserve_final_frame=False, terminal_merged=False,
    )
    assert uniform["target_frames"] == 360 and "chunk_durations" not in uniform
    _, same = prepare_physical_decode_entries(
        entries, chunk_seconds=5.0, preserve_final_frame=False, terminal_merged=False,
        chunk_durations=(5.0, 5.0, 5.0),
    )
    assert same == uniform
    _, timed = prepare_physical_decode_entries(
        entries, chunk_seconds=5.0, preserve_final_frame=False, terminal_merged=False,
        chunk_durations=(5.0, 8.0, 5.0),
    )
    assert timed["target_frames"] == 432 and timed["chunk_durations"] == [5.0, 8.0, 5.0]
    _, scoped = prepare_physical_decode_entries(
        entries[1:2], chunk_seconds=5.0, preserve_final_frame=False, terminal_merged=False,
        chunk_durations=(8.0,),
    )
    assert scoped["target_frames"] == 192
    broken = dict(timed, target_frames=360)
    with pytest.raises(ValueError, match="sum\\(chunk_durations\\)"):
        validate_assembly_plan_contract(broken)
    with pytest.raises(ValueError, match="does not match chunks"):
        validate_assembly_plan_contract(dict(timed, chunk_durations=[5.0, 8.0]))
    with pytest.raises(AssemblyPlanError, match="do not match the assembled chunks"):
        prepare_physical_decode_entries(
            entries, chunk_seconds=5.0, preserve_final_frame=False, terminal_merged=False,
            chunk_durations=(5.0, 8.0),
        )


def test_v3_sampler_plans_decode_with_the_scoped_timings(monkeypatch):
    session = _run_three_chunks(monkeypatch, (5.0, 8.0, 5.0), exact=False)[0][3]
    entries = session["chunks"]

    class FakeV2:
        def run(self, **kwargs):
            return list(entries), None, session, "status"

    monkeypatch.setattr(v3_nodes, "H3ContinuumSamplerV2", FakeV2)
    arguments = dict(
        model=object(), clip=object(), video_vae=object(), sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]), sequence_prompt="shot", prompt_mode="Auto",
        chunks=3, chunk_seconds=5.0, width=96, height=64,
        continuity=V2_CONTINUITY_OPTIONS[1], base_seed=1,
    )
    plain_plan = v3_nodes.H3ContinuumSamplerV3().run(**arguments)[2]
    with chunk_timings_scope(parse_chunk_timings("2: 8")):
        timed_plan = v3_nodes.H3ContinuumSamplerV3().run(**arguments)[2]
    assert plain_plan["target_frames"] == 360 and "chunk_durations" not in plain_plan
    assert timed_plan["target_frames"] == 432
    assert timed_plan["chunk_durations"] == [5.0, 8.0, 5.0]


# --- public surface -------------------------------------------------------------


def test_helper_node_is_public_and_builds_a_bundle():
    assert public_nodes.NODE_CLASS_MAPPINGS["H3ContinuumChunkTimings"] is H3ContinuumChunkTimings
    assert public_nodes.NODE_DISPLAY_NAME_MAPPINGS["H3ContinuumChunkTimings"] == (
        "H3 Continuum Chunk Timings (Experimental)"
    )
    assert H3ContinuumChunkTimings.RETURN_TYPES == (CHUNK_TIMINGS_TYPE,)
    schema = H3ContinuumChunkTimings.INPUT_TYPES()
    assert tuple(schema["required"]) == ("timings",) and "optional" not in schema
    (bundle,) = H3ContinuumChunkTimings().build(timings="2: 8")
    assert resolve_chunk_durations(bundle, chunks=3, chunk_seconds=5.0) == (5.0, 8.0, 5.0)
    assert H3ContinuumChunkTimings().build() == (ChunkTimings(()),)


def test_sampler_input_is_appended_last_and_only_on_v39():
    old = driving_nodes.H3ContinuumSamplerV38.INPUT_TYPES()
    new = driving_nodes.H3ContinuumSamplerV39.INPUT_TYPES()
    assert "chunk_timings" not in old["optional"] and "chunk_timings" not in old["required"]
    assert new["required"] == old["required"]
    assert list(new["optional"])[-1] == "chunk_timings"
    assert new["optional"]["chunk_timings"][0] == CHUNK_TIMINGS_TYPE


def test_sampler_scopes_the_timings_and_does_not_forward_them(monkeypatch):
    seen = []
    plan = {"mode": "Custom", "accepted_chunks": 0, "connected_slots": [], "groups": []}

    def old_run(_self, **kwargs):
        seen.append((active_chunk_timings(), active_chunk_durations(chunks=3, chunk_seconds=5.0), kwargs))
        return (object(), object(), {"reference_routing_v1": plan}, "status", object())

    monkeypatch.setattr(driving_nodes.H3ContinuumSamplerV38, "run", old_run)
    sampler = driving_nodes.H3ContinuumSamplerV39()
    sampler.run(chunks=3)
    timings = parse_chunk_timings("3: 10")
    sampler.run(chunks=3, chunk_timings=timings)
    assert seen[0][0] is None and seen[0][1] is None
    assert seen[1][0] is timings and seen[1][1] == (5.0, 5.0, 10.0)
    assert all("chunk_timings" not in kwargs for *_, kwargs in seen)
    assert seen[0][2] == seen[1][2], "the runtime call itself is unchanged by timings"
    assert active_chunk_timings() is None
    with pytest.raises(ChunkTimingsError):
        sampler.run(chunks=3, chunk_timings="3: 10")


def test_review_decode_scope_targets_the_reviewed_chunks_own_length():
    from . import test_v39_stability_r3 as stability

    plain, _ = stability.scope(chunk=2)
    with chunk_timings_scope(parse_chunk_timings("2: 8")):
        timed, full = stability.scope(chunk=2)
    assert plain[2]["target_frames"] == 120 and "chunk_durations" not in plain[2]
    assert timed[2]["target_frames"] == 192
    assert timed[2]["chunk_durations"] == [8.0]
    assert full[2]["target_frames"] == 360, "the fixture's own full plan is untouched"
    with chunk_timings_scope(parse_chunk_timings("3: 8")):
        other, _ = stability.scope(chunk=2)
    assert other[2]["target_frames"] == 120 and "chunk_durations" not in other[2]
