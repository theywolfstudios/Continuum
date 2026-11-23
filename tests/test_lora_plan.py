"""LoRA Plan: schedule, storage identity, runtime and public surface."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from ComfyUI_H3_Continuum_Join import nodes as public_nodes
from ComfyUI_H3_Continuum_Join import run_storage as storage
from ComfyUI_H3_Continuum_Join.constants import (
    DIAGNOSTICS_BASIC,
    PROMPT_MODE_FIXED,
    V2_CONTINUITY_OPTIONS,
)
from ComfyUI_H3_Continuum_Join.run_storage import RunStorageController, RunStorageError
from ComfyUI_H3_Continuum_Join.v2 import sequence
from ComfyUI_H3_Continuum_Join.v2.h3_builder import IdentityAssets
from ComfyUI_H3_Continuum_Join.v2.prompts import make_prompt_plan
from ComfyUI_H3_Continuum_Join.v3 import driving_nodes, lora_runtime
from ComfyUI_H3_Continuum_Join.v3.lora_plan import (
    LORA_PLAN_TYPE,
    LoraPlan,
    LoraPlanError,
    LoraUse,
    append_lora_plan_row,
    compile_lora_schedule,
    lora_group_record,
    validate_lora_group_contracts,
)
from ComfyUI_H3_Continuum_Join.v3.lora_plan_nodes import H3ContinuumLoraPlan
from ComfyUI_H3_Continuum_Join.v3.lora_runtime import (
    LoraPlanRuntime,
    active_lora_plan_runtime,
    file_sha256,
    lora_plan_scope,
    make_lora_plan_runtime,
)

from . import test_preview_reuse_diagnostics as diagnostics
from . import test_v2_sequence_orchestration as orchestration
from . import test_v31_run_storage as signatures
from . import test_v38_review_run_storage as review


SHA_A, SHA_B = "a" * 64, "b" * 64


def _plan(*rows):
    plan = None
    for name, strength, chunks in rows:
        plan = append_lora_plan_row(plan, lora_name=name, strength=strength, chunks=chunks)
    return plan


# --- plan and schedule -----------------------------------------------------


def test_rows_chain_in_order_without_mutating_the_previous_plan():
    first = _plan(("style.safetensors", 0.8, "all"))
    second = append_lora_plan_row(first, lora_name="motion.safetensors", strength=1, chunks="2-3")
    assert [row.lora_name for row in first.rows] == ["style.safetensors"]
    assert [(row.lora_name, row.strength, row.chunks) for row in second.rows] == [
        ("style.safetensors", 0.8, "all"), ("motion.safetensors", 1.0, "2-3"),
    ]


@pytest.mark.parametrize("bad", [
    dict(lora_name="a", strength=1.0, chunks="1,,2"),
    dict(lora_name="a", strength=1.0, chunks="3-1"),
    dict(lora_name="a", strength=1.0, chunks="65"),
    dict(lora_name="a", strength=1.0, chunks=""),
    dict(lora_name="a", strength=1.0, chunks=3),
    dict(lora_name="a", strength=float("nan"), chunks="all"),
    dict(lora_name="a", strength=True, chunks="all"),
    dict(lora_name="", strength=1.0, chunks="all"),
])
def test_invalid_rows_are_rejected_when_the_plan_is_built(bad):
    with pytest.raises(LoraPlanError):
        append_lora_plan_row(None, **bad)


def test_plan_input_must_be_a_lora_plan():
    with pytest.raises(LoraPlanError, match="H3 Continuum LoRA Plan"):
        append_lora_plan_row({"rows": []}, lora_name="a", strength=1.0, chunks="all")
    with pytest.raises(LoraPlanError):
        make_lora_plan_runtime("style.safetensors")


def test_schedule_resolves_selectors_inactive_rows_and_order():
    plan = _plan(
        ("style", 0.8, "all"),
        ("motion", 1.0, "2-3,9"),      # 9 waits for a longer run
        ("disabled", 0.0, "all"),      # strength 0 is Core's no-op
        ("None", 1.0, "all"),
        ("off", 1.0, "off"),
        ("style", 0.2, "3"),           # the same file twice is allowed
    )
    schedule = compile_lora_schedule(plan, total_chunks=4, terminal_merge_enabled=False)
    assert [group.loras for group in schedule.groups] == [
        (LoraUse("style", 0.8),),
        (LoraUse("style", 0.8), LoraUse("motion", 1.0)),
        (LoraUse("style", 0.8), LoraUse("motion", 1.0), LoraUse("style", 0.2)),
        (LoraUse("style", 0.8),),
    ]
    assert [group.physical_group for group in schedule.active_groups] == [1, 2, 3, 4]
    longer = compile_lora_schedule(plan, total_chunks=9, terminal_merge_enabled=False)
    assert LoraUse("motion", 1.0) in longer.groups[8].loras


def test_terminal_merged_pair_needs_one_lora_set():
    same = compile_lora_schedule(
        _plan(("style", 1.0, "3-4")), total_chunks=4, terminal_merge_enabled=True,
    )
    assert [(g.physical_group, g.logical_chunks) for g in same.active_groups] == [(3, (3, 4))]
    with pytest.raises(LoraPlanError, match="chunks 3 and 4 are sampled together"):
        compile_lora_schedule(
            _plan(("style", 1.0, "4")), total_chunks=4, terminal_merge_enabled=True,
        )


def _records(*groups, chunk_count=4, terminal=False):
    records = [
        lora_group_record(physical_group=group, logical_chunks=chunks, loras=loras)
        for group, chunks, loras in groups
    ]
    return validate_lora_group_contracts(
        records, chunk_count=chunk_count, terminal_merge_enabled=terminal,
    )


def test_group_records_are_validated_against_topology_and_content():
    records = _records((3, (3,), ((SHA_A, 0.8),)))
    assert records[0]["descriptor"]["loras"] == [{"sha256": SHA_A, "strength_model": 0.8}]

    def broken(change):
        value = copy.deepcopy(records)
        change(value)
        with pytest.raises(LoraPlanError):
            validate_lora_group_contracts(value, chunk_count=4, terminal_merge_enabled=False)

    broken(lambda v: v[0]["descriptor"]["loras"][0].update(strength_model=0.9))  # stale SHA
    broken(lambda v: v[0].update(sha256="0" * 64))
    broken(lambda v: v[0]["descriptor"].update(logical_chunks=[3, 4]))
    broken(lambda v: v.append(copy.deepcopy(v[0])))                               # not ascending
    broken(lambda v: v[0]["descriptor"].update(extra=1))
    with pytest.raises(LoraPlanError):
        validate_lora_group_contracts([], chunk_count=4, terminal_merge_enabled=False)
    with pytest.raises(LoraPlanError):
        lora_group_record(physical_group=1, logical_chunks=(1,), loras=())
    with pytest.raises(LoraPlanError):  # group 4 does not exist once 3-4 are merged
        _records((4, (4,), ((SHA_A, 1.0),)), terminal=True)


# --- storage identity ------------------------------------------------------


def _contract(lora_groups=None, *, chunks=4, **kwargs):
    contract, safe, _ = signatures._sampling_contract(
        chunks=chunks, reroll_from_chunk=0, **kwargs,
    ) if lora_groups is None else _contract_with(lora_groups, chunks=chunks, **kwargs)
    assert safe
    return contract


def _contract_with(lora_groups, *, chunks, **kwargs):
    original = storage.build_sampling_contract

    def build(**values):
        return original(**values, lora_group_contracts=lora_groups)

    signatures.build_sampling_contract, saved = build, signatures.build_sampling_contract
    try:
        return signatures._sampling_contract(chunks=chunks, reroll_from_chunk=0, **kwargs)
    finally:
        signatures.build_sampling_contract = saved


def test_absent_or_empty_plan_leaves_the_sampling_contract_byte_identical():
    torch.manual_seed(7)
    model, clip, vae = signatures._Model(), signatures._Clip(), signatures._VideoVAE()
    base = _contract(model=model, clip=clip, video_vae=vae)
    for empty in ([], None):
        same, safe, _ = _contract_with(empty, chunks=4, model=model, clip=clip, video_vae=vae)
        assert safe
        assert storage._canonical(same) == storage._canonical(base)
        assert storage.revision_identity(same) == storage.revision_identity(base)
    assert "lora_group_contracts" not in base
    assert all("physical_group_lora_sha256" not in item for item in base["chunk_contracts"])
    assert storage.SAMPLING_CONTRACT_VERSION == 6


def test_only_chunks_that_receive_a_lora_change_their_contract_hash():
    torch.manual_seed(7)
    model, clip, vae = signatures._Model(), signatures._Clip(), signatures._VideoVAE()
    base = _contract(model=model, clip=clip, video_vae=vae)
    records = _records((3, (3,), ((SHA_A, 0.8),)))
    planned, safe, _ = _contract_with(records, chunks=4, model=model, clip=clip, video_vae=vae)
    assert safe
    assert storage._hash(planned["global"]) == storage._hash(base["global"])
    assert planned["nonce_lineage_sha256"] != base["nonce_lineage_sha256"]
    same = [a == b for a, b in zip(planned["chunk_contract_hashes"], base["chunk_contract_hashes"])]
    assert same == [True, True, False, True]
    assert planned["chunk_contracts"][2]["physical_group_lora_sha256"] == records[0]["sha256"]
    assert planned["lora_group_contracts"] == records

    stronger, _, _ = _contract_with(
        _records((3, (3,), ((SHA_A, 0.9),))), chunks=4, model=model, clip=clip, video_vae=vae,
    )
    other_file, _, _ = _contract_with(
        _records((3, (3,), ((SHA_B, 0.8),))), chunks=4, model=model, clip=clip, video_vae=vae,
    )
    hashes = {c["chunk_contract_hashes"][2] for c in (base, planned, stronger, other_file)}
    assert len(hashes) == 4


def test_invalid_lora_group_contracts_cannot_enter_a_sampling_contract():
    records = _records((3, (3,), ((SHA_A, 0.8),)))
    records[0]["sha256"] = "0" * 64
    with pytest.raises(RunStorageError, match="LoRA Plan contract is invalid"):
        _contract_with(records, chunks=4)


def _review_contract(lora_groups=None, *, chunks=4, terminal=False):
    """The review fixture's contract, with LoRA records added before hashing."""

    contract = review._contract(chunks=chunks, terminal=terminal)
    if not lora_groups:
        return contract
    plain = {key: contract[key] for key in (
        "global", "chunk_count", "prompt_mode", "prompt_hashes", "last_frame_hash",
    )}
    plain["lora_group_contracts"] = lora_groups
    plain["nonce_lineage_sha256"] = storage._nonce_lineage_hash(plain)
    return storage._apply_reroll_branch_contract(
        plain, boundary=0, requested_nonce=0, effective_nonce=0,
    )


def _import(tmp_path, source_contract, current_contract, *, prefix):
    source = review._persist(
        tmp_path, source_contract, prefix=prefix, review_unit=(prefix, prefix),
        updated_utc="2026-11-23T00:00:00+00:00",
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


def test_attaching_a_plan_mid_run_keeps_every_chunk_before_the_first_lora(tmp_path):
    later = _review_contract(_records((4, (4,), ((SHA_A, 0.8),))))
    assert _import(tmp_path, _review_contract(), later, prefix=3) == 3


def test_a_lora_on_an_accepted_chunk_regenerates_from_that_chunk(tmp_path):
    earlier = _review_contract(_records((2, (2,), ((SHA_A, 0.8),))))
    assert _import(tmp_path, _review_contract(), earlier, prefix=3) == 1


def test_changing_or_removing_a_lora_keeps_only_the_chunks_before_it(tmp_path):
    source = _review_contract(_records((3, (3,), ((SHA_A, 0.8),))))
    changed = _review_contract(_records((3, (3,), ((SHA_A, 0.5),))))
    assert _import(tmp_path / "changed", source, changed, prefix=3) == 2
    assert _import(tmp_path / "removed", source, _review_contract(), prefix=3) == 2
    moved = _review_contract(_records((4, (4,), ((SHA_A, 0.8),))))
    assert _import(tmp_path / "moved", source, moved, prefix=3) == 2


def test_the_same_plan_resumes_its_own_saved_prefix(tmp_path):
    contract = _review_contract(_records((2, (2,), ((SHA_A, 0.8),)), (3, (3,), ((SHA_B, 1.0),))))
    review._persist(tmp_path, contract, prefix=3, review_unit=(3, 3),
                    updated_utc="2026-11-23T00:00:00+00:00")
    reader = review._controller(tmp_path, contract)
    reader.configure_review(generation_mode=review.GENERATION_MODE_REVIEW,
                            review_action=review.REVIEW_ACTION_CONTINUE)
    assert len(reader._load_validated_review_prefix(contract, resume_safe=True).entries) == 3


def test_review_continue_after_attaching_a_plan_reuses_the_prefix(tmp_path):
    review._persist(tmp_path, _review_contract(), prefix=3, review_unit=(3, 3),
                    updated_utc="2026-11-23T00:00:00+00:00")
    current = _review_contract(_records((4, (4,), ((SHA_A, 0.8),))))
    reader = review._controller(tmp_path, current)
    reader.configure_review(generation_mode=review.GENERATION_MODE_REVIEW,
                            review_action=review.REVIEW_ACTION_CONTINUE)
    assert len(reader._load_validated_review_prefix(current, resume_safe=True).entries) == 3


def test_stored_manifest_with_inconsistent_lora_contracts_is_rejected(tmp_path):
    contract = _review_contract(_records((3, (3,), ((SHA_A, 0.8),))))
    source = review._persist(tmp_path, contract, prefix=3, review_unit=(3, 3),
                             updated_utc="2026-11-23T00:00:00+00:00")
    storage._manifest_sampling_identity(source.manifest)

    stripped = copy.deepcopy(source.manifest)
    del stripped["contract"]["lora_group_contracts"]
    with pytest.raises(RunStorageError, match="LoRA group contract is invalid"):
        storage._manifest_sampling_identity(stripped)

    moved = copy.deepcopy(source.manifest)
    chunks = moved["contract"]["chunk_contracts"]
    chunks[1]["physical_group_lora_sha256"] = chunks[2].pop("physical_group_lora_sha256")
    with pytest.raises(RunStorageError, match="LoRA group/chunk contracts disagree"):
        storage._manifest_sampling_identity(moved)


# --- runtime ---------------------------------------------------------------


class _FakeCore:
    def __init__(self):
        self.resolved, self.hashed, self.loaded, self.applied = [], [], [], []

    def resolve(self, name):
        self.resolved.append(name)
        if name == "missing":
            raise FileNotFoundError(name)
        return f"/loras/{name}"

    def hash(self, path):
        self.hashed.append(path)
        return {"/loras/style": SHA_A, "/loras/motion": SHA_B}[path]

    def load(self, path):
        self.loaded.append(path)
        return {"file": path}, {"meta": path}

    def apply(self, model, lora, strength, metadata):
        self.applied.append((model.name, lora["file"], strength, metadata["meta"]))
        return SimpleNamespace(name=f"{model.name}+{lora['file']}@{strength:g}"), 5

    def runtime(self, plan):
        return LoraPlanRuntime(plan, resolve_path=self.resolve, hash_file=self.hash,
                               load_file=self.load, apply_lora=self.apply)


def test_runtime_hashes_before_sampling_and_loads_only_when_a_group_is_sampled():
    core = _FakeCore()
    runtime = core.runtime(_plan(("style", 0.8, "2-4"), ("motion", 1.0, "4")))
    records = runtime.compile(chunks=5, terminal_merge_enabled=False)
    assert [r["descriptor"]["physical_group"] for r in records] == [2, 3, 4]
    assert records[2]["descriptor"]["loras"] == [
        {"sha256": SHA_A, "strength_model": 0.8}, {"sha256": SHA_B, "strength_model": 1.0},
    ]
    validate_lora_group_contracts(records, chunk_count=5, terminal_merge_enabled=False)
    assert core.hashed == ["/loras/style", "/loras/motion"] and core.loaded == []

    base = SimpleNamespace(name="base")
    assert runtime.model_for_group(base, physical_group=1) is base
    second = runtime.model_for_group(base, physical_group=2)
    third = runtime.model_for_group(base, physical_group=3)
    assert second is third and second.name == "base+/loras/style@0.8"
    assert len(core.applied) == 1, "an unchanged set must reuse one patched variant"
    fourth = runtime.model_for_group(base, physical_group=4)
    assert fourth.name == "base+/loras/style@0.8+/loras/motion@1"
    assert core.applied[1:] == [
        ("base", "/loras/style", 0.8, "/loras/style"),
        ("base+/loras/style@0.8", "/loras/motion", 1.0, "/loras/motion"),
    ]
    assert core.loaded == ["/loras/style", "/loras/motion"], "each file is read once"
    assert runtime.model_for_group(base, physical_group=5) is base
    assert runtime.applied_report(1) == ""
    assert runtime.applied_report(4) == "lora=style @ 0.8 (5 keys); motion @ 1 (5 keys)"
    assert runtime.audit()[0] == {
        "physical_group": 2, "logical_chunks": [2],
        "loras": [{"name": "style", "strength_model": 0.8, "sha256": SHA_A}],
    }
    assert runtime.plan_report_lines()[1] == "- Chunk 2: style @ 0.8"
    runtime.release()
    assert runtime._variant is None and runtime._files == {}


def test_runtime_reports_unreadable_files_and_uncompiled_use():
    core = _FakeCore()
    with pytest.raises(LoraPlanError, match="cannot read 'missing'"):
        core.runtime(_plan(("missing", 1.0, "1"))).compile(chunks=2, terminal_merge_enabled=False)
    idle = core.runtime(_plan(("missing", 1.0, "9")))
    assert idle.compile(chunks=2, terminal_merge_enabled=False) == []
    assert idle.plan_report_lines() == [
        "LoRA Plan: connected, but no LoRA is assigned to a configured chunk."
    ]
    with pytest.raises(LoraPlanError, match="not compiled"):
        core.runtime(_plan(("style", 1.0, "1"))).model_for_group(object(), physical_group=1)


def test_file_identity_is_content_based_and_cached(tmp_path, monkeypatch):
    first, second = tmp_path / "a.safetensors", tmp_path / "renamed.safetensors"
    first.write_bytes(b"weights-one")
    second.write_bytes(b"weights-one")
    assert file_sha256(str(first)) == file_sha256(str(second))
    opened = []
    real_open = open
    monkeypatch.setattr("builtins.open", lambda *a, **k: opened.append(a[0]) or real_open(*a, **k))
    file_sha256(str(first))
    assert opened == [], "an unchanged file is not read again"
    first.write_bytes(b"weights-two-longer")
    assert file_sha256(str(first)) != file_sha256(str(second))


def test_scope_exposes_the_runtime_only_inside_one_execution():
    assert make_lora_plan_runtime(None) is None
    assert make_lora_plan_runtime(LoraPlan()) is None
    runtime = make_lora_plan_runtime(_plan(("style", 1.0, "all")))
    assert active_lora_plan_runtime() is None
    with lora_plan_scope(runtime) as scoped:
        assert scoped is runtime and active_lora_plan_runtime() is runtime
    assert active_lora_plan_runtime() is None
    with pytest.raises(RuntimeError):
        with lora_plan_scope(runtime):
            raise RuntimeError("sampling failed")
    assert active_lora_plan_runtime() is None


# --- public surface --------------------------------------------------------


def test_helper_node_is_public_and_chains():
    assert public_nodes.NODE_CLASS_MAPPINGS["H3ContinuumLoraPlan"] is H3ContinuumLoraPlan
    assert public_nodes.NODE_DISPLAY_NAME_MAPPINGS["H3ContinuumLoraPlan"] == (
        "H3 Continuum LoRA Plan (Experimental)"
    )
    schema = H3ContinuumLoraPlan.INPUT_TYPES()
    assert list(schema["required"]) == ["lora_name", "strength", "chunks"]
    assert schema["required"]["lora_name"][0][0] == "None"
    assert schema["required"]["chunks"][1]["default"] == "all"
    assert schema["optional"]["lora_plan"][0] == LORA_PLAN_TYPE == H3ContinuumLoraPlan.RETURN_TYPES[0]
    node = H3ContinuumLoraPlan()
    first, = node.pack(lora_name="style", strength=0.8, chunks="1-2")
    second, = node.pack(lora_name="motion", strength=1.0, chunks="3", lora_plan=first)
    assert [row.lora_name for row in second.rows] == ["style", "motion"]
    with pytest.raises(LoraPlanError):
        node.pack(lora_name="style", strength=1.0, chunks="1,,2")


def test_sampler_input_is_appended_last_and_only_on_v39():
    old = driving_nodes.H3ContinuumSamplerV38.INPUT_TYPES()
    new = driving_nodes.H3ContinuumSamplerV39.INPUT_TYPES()
    assert "lora_plan" not in old["optional"] and "lora_plan" not in old["required"]
    assert new["required"] == old["required"]
    assert list(new["optional"])[-1] == "lora_plan"
    assert new["optional"]["lora_plan"][0] == LORA_PLAN_TYPE
    without = [name for name in new["optional"] if name != "lora_plan"]
    assert without == list(new["optional"])[:-1]


def test_sampler_scopes_the_plan_and_does_not_forward_it(monkeypatch):
    seen = []
    plan = {"mode": "Custom", "accepted_chunks": 0, "connected_slots": [], "groups": []}

    def old_run(_self, **kwargs):
        seen.append((active_lora_plan_runtime(), kwargs))
        return (object(), object(), {"reference_routing_v1": plan}, "status", object())

    monkeypatch.setattr(driving_nodes.H3ContinuumSamplerV38, "run", old_run)
    sampler = driving_nodes.H3ContinuumSamplerV39()
    sampler.run(chunks=3)
    sampler.run(chunks=3, lora_plan=LoraPlan())
    lora_plan = _plan(("style", 1.0, "2"))
    sampler.run(chunks=3, lora_plan=lora_plan)
    assert seen[0][0] is None and seen[1][0] is None
    assert seen[2][0].plan is lora_plan
    assert all("lora_plan" not in kwargs for _, kwargs in seen)
    assert seen[0][1] == seen[2][1], "the runtime call itself is unchanged by a plan"
    assert active_lora_plan_runtime() is None
    with pytest.raises(LoraPlanError):
        sampler.run(chunks=3, lora_plan="style.safetensors")


# --- sequence integration --------------------------------------------------


def _run_three_chunks(monkeypatch, runtime):
    events = []
    width, height = 96, 64
    fake_model = SimpleNamespace(
        name="base", model=SimpleNamespace(diffusion_model=SimpleNamespace()),
        model_options={}, wrappers={}, model_dtype=lambda: torch.bfloat16, model_size=lambda: 0,
    )
    assets = IdentityAssets(None, None, None, None, "none")
    monkeypatch.setattr(sequence, "check_comfy_h3_runtime", lambda: [])

    def clone_model_for_chunk(model, *, strict, debug, chunk_index, context_frames):
        events.append(("clone", chunk_index, model.name))
        return SimpleNamespace(
            name=model.name, model=model.model, model_options={"transformer_options": {}},
            wrappers=model.wrappers, model_dtype=model.model_dtype, model_size=model.model_size,
        )

    def apply(model, lora, strength, metadata):
        return SimpleNamespace(
            name=f"{model.name}+{lora['file']}", model=model.model, model_options={},
            wrappers=model.wrappers, model_dtype=model.model_dtype, model_size=model.model_size,
        ), 3

    if runtime is not None:
        runtime._apply_lora = apply
    monkeypatch.setattr(sequence, "clone_model_for_chunk", clone_model_for_chunk)
    monkeypatch.setattr(sequence, "prepare_identity_assets", lambda *a, **k: assets)
    monkeypatch.setattr(sequence, "encode_identity_latents", lambda *a, **k: assets)
    monkeypatch.setattr(sequence, "empty_h3_latent", orchestration._latent)
    monkeypatch.setattr(sequence, "accelerator_summary", lambda model: "accelerators")

    def sample_chunk(**kwargs):
        events.append(("sample", kwargs["model"].name))
        return kwargs["latent"]

    monkeypatch.setattr(sequence, "sample_chunk", sample_chunk)
    monkeypatch.setattr(sequence, "decode_sequence", lambda **kwargs: (
        torch.zeros(362, 2, 2, 3),
        {"waveform": torch.zeros(1, 2, round(362 / 24 * 32000)), "sample_rate": 32000}, [],
    ))
    plan = make_prompt_plan(mode=PROMPT_MODE_FIXED, script="continuous shot",
                            chunks=3, chunk_seconds=5.0)
    with lora_plan_scope(runtime):
        result = sequence.run_sequence(
            model=fake_model, clip=orchestration.FakeClip([]), video_vae=object(),
            audio_vae=object(), sampler=object(), sigmas=torch.tensor([1.0, 0.0]),
            first_frame=None, last_frame=None, prompt_plan=plan, width=width, height=height,
            continuity=V2_CONTINUITY_OPTIONS[1], base_seed=42, audio_continuity=True,
            exact_total_duration=True, diagnostics_mode=DIAGNOSTICS_BASIC,
            reroll_from_chunk=0, reroll_nonce=0, strict_compatibility=True, debug=False,
        )
    return events, result[-1]


def test_sequence_samples_each_group_with_its_own_lora_set(monkeypatch):
    core = _FakeCore()
    runtime = core.runtime(_plan(("style", 0.8, "2")))
    events, report = _run_three_chunks(monkeypatch, runtime)
    assert [event[2] for event in events if event[0] == "clone"] == [
        "base", "base+/loras/style", "base",
    ]
    assert [event[1] for event in events if event[0] == "sample"] == [
        "base", "base+/loras/style", "base",
    ]
    assert core.loaded == ["/loras/style"]
    assert "LoRA Plan (Experimental, model-only):" in report
    assert "- Chunk 2: style @ 0.8" in report
    assert "chunk 2/3: lora=style @ 0.8 (3 keys)" in report
    assert runtime._variant is None, "the scope releases patched variants when the run ends"


def test_sequence_without_a_plan_is_unchanged(monkeypatch):
    events, report = _run_three_chunks(monkeypatch, None)
    assert [event[2] for event in events if event[0] == "clone"] == ["base"] * 3
    assert "LoRA" not in report and "lora=" not in report


def test_sequence_rejects_a_missing_lora_before_any_sampling(monkeypatch):
    runtime = _FakeCore().runtime(_plan(("missing", 1.0, "3")))
    with pytest.raises(Exception, match="cannot read 'missing'"):
        _run_three_chunks(monkeypatch, runtime)
    assert lora_runtime.active_lora_plan_runtime() is None


# --- Run Storage end to end (real prepare/commit/finalize) -----------------


def _review_queue(path, kwargs, *, new_position=None, previous=(), **lora):
    """One Review Each Chunk Queue against real Run Storage."""

    controller = diagnostics.controller_at(path)
    session = controller.prepare(**kwargs, **lora)
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


def test_real_run_storage_keeps_the_prefix_when_a_plan_is_attached(tmp_path):
    torch.manual_seed(7)
    kwargs = diagnostics.prepare_kwargs(signatures._Model())
    _, entries = _review_queue(tmp_path, kwargs, new_position=0)
    plain, entries = _review_queue(tmp_path, kwargs, new_position=1, previous=entries)
    assert "lora_group_contracts" not in plain.contract
    assert "lora_plan_audit" not in plain.manifest
    plain_files = {str(p): p.read_bytes() for p in plain.revision_root.rglob("*") if p.is_file()}

    records = _records((3, (3,), ((SHA_A, 0.8),)), chunk_count=3)
    audit = [{"physical_group": 3, "logical_chunks": [3],
              "loras": [{"name": "style.safetensors", "strength_model": 0.8, "sha256": SHA_A}]}]
    attached, reused = _review_queue(
        tmp_path, kwargs, lora_group_contracts=records, lora_plan_audit=audit,
    )
    assert len(reused) == 2 and attached.reused_count == 2
    assert attached.contract["lora_group_contracts"] == records
    assert attached.manifest["lora_plan_audit"] == audit
    assert attached.revision_id != plain.revision_id
    assert storage._hash(attached.contract["global"]) == storage._hash(plain.contract["global"])
    assert plain_files == {
        str(p): p.read_bytes() for p in plain.revision_root.rglob("*") if p.is_file()
    }, "the Take generated without a plan is not rewritten"


def test_real_run_storage_regenerates_from_the_chunk_that_gains_a_lora(tmp_path):
    torch.manual_seed(7)
    kwargs = diagnostics.prepare_kwargs(signatures._Model())
    _, entries = _review_queue(tmp_path, kwargs, new_position=0)
    _review_queue(tmp_path, kwargs, new_position=1, previous=entries)
    records = _records((2, (2,), ((SHA_A, 0.8),)), chunk_count=3)
    controller, reused = _review_queue(tmp_path, kwargs, lora_group_contracts=records)
    assert len(reused) == 1 and controller.reused_count == 1


def test_real_run_storage_resumes_a_run_generated_with_a_plan(tmp_path):
    torch.manual_seed(7)
    kwargs = diagnostics.prepare_kwargs(signatures._Model())
    records = _records((2, (2,), ((SHA_A, 0.8),)), chunk_count=3)
    lora = dict(lora_group_contracts=records)
    _, entries = _review_queue(tmp_path, kwargs, new_position=0, **lora)
    first, entries = _review_queue(tmp_path, kwargs, new_position=1, previous=entries, **lora)
    again, reused = _review_queue(tmp_path, kwargs, **lora)
    assert len(reused) == 2 and again.contract_sha256 == first.contract_sha256
    stronger = dict(lora_group_contracts=_records((2, (2,), ((SHA_A, 1.0),)), chunk_count=3))
    _, reused = _review_queue(tmp_path, kwargs, **stronger)
    assert len(reused) == 1
    _, reused = _review_queue(tmp_path, kwargs)
    assert len(reused) == 1, "removing the plan keeps only the chunk before the LoRA"
