"""Core's sampling allocations are returned between physical samples.

CPU only: Core is replaced by recording stand-ins.  These tests fix the call
the package makes and its order; they say nothing about GPU memory.
"""

from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from ComfyUI_H3_Continuum_Join import core_allocations
from ComfyUI_H3_Continuum_Join.v2 import sampling as chunk_sampling
from ComfyUI_H3_Continuum_Join.v3 import refine_sampling, targeted_refine_sampling
from ComfyUI_H3_Continuum_Join.v3.targeted_refine_sampling import (
    sample_video_audio_refine_chunk,
)


class _Nested:
    is_nested = True

    def __init__(self, tensors):
        self.tensors = list(tensors)

    def unbind(self):
        return self.tensors

    def to(self, *_args, **_kwargs):
        return self


def _av_latent():
    video = torch.zeros((1, 24, 3, 4, 4))
    audio = torch.zeros((1, 32, 2, 12))
    return {"samples": _Nested((video, audio))}


@pytest.fixture(autouse=True)
def _fresh_module_state(monkeypatch):
    monkeypatch.setattr(core_allocations, "_announced", False)
    monkeypatch.setattr(core_allocations, "_warned", False)
    monkeypatch.delenv(core_allocations.KEEP_ENV, raising=False)


def _install_core(
    monkeypatch,
    *,
    aimdo_enabled=True,
    with_prefetch=True,
    with_cleanup=True,
    with_synchronize=True,
    cleanup_error=None,
):
    """Install the three Core modules the helper reads; return the call log."""

    calls = []
    comfy = ModuleType("comfy")
    comfy.__path__ = []
    memory_management = ModuleType("comfy.memory_management")
    memory_management.aimdo_enabled = aimdo_enabled
    model_management = ModuleType("comfy.model_management")
    if with_synchronize:
        model_management.synchronize = lambda: calls.append("synchronize")
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.memory_management", memory_management)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    comfy.memory_management = memory_management
    comfy.model_management = model_management
    if with_prefetch:
        model_prefetch = ModuleType("comfy.model_prefetch")
        if with_cleanup:

            def cleanup_prefetch_queues():
                calls.append("cleanup_prefetch_queues")
                if cleanup_error is not None:
                    raise cleanup_error

            model_prefetch.cleanup_prefetch_queues = cleanup_prefetch_queues
        monkeypatch.setitem(sys.modules, "comfy.model_prefetch", model_prefetch)
        comfy.model_prefetch = model_prefetch
    else:
        # A Core older than its allocation graph has no such module.
        monkeypatch.setitem(sys.modules, "comfy.model_prefetch", None)
    return calls


def test_release_runs_cores_node_boundary_cleanup_after_a_synchronize(monkeypatch):
    calls = _install_core(monkeypatch)

    assert core_allocations.release_sampling_allocations() is True
    assert calls == ["synchronize", "cleanup_prefetch_queues"]


def test_release_runs_once_per_call_so_every_physical_sample_gets_its_own(monkeypatch):
    calls = _install_core(monkeypatch)

    for _ in range(3):
        assert core_allocations.release_sampling_allocations() is True
    assert calls.count("cleanup_prefetch_queues") == 3


def test_release_follows_cores_executor_and_does_nothing_without_its_allocator(monkeypatch):
    calls = _install_core(monkeypatch, aimdo_enabled=False)

    assert core_allocations.release_sampling_allocations() is False
    assert calls == []


@pytest.mark.parametrize(
    "core",
    [
        {"with_prefetch": False},
        {"with_cleanup": False},
    ],
)
def test_release_is_a_no_op_on_a_core_without_the_cleanup(monkeypatch, core):
    calls = _install_core(monkeypatch, **core)

    assert core_allocations.release_sampling_allocations() is False
    assert calls == []


def test_release_is_a_no_op_without_core_installed(monkeypatch):
    for name in ("comfy", "comfy.memory_management", "comfy.model_management", "comfy.model_prefetch"):
        monkeypatch.setitem(sys.modules, name, None)

    assert core_allocations.release_sampling_allocations() is False


def test_release_still_cleans_up_on_a_core_without_synchronize(monkeypatch):
    calls = _install_core(monkeypatch, with_synchronize=False)

    assert core_allocations.release_sampling_allocations() is True
    assert calls == ["cleanup_prefetch_queues"]


def test_a_failing_cleanup_never_fails_the_sample_and_warns_once(monkeypatch, caplog):
    calls = _install_core(monkeypatch, cleanup_error=RuntimeError("unpin failed"))

    with caplog.at_level(logging.WARNING, logger="h3_continuum_join"):
        assert core_allocations.release_sampling_allocations() is False
        assert core_allocations.release_sampling_allocations() is False

    assert calls.count("cleanup_prefetch_queues") == 2
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "RuntimeError: unpin failed" in warnings[0].getMessage()


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_the_opt_out_keeps_cores_allocations_for_the_whole_node(monkeypatch, value):
    calls = _install_core(monkeypatch)
    monkeypatch.setenv(core_allocations.KEEP_ENV, value)

    assert core_allocations.release_sampling_allocations() is False
    assert calls == []


@pytest.mark.parametrize("value", ["", "0", "false", "off"])
def test_other_opt_out_values_leave_the_release_on(monkeypatch, value):
    calls = _install_core(monkeypatch)
    monkeypatch.setenv(core_allocations.KEEP_ENV, value)

    assert core_allocations.release_sampling_allocations() is True
    assert calls == ["synchronize", "cleanup_prefetch_queues"]


def test_the_first_release_is_announced_once_with_the_opt_out_name(monkeypatch, caplog):
    _install_core(monkeypatch)

    with caplog.at_level(logging.INFO, logger="h3_continuum_join"):
        core_allocations.release_sampling_allocations()
        core_allocations.release_sampling_allocations()

    messages = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    assert len(messages) == 1
    assert core_allocations.KEEP_ENV in messages[0]


class _OrderedGuider:
    def __init__(self, order, result=None, error=None):
        self.order = order
        self.result = result
        self.error = error

    def sample(self, noise, latent_image, sampler, sigmas, **kwargs):
        self.order.append("sample")
        if self.error is not None:
            raise self.error
        return latent_image if self.result is None else self.result


def _fake_refine_runtime():
    return (
        SimpleNamespace(intermediate_device=lambda: torch.device("cpu")),
        SimpleNamespace(NestedTensor=_Nested),
        SimpleNamespace(
            fix_empty_latent_channels=lambda _model, samples, _spatial, _temporal: samples,
            prepare_noise=lambda samples, _seed, _batch: torch.zeros_like(samples, device="cpu"),
        ),
        SimpleNamespace(PROGRESS_BAR_ENABLED=False),
        SimpleNamespace(prepare_callback=lambda *_args: None),
    )


def _install_chunk_sampling_core(monkeypatch):
    """The Core modules ``sample_chunk`` imports, as recording stand-ins."""

    comfy = ModuleType("comfy")
    comfy.__path__ = []
    model_management = ModuleType("comfy.model_management")
    model_management.intermediate_device = lambda: torch.device("cpu")
    sample = ModuleType("comfy.sample")
    sample.fix_empty_latent_channels = lambda _model, samples, _spatial, _temporal: samples
    sample.prepare_noise = lambda samples, _seed, _batch: samples
    utils = ModuleType("comfy.utils")
    utils.PROGRESS_BAR_ENABLED = False
    latent_preview = ModuleType("latent_preview")
    latent_preview.prepare_callback = lambda *_args: None
    comfy.model_management = model_management
    comfy.sample = sample
    comfy.utils = utils
    for name, module in (
        ("comfy", comfy),
        ("comfy.model_management", model_management),
        ("comfy.sample", sample),
        ("comfy.utils", utils),
        ("latent_preview", latent_preview),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def _sample_arguments():
    return dict(
        model=object(),
        conditioning=["conditioning"],
        latent=_av_latent(),
        sampler=object(),
        sigmas=torch.tensor([0.35, 0.2, 0.0]),
        seed=7,
        enable_preview=False,
    )


def test_a_generated_chunk_releases_after_its_sample(monkeypatch):
    order = []
    _install_chunk_sampling_core(monkeypatch)
    monkeypatch.setattr(chunk_sampling, "_make_basic_guider", lambda *_args: _OrderedGuider(order))
    monkeypatch.setattr(chunk_sampling, "release_sampling_allocations", lambda: order.append("release"))

    output = chunk_sampling.sample_chunk(**_sample_arguments())

    assert order == ["sample", "release"]
    assert hasattr(output["samples"], "unbind")


def test_a_chunk_that_fails_to_sample_leaves_the_cleanup_to_core(monkeypatch):
    order = []
    _install_chunk_sampling_core(monkeypatch)
    failure = RuntimeError("Allocation on device 0 would exceed allowed memory.")
    monkeypatch.setattr(
        chunk_sampling,
        "_make_basic_guider",
        lambda *_args: _OrderedGuider(order, error=failure),
    )
    monkeypatch.setattr(chunk_sampling, "release_sampling_allocations", lambda: order.append("release"))

    with pytest.raises(RuntimeError) as raised:
        chunk_sampling.sample_chunk(**_sample_arguments())

    assert raised.value is failure
    assert order == ["sample"]


def test_a_second_pass_chunk_releases_after_its_sample(monkeypatch):
    order = []
    monkeypatch.setattr(refine_sampling, "_load_runtime_modules", _fake_refine_runtime)
    monkeypatch.setattr(refine_sampling, "_make_basic_guider", lambda *_args: _OrderedGuider(order))
    monkeypatch.setattr(refine_sampling, "release_sampling_allocations", lambda: order.append("release"))

    refine_sampling.sample_refine_chunk(**_sample_arguments())

    assert order == ["sample", "release"]


def test_a_targeted_second_pass_chunk_releases_after_its_sample(monkeypatch):
    order = []
    monkeypatch.setattr(targeted_refine_sampling, "_load_runtime_modules", _fake_refine_runtime)
    monkeypatch.setattr(
        targeted_refine_sampling,
        "_make_basic_guider",
        lambda *_args: _OrderedGuider(order),
    )
    monkeypatch.setattr(
        targeted_refine_sampling,
        "release_sampling_allocations",
        lambda: order.append("release"),
    )

    sample_video_audio_refine_chunk(**_sample_arguments())

    assert order == ["sample", "release"]


def test_the_pack_has_no_other_sampling_call_site():
    """A new ``guider.sample`` must also end its sample the way these do."""

    from pathlib import Path

    root = Path(core_allocations.__file__).resolve().parent
    sites = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if relative.startswith(("tests/", "tools/", ".")):
            continue
        source = path.read_text(encoding="utf-8")
        if "guider.sample(" in source:
            assert "release_sampling_allocations()" in source, relative
            sites.append(relative)
    assert sites == [
        "v2/sampling.py",
        "v3/refine_sampling.py",
        "v3/targeted_refine_sampling.py",
    ]
