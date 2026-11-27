"""Runtime for a LoRA Plan: file identity, per-group MODEL variants, scope.

LoRAs are loaded and applied with the public ComfyUI Core calls used by
Core's model-only LoRA loader. One patched MODEL is built per distinct LoRA
set and then cloned for each physical group, so consecutive groups with the
same set share one weight-patch identity and Core does not re-patch between
them. Patch tensors stay on the device Core loads them to (CPU); nothing here
moves the base model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import inspect
import os
import re
from typing import Any

from .lora_plan import (
    LoraPlan,
    LoraPlanError,
    LoraSchedule,
    LoraUse,
    compile_lora_schedule,
    lora_group_record,
    validate_lora_plan,
)


_FILE_HASHES: dict[tuple[str, int, int], str] = {}
_ACTIVE: ContextVar["LoraPlanRuntime | None"] = ContextVar(
    "h3_continuum_lora_plan_runtime", default=None,
)


def _display_name(lora_name: str) -> str:
    """File name without its models/loras subfolder, for reports only."""

    return re.split(r"[\\/]", str(lora_name))[-1]


def _resolve_lora_path(lora_name: str) -> str:
    import folder_paths

    resolver = getattr(folder_paths, "get_full_path_or_raise", None)
    if resolver is not None:
        return str(resolver("loras", lora_name))
    path = folder_paths.get_full_path("loras", lora_name)
    if path is None:
        raise FileNotFoundError(f"LoRA file not found: {lora_name}")
    return str(path)


def file_sha256(path: str) -> str:
    """Content SHA-256, cached by resolved path, size and modification time."""

    resolved = os.path.realpath(path)
    stat = os.stat(resolved)
    key = (resolved, int(stat.st_size), int(stat.st_mtime_ns))
    cached = _FILE_HASHES.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    for stale in [item for item in _FILE_HASHES if item[0] == resolved]:
        del _FILE_HASHES[stale]
    _FILE_HASHES[key] = digest.hexdigest()
    return _FILE_HASHES[key]


def _load_lora_file(path: str) -> tuple[Any, Any]:
    import comfy.utils

    loader = comfy.utils.load_torch_file
    if "return_metadata" in inspect.signature(loader).parameters:
        return loader(path, safe_load=True, return_metadata=True)
    return loader(path, safe_load=True), None


def _apply_lora(model: Any, lora: Any, strength: float, metadata: Any) -> tuple[Any, int]:
    """Model-only application; returns the patched clone and its key count."""

    import comfy.sd

    apply = comfy.sd.load_lora_for_models
    if metadata is not None and "lora_metadata" in inspect.signature(apply).parameters:
        patched, _ = apply(model, None, lora, float(strength), 0, lora_metadata=metadata)
    else:
        patched, _ = apply(model, None, lora, float(strength), 0)
    before = getattr(model, "patches", None) or {}
    after = getattr(patched, "patches", None) or {}
    changed = sum(1 for key in after if len(after[key]) != len(before.get(key, ())))
    return patched, changed


class LoraPlanRuntime:
    """One Sampler execution's LoRA Plan state."""

    def __init__(
        self,
        plan: LoraPlan,
        *,
        resolve_path: Callable[[str], str] = _resolve_lora_path,
        hash_file: Callable[[str], str] = file_sha256,
        load_file: Callable[[str], tuple[Any, Any]] = _load_lora_file,
        apply_lora: Callable[[Any, Any, float, Any], tuple[Any, int]] = _apply_lora,
    ) -> None:
        self.plan = validate_lora_plan(plan)
        self._resolve_path = resolve_path
        self._hash_file = hash_file
        self._load_file = load_file
        self._apply_lora = apply_lora
        self.schedule: LoraSchedule | None = None
        self._paths: dict[str, str] = {}
        self._hashes: dict[str, str] = {}
        self._files: dict[str, tuple[Any, Any]] = {}
        self._variant: tuple[Any, tuple[tuple[str, float], ...], Any] | None = None
        self._applied: dict[int, list[str]] = {}
        self._last_applied: list[str] = []

    def compile(self, *, chunks: int, terminal_merge_enabled: bool) -> list[dict[str, Any]]:
        """Fix the schedule for this run and return its storage records.

        Files are resolved and hashed here, before any sampling. They are
        loaded only when a group that uses them is actually sampled.
        """

        self.schedule = compile_lora_schedule(
            self.plan, total_chunks=int(chunks),
            terminal_merge_enabled=bool(terminal_merge_enabled),
        )
        records = []
        for group in self.schedule.active_groups:
            identities = []
            for use in group.loras:
                if use.lora_name not in self._hashes:
                    try:
                        path = self._resolve_path(use.lora_name)
                        self._hashes[use.lora_name] = self._hash_file(path)
                    except (OSError, ValueError) as exc:
                        raise LoraPlanError(
                            f"LoRA Plan cannot read {use.lora_name!r}: {exc}"
                        ) from exc
                    self._paths[use.lora_name] = path
                identities.append((self._hashes[use.lora_name], use.strength))
            records.append(lora_group_record(
                physical_group=group.physical_group,
                logical_chunks=group.logical_chunks,
                loras=tuple(identities),
            ))
        return records

    def _group_loras(self, physical_group: int) -> tuple[LoraUse, ...]:
        if self.schedule is None:
            raise LoraPlanError("LoRA Plan was not compiled before sampling")
        for group in self.schedule.groups:
            if group.physical_group == int(physical_group):
                return group.loras
        raise LoraPlanError(f"LoRA Plan has no physical group {physical_group}")

    def model_for_group(self, model: Any, *, physical_group: int) -> Any:
        """Return ``model`` itself, or its patched variant for this group."""

        loras = self._group_loras(physical_group)
        if not loras:
            return model
        key = tuple((self._hashes[use.lora_name], use.strength) for use in loras)
        if self._variant is not None and self._variant[0] is model and self._variant[1] == key:
            variant = self._variant[2]
        else:
            # Keep one variant: a new set replaces the previous one so its
            # patch tensors are not retained for the rest of the run.
            self._variant = None
            variant = model
            applied = []
            for use in loras:
                if use.lora_name not in self._files:
                    self._files[use.lora_name] = self._load_file(self._paths[use.lora_name])
                lora, metadata = self._files[use.lora_name]
                variant, patched_keys = self._apply_lora(variant, lora, use.strength, metadata)
                applied.append(f"{use.lora_name} @ {use.strength:g} ({patched_keys} keys)")
            self._variant = (model, key, variant)
            self._last_applied = applied
        self._applied[int(physical_group)] = list(self._last_applied)
        return variant

    def audit(self) -> list[dict[str, Any]]:
        """Display names per active group for the manifest; not identity."""

        if self.schedule is None:
            return []
        return [
            {
                "physical_group": group.physical_group,
                "logical_chunks": list(group.logical_chunks),
                "loras": [
                    {"name": _display_name(use.lora_name), "strength_model": use.strength,
                     "sha256": self._hashes.get(use.lora_name, "")}
                    for use in group.loras
                ],
            }
            for group in self.schedule.active_groups
        ]

    def plan_report_lines(self) -> list[str]:
        if self.schedule is None:
            return []
        active = self.schedule.active_groups
        if not active:
            return ["LoRA Plan: connected, but no LoRA is assigned to a configured chunk."]
        lines = ["LoRA Plan (Experimental, model-only):"]
        for group in active:
            first, last = group.logical_chunks[0], group.logical_chunks[-1]
            label = f"Chunk {first}" if first == last else f"Chunks {first}-{last}"
            names = ", ".join(
                f"{_display_name(use.lora_name)} @ {use.strength:g}" for use in group.loras
            )
            lines.append(f"- {label}: {names}")
        return lines

    def applied_report(self, physical_group: int) -> str:
        applied = self._applied.get(int(physical_group))
        return "" if not applied else "lora=" + "; ".join(applied)

    def release(self) -> None:
        self._variant = None
        self._files.clear()


def make_lora_plan_runtime(plan: Any) -> LoraPlanRuntime | None:
    """Return a runtime, or ``None`` when the input cannot affect any chunk."""

    if plan is None:
        return None
    validated = validate_lora_plan(plan)
    if not validated.rows:
        return None
    return LoraPlanRuntime(validated)


def active_lora_plan_runtime() -> LoraPlanRuntime | None:
    return _ACTIVE.get()


@contextmanager
def lora_plan_scope(runtime: LoraPlanRuntime | None) -> Iterator[LoraPlanRuntime | None]:
    """Expose ``runtime`` to the sequence runtime for one Sampler execution."""

    token = _ACTIVE.set(runtime)
    try:
        yield runtime
    finally:
        _ACTIVE.reset(token)
        if runtime is not None:
            runtime.release()
