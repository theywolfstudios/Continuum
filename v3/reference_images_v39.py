"""Fixed-slot Reference Images input for the V3.9 sampler.

Slots 1-9 are the original nine. Slots 10 and above are extended slots: they
add nothing to a run that leaves them unconnected.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..constants import (  # Shared Sampler Chunks and Reference limits.
    MAX_CHUNKS,
    MAX_REFERENCE_SLOTS,
    MAX_REFERENCES_PER_CHUNK,
)
from ..reference import ReferenceImageBundle
from .reference_routing import REFERENCE_SLOT_IDS, TAGGED_SELECTOR, parse_chunk_selector


REFERENCE_IMAGES_V39_TYPE = "H3_CONTINUUM_REFERENCE_IMAGES_V39"
REFERENCE_USE_ALL = "All chunks"
REFERENCE_USE_PER_CHUNK = "Per chunk"
REFERENCE_USE_TAGGED = "Tagged chunks"
REFERENCE_USE_OPTIONS = (REFERENCE_USE_ALL, REFERENCE_USE_PER_CHUNK, REFERENCE_USE_TAGGED)
_BASE_SLOTS = len(REFERENCE_SLOT_IDS)
_SLOT_NUMBERS = range(1, MAX_REFERENCE_SLOTS + 1)


@dataclass(frozen=True, slots=True)
class ReferenceImagesV39Bundle:
    images: tuple[torch.Tensor | None, ...]
    reference_use: str
    selectors: tuple[str, ...]


def _saved_selection(value, *, index: int) -> str:
    """An extended slot restored from an older workflow may hold a blank value."""
    text = str(value)
    return "off" if index > _BASE_SLOTS and not text.strip() else text


def _canonical_selector(value: str) -> str:
    parsed = parse_chunk_selector(value, total_chunks=MAX_CHUNKS)
    if not parsed.valid:
        details = "; ".join(issue.message for issue in parsed.issues)
        raise ValueError(f"invalid V3.9 Reference chunk selection: {details}")
    return ",".join(map(str, parsed.logical_chunks)) or "off"


def effective_reference_inputs(bundle: ReferenceImagesV39Bundle | None, *, chunks: int):
    """Return legacy-shaped images and private selectors without changing saved data."""
    if not isinstance(chunks, int) or isinstance(chunks, bool) or not 1 <= chunks <= MAX_CHUNKS:
        raise ValueError(f"V3.9 Reference Chunks must be within 1..{MAX_CHUNKS}")
    if bundle is None:
        images = (None,) * _BASE_SLOTS
        mode = REFERENCE_USE_ALL
        selections = ("off",) * _BASE_SLOTS
    elif (not isinstance(bundle, ReferenceImagesV39Bundle)
          or not _BASE_SLOTS <= len(bundle.images) <= MAX_REFERENCE_SLOTS
          or len(bundle.selectors) != len(bundle.images)
          or bundle.reference_use not in REFERENCE_USE_OPTIONS):
        raise ValueError("V3.9 Reference Images requires its nine-slot V3.9 bundle")
    else:
        images = bundle.images
        mode = bundle.reference_use
        selections = bundle.selectors
    selectors = {}
    for index, (image, selection) in enumerate(zip(images, selections, strict=True), start=1):
        if image is None:
            # An unconnected extended slot is left out of the private request,
            # so a run that does not use one is the same as before they existed.
            if index <= _BASE_SLOTS:
                selectors[f"R{index}"] = "off"
        elif mode == REFERENCE_USE_ALL:
            selectors[f"R{index}"] = "all"
        elif mode == REFERENCE_USE_TAGGED:
            # The Sampler owns the prompts; it resolves this to the chunks
            # whose prompt names the slot's @R tag.
            selectors[f"R{index}"] = TAGGED_SELECTOR
        else:
            parsed = parse_chunk_selector(
                _saved_selection(selection, index=index), total_chunks=MAX_CHUNKS,
            )
            if not parsed.valid:
                details = "; ".join(issue.message for issue in parsed.issues)
                raise ValueError(f"R{index} chunk selection: {details}")
            active = (n for n in parsed.logical_chunks if n <= chunks)
            selectors[f"R{index}"] = ",".join(map(str, active)) or "off"
    if tuple(selectors)[:_BASE_SLOTS] != REFERENCE_SLOT_IDS:
        raise ValueError("V3.9 Reference slot ordering changed")
    extended_connected = any(image is not None for image in images[_BASE_SLOTS:])
    carried = images[3:] if extended_connected else images[3:_BASE_SLOTS]
    return images[:3], ReferenceImageBundle(carried), selectors


class H3ContinuumReferenceImagesV39:
    CATEGORY = "MiniMax H3/Continuum/Input"
    DESCRIPTION = (
        f"Connect up to {MAX_REFERENCE_SLOTS} fixed Reference Images; choose chunks in this "
        f"node. One chunk can use at most {MAX_REFERENCES_PER_CHUNK}."
    )
    RETURN_TYPES = (REFERENCE_IMAGES_V39_TYPE,)
    RETURN_NAMES = ("reference_images",)
    FUNCTION = "pack"

    @classmethod
    def INPUT_TYPES(cls):
        def chunks_widget(index):
            return ("STRING", {
                "default": "off", "advanced": True,
                "display_name": f"Image {index} chunks",
                "tooltip": f"Saved assignments for Reference Image {index}; use the in-node table.",
            })

        def image_socket(index):
            return ("IMAGE", {
                "display_name": f"Reference Image {index} (Optional)",
                "tooltip": f"Fixed Image {index} / @R{index}. Its number never changes with other connections.",
            })

        extended = range(_BASE_SLOTS + 1, MAX_REFERENCE_SLOTS + 1)
        return {
            "required": {
                "reference_use": (REFERENCE_USE_OPTIONS, {
                    "default": REFERENCE_USE_ALL,
                    "display_name": "Reference Use",
                    "tooltip": (
                        "All chunks uses connected images throughout. Per chunk uses the "
                        "assignments below. Tagged chunks sends each image only to chunks "
                        "whose prompt contains its @R tag. One chunk can use at most "
                        f"{MAX_REFERENCES_PER_CHUNK} images."
                    ),
                }),
                **{
                    f"reference_r{index}_chunks": chunks_widget(index)
                    for index in range(1, _BASE_SLOTS + 1)
                },
            },
            # Extended slots are appended and optional, so saved workflows and
            # API prompts written for nine slots keep loading and validating.
            "optional": {
                **{f"reference_image_{index}": image_socket(index) for index in _SLOT_NUMBERS},
                **{f"reference_r{index}_chunks": chunks_widget(index) for index in extended},
            },
        }

    def pack(self, reference_use=REFERENCE_USE_ALL, **kwargs):
        if reference_use not in REFERENCE_USE_OPTIONS:
            raise ValueError("Reference Use must be All chunks, Per chunk or Tagged chunks")
        images = tuple(kwargs.get(f"reference_image_{index}") for index in _SLOT_NUMBERS)
        saved = tuple(
            _saved_selection(kwargs.get(f"reference_r{index}_chunks", "off"), index=index)
            for index in _SLOT_NUMBERS
        )
        selectors = tuple(
            (_canonical_selector(saved[index - 1])
             if reference_use == REFERENCE_USE_PER_CHUNK and images[index - 1] is not None else
             saved[index - 1])
            for index in _SLOT_NUMBERS
        )
        return (ReferenceImagesV39Bundle(images, reference_use, selectors),)
