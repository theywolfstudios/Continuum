"""Return ComfyUI Core's sampling allocations between physical samples.

Core records the GPU allocations of a model forward in a per-thread
allocation graph (``comfy.model_prefetch``, the "Comfy model compiler") and
releases it when the executing node returns.  A Core sampler node is one
sampling run.  A Continuum Sampler or Second Pass runs every physical group
inside one node, and groups differ in token count (carried context, per-chunk
references, a shorter last chunk, the terminal pair), so the graph is recorded
again group after group while the node still holds what earlier groups left.
The largest single request of a later group can then be refused while the
device still reports free memory.

Core's own multi-phase loops run the node-boundary cleanup between phases for
the same reason.  ``release_sampling_allocations`` does that after one
completed physical sample: under the condition Core's executor uses, through
Core's public function, and nothing else.  It reads and changes no sampling
input, tensor, contract or stored value.
"""

from __future__ import annotations

import logging
import os

LOG = logging.getLogger("h3_continuum_join")

# Set to 1/true/yes/on to keep Core's allocations for the whole node, as
# before this module existed.  Read on every call, so it can be compared
# without editing a workflow.
KEEP_ENV = "H3_CONTINUUM_KEEP_CORE_ALLOCATIONS"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

_announced = False
_warned = False


def keep_requested() -> bool:
    return os.environ.get(KEEP_ENV, "").strip().lower() in _TRUE_VALUES


def release_sampling_allocations() -> bool:
    """Run Core's node-boundary allocation cleanup after one physical sample.

    Returns True when Core's cleanup ran.  Returns False without raising when
    the opt-out is set, when this Core has no such cleanup, when its allocator
    is not enabled, or when the cleanup fails.  Core runs the same cleanup
    again when the node returns, so a skipped call only restores the earlier
    behavior.
    """

    global _announced, _warned
    if keep_requested():
        return False
    try:
        import comfy.memory_management as memory_management
        import comfy.model_management as model_management
        import comfy.model_prefetch as model_prefetch
    except Exception:
        return False
    cleanup = getattr(model_prefetch, "cleanup_prefetch_queues", None)
    if not callable(cleanup):
        return False
    if not getattr(memory_management, "aimdo_enabled", False):
        return False
    try:
        # The last forward may still be using the graph's pages.
        synchronize = getattr(model_management, "synchronize", None)
        if callable(synchronize):
            synchronize()
        cleanup()
    except Exception as exc:
        if not _warned:
            _warned = True
            LOG.warning(
                "H3 Continuum could not release Core's sampling allocations "
                "between physical samples (%s: %s). Sampling continues; Core "
                "releases them when the node returns.",
                type(exc).__name__,
                exc,
            )
        return False
    if not _announced:
        _announced = True
        LOG.info(
            "H3 Continuum releases Core's sampling allocations between "
            "physical samples. Set %s=1 to keep them for the whole node.",
            KEEP_ENV,
        )
    return True
