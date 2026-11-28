"""Public helper node that sets per-chunk durations for the V3.9 Sampler."""

from __future__ import annotations

from .chunk_timings import CHUNK_TIMINGS_TYPE, parse_chunk_timings


class H3ContinuumChunkTimings:
    """Different lengths for selected chunks; others keep Chunk Seconds."""

    CATEGORY = "MiniMax H3/Continuum/Input"
    DESCRIPTION = (
        "Experimental. Gives selected chunks of the H3 Continuum Sampler V3.9 "
        "their own length in seconds. Chunks you do not list keep the "
        "Sampler's Chunk Seconds. Empty text changes nothing."
    )
    SEARCH_ALIASES = [
        "H3 Continuum Chunk Timings",
        "MiniMax H3 per-chunk duration",
        "chunk seconds per chunk",
        "chunk length",
    ]
    RETURN_TYPES = (CHUNK_TIMINGS_TYPE,)
    RETURN_NAMES = ("chunk_timings",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timings": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "display_name": "Chunk Lengths",
                    "placeholder": "5, 8, 6.5\n\nor one line per change:\n2: 8\n4-6: 10",
                    "tooltip": (
                        "A list such as 5, 8, 6.5 sets chunk 1, 2, 3 in order. Lines such "
                        "as 2: 8 or 4-6: 10 set only the chunks they name; a later line "
                        "wins. Each value is 4-30 seconds. Unlisted chunks, and chunks "
                        "above the Sampler's Chunks, keep Chunk Seconds."
                    ),
                }),
            },
        }

    def build(self, timings=""):
        return (parse_chunk_timings(timings),)


NODE_CLASS_MAPPINGS = {"H3ContinuumChunkTimings": H3ContinuumChunkTimings}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumChunkTimings": "H3 Continuum Chunk Timings (Experimental)",
}
