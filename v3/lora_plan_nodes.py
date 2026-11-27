"""Public helper node that builds a LoRA Plan for the V3.9 Sampler."""

from __future__ import annotations

from .lora_plan import LORA_NONE, LORA_PLAN_TYPE, append_lora_plan_row


def _lora_choices() -> list[str]:
    try:
        import folder_paths
    except ModuleNotFoundError as error:
        if error.name != "folder_paths":
            raise
        return [LORA_NONE]  # Standalone import without ComfyUI Core.
    return [LORA_NONE, *folder_paths.get_filename_list("loras")]


class H3ContinuumLoraPlan:
    """One LoRA with a chunk selection; chain nodes to add more."""

    CATEGORY = "MiniMax H3/Continuum/Input"
    DESCRIPTION = (
        "Experimental. Applies one model-only LoRA to selected chunks of the "
        "H3 Continuum Sampler V3.9. Chain several LoRA Plan nodes through "
        "Previous LoRA Plan to add more LoRAs. LoRAs meant for every chunk, "
        "including Turbo LoRAs, can stay on the MODEL input."
    )
    SEARCH_ALIASES = [
        "H3 Continuum LoRA Plan",
        "MiniMax H3 per-chunk LoRA",
        "chunk lora",
    ]
    RETURN_TYPES = (LORA_PLAN_TYPE,)
    RETURN_NAMES = ("lora_plan",)
    FUNCTION = "pack"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (_lora_choices(), {
                    "default": LORA_NONE,
                    "display_name": "LoRA",
                    "tooltip": "LoRA file from models/loras. None passes the plan through unchanged.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "display_name": "Strength",
                    "tooltip": "Model strength, as in Core's model-only LoRA loader. 0 disables this LoRA.",
                }),
                "chunks": ("STRING", {
                    "default": "all",
                    "display_name": "Chunks",
                    "tooltip": (
                        "Chunks that use this LoRA: all, off, or numbers and ranges such "
                        "as 1,3-5. Numbers above the Sampler's Chunks are ignored until "
                        "the run is that long."
                    ),
                }),
            },
            "optional": {
                "lora_plan": (LORA_PLAN_TYPE, {
                    "display_name": "Previous LoRA Plan (Optional)",
                    "tooltip": "Output of another LoRA Plan node. This node's LoRA is added after it.",
                }),
            },
        }

    def pack(self, lora_name=LORA_NONE, strength=1.0, chunks="all", lora_plan=None):
        return (append_lora_plan_row(
            lora_plan, lora_name=lora_name, strength=strength, chunks=chunks,
        ),)


NODE_CLASS_MAPPINGS = {"H3ContinuumLoraPlan": H3ContinuumLoraPlan}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumLoraPlan": "H3 Continuum LoRA Plan (Experimental)",
}
