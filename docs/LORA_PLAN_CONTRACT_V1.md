# LoRA Plan contract v1 (Experimental)

Status: implemented in source and checked on CPU. Not yet run against ComfyUI Core, in a browser or on a GPU.

A LoRA Plan applies model-only LoRAs to selected chunks of the V3.9 Sampler. This document fixes how a plan becomes generation identity in Run Storage. Sampling Contract v6 and Graph Contract v4 are unchanged.

## Input

`H3 Continuum LoRA Plan` produces an ordered list of rows: LoRA name, model strength, chunk selection. Nodes chain through `Previous LoRA Plan`; a later node's row comes after the earlier ones.

- Chunk selection uses the Reference Images selector grammar: `all`, `off`, or numbers and inclusive ranges such as `1,3-5`, within 1–64.
- Numbers above the Sampler's `Chunks` are kept but inactive until the run is that long.
- A row with LoRA `None` or strength `0` is inactive, as in Core's LoRA loader.
- LoRAs for every chunk, including Turbo LoRAs, can stay on the MODEL input. Those are part of the common model identity, as before.

## Schedule

The plan is resolved to one ordered LoRA set per **physical group**. A physical group is one chunk, except that the last two chunks form one group when First and Last Image are both connected. Both chunks of that pair must select the same LoRAs; otherwise the Sampler stops before sampling with a message naming the two chunks.

## Storage records

Only groups that receive at least one LoRA produce a record:

```json
{
  "descriptor": {
    "version": 1,
    "physical_group": 3,
    "logical_chunks": [3],
    "loras": [
      {"sha256": "<SHA-256 of the LoRA file content>", "strength_model": 0.8}
    ]
  },
  "sha256": "<SHA-256 of the canonical JSON of descriptor>"
}
```

- `loras` is in application order.
- A LoRA is identified by its file content, not its name or path. Renaming a file does not change identity; replacing its content does.
- Records are stored in ascending group order as `contract["lora_group_contracts"]`.
- Each chunk contract of a recorded group gains `physical_group_lora_sha256`.
- The lineage hash gains `lora_group_hashes`, the list of record hashes.
- `contract["global"]` gains nothing.

## What does not change

- **No plan, an empty plan, or a plan whose rows select no configured chunk:** none of the keys above are written. The contract is byte-identical to a run without the feature.
- **Chunks without a LoRA:** their chunk contract hash is the same as without a plan.

## Reuse

Reuse stays a prefix match on per-chunk contract hashes, and adopting a prefix from another revision still requires an identical `global` hash. Consequently:

| Change | Chunks kept |
|---|---|
| Attach a plan whose first LoRA is on chunk N | 1 … N−1 |
| Change a LoRA file, strength, order or selection first affecting chunk N | 1 … N−1 |
| Remove a plan whose first LoRA was on chunk N | 1 … N−1 |
| Queue again with the same plan | the whole saved prefix |

Takes produced under a different LoRA set are separate revisions. They remain readable and selectable; a selected Take is imported only up to the first group whose contract differs from the current plan.

Stored manifests are validated on read: the records must match the run's physical groups, every record hash must match its descriptor, and the per-chunk fields must agree with the records.

`manifest["lora_plan_audit"]` records display names per group. It is not identity.

An older package version does not write these keys. It recomputes chunk contracts without them, so it will not reuse chunks generated with a LoRA.

## Runtime

- Files are resolved and hashed before sampling starts. A missing or unreadable file stops the Sampler before any chunk is generated.
- A file is loaded when the first group that uses it is sampled. Reused groups load nothing.
- Loading and patching use the calls Core's model-only LoRA loader uses (`comfy.utils.load_torch_file`, `comfy.sd.load_lora_for_models` with no CLIP).
- One patched MODEL is built per distinct LoRA set and cloned for each group, so consecutive groups with the same set do not trigger a weight re-patch. A change of set does.

## Outside v1

- CLIP strength.
- Per-chunk sampler or sigma schedules. A Turbo LoRA on only some chunks would need one.
- Second Pass. It uses its own MODEL input and does not read the plan.
- The V3.8 Sampler and unregistered legacy Samplers.
