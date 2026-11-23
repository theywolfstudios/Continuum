# Roadmap — more than 16 chunks, and per-chunk LoRAs

Status: A is implemented in source with a limit of 64 (phases A0–A4; CPU and JavaScript checks only). A5 acceptance is open. B is a proposal.
Written against package 3.10.3 (commit `12bf61e`). Line numbers below refer to that commit.

## Summary

| | A. More than 16 chunks | B. Per-chunk LoRAs |
|---|---|---|
| Kind of change | Lift a cap; the runtime is already N-generic | New feature: helper node, runtime, storage identity, UI |
| Size | Small | Large |
| Sampling Contract | Unchanged | Additive per-group contracts; nothing added when unused |
| Public surface | Same 12 nodes, same widget order | A 13th node plus a new Sampler input or Sampler ID |
| Main risk | Decode/assembly memory and long-chain drift, not code | Reuse identity, weight re-patch cost, accelerator interop |

Do A first. It is independent, and B's per-chunk selector UI should be built against the new cap.

## How generation works today (what both features touch)

- The public Sampler is `H3ContinuumSamplerV39`. It inherits V38 → V37 → V36 → V35 → V34 (`v3/driving_nodes.py`) → `H3ContinuumSamplerProduction` → `H3ContinuumSamplerV3` (`v3/nodes.py`), which calls `H3ContinuumSamplerV2().run()` (`v2/nodes.py`) and finally `_run_runtime_internal` in `v2/sequence.py`.
- The loop runs once per **physical group**: one logical chunk, except that the last two chunks merge into one group when First and Last images are both used.
- Per group: build conditioning, `clone_model_for_chunk(model)` (`model_patch.py:329`), sample, copy the latent to CPU, `commit_group` to Run Storage. Call sites: `v2/sequence.py:1152` (normal) and `:1449` (terminal-merged).
- Run Storage reuse is decided by the Sampling Contract (`run_storage.py:1357`). `contract["global"]` holds the model signature, including LoRA patches and the upstream graph route. Every chunk contract embeds the hash of `global` (`run_storage.py:1229`). Reuse is a prefix match on per-chunk hashes, and importing a prefix into a new plan requires an identical `global` hash (`run_storage.py:2354`, `:2430`).

---

## A. More than 16 chunks

### Findings

The sequence loop, seed derivation, physical-group planner, prompt parser, Reference Routing compiler and Run Storage take the chunk count as a parameter. `chunk_count` is outside `contract["global"]`, so raising Chunks on an existing run already reuses the accepted prefix. The cap of 16 is enforced only in schema, validation and frontend code:

| Location | What it is | Action |
|---|---|---|
| `v3/nodes.py:789` | `chunks` widget max (Production schema, inherited by V3.8/V3.9) | Raise |
| `v3/nodes.py:39` | `REGENERATE_OPTIONS` combo, `Chunk 1`…`Chunk 16` | Raise |
| `v3/driving_nodes.py:632` | `take_group` max | Raise |
| `v3/driving_nodes.py:764` | `IS_CHANGED` validates Regenerate From with `chunks=16` | Raise (see trap 1) |
| `v2/prompts.py:170`, `:199` | Prompt plan rejects `chunks > 16` | Raise |
| `v3/reference_images_v39.py:17` | `MAX_CHUNKS` for per-chunk Reference selectors | Raise |
| `web/project_id.js:310` | Regenerate From list clamps to 16 | Raise |
| `web/reference_images_v39.js:3`, `:146`, `:206` | Per-chunk assignment table and its "1–16" labels | Raise |
| `v3/nodes.py:450`, `:345`; `v2/nodes.py:146`, `:206`, `:270`, `:473`, `:645`, `:735` | Legacy V2/V3 nodes, not publicly registered | Leave |
| `v3/easy_nodes.py:53` | `EASY_MAX_CHUNKS` | Leave (see trap 2) |

Tests that pin 16: `tests/test_v38_review_cache.py:121` (expects `Chunk 17` to be rejected), `tests/test_v39_rr_r6_public.py:183`, `tests/frontend_reference_images_v39.cjs:125-146`, `tests/test_v2_registration.py:32` (legacy, stays).
Docs that say 1–16: `README.md:394`, `CHANGELOG.md:162`, plus the JA mirrors.

### Traps

1. **`IS_CHANGED` silently caches.** `v3/driving_nodes.py:763` parses Regenerate From with `chunks=16` and swallows `ValueError`. If only the widget max is raised, `Regenerate From = Chunk 20` falls through to `return False`, Core treats the node as unchanged, and the regeneration does not run.
2. **`EASY_MAX_CHUNKS` is generation identity.** `resolve_easy_duration` computes `min(16, ceil(duration / 10))`. Raising it changes how any duration above 160 s is split, which changes existing Easy workflows. It must stay at 16.
3. **Reference selector spelling.** The V3.9 helper canonicalises `all` against `MAX_CHUNKS`. The effective selectors are filtered to the Sampler's chunk count and the storage descriptor records effective conditioning, so identity should not move. This needs a golden test, not an assumption.
4. **Memory grows with total frames, not chunk count.** Decoded float32 frames cost `chunks × seconds × 24 × width × height × 12` bytes: 16 × 10 s at 1344×768 is about 44 GiB, 32 chunks about 89 GiB. Finalize's Auto buffer has a 4 GiB RAM threshold before it goes disk-backed (`v3/file_backed_buffer.py`), and Decode Cache Helper keeps decoded video on disk (default quota 8 GiB, max 128). Core VAE Decode outputs without the helper stay in RAM. The legacy budget check (`v2/sequence.py:83`) does not run on the latent-first path.
5. **Prompt cache sized for 16.** The prompt conditioning cache keeps 16 entries per CLIP (`v2/h3_builder.py:22`) with no byte bound. With more than 16 distinct prompts, entries are evicted and re-encoded on later Queues. Left at 16: raising it retains more conditioning in RAM and should be measured first. The Reference encode cache (`v3/ref_encode_cache.py:20`) is keyed by image, not by chunk, and is unaffected.
6. **Long-chain quality.** Identity, colour and audio drift over 30+ continuation hops is unmeasured. `tools/issue13_drift_*` already exist for this.

### Phases

- **A0 — Decide the cap. Done:** `MAX_CHUNKS = 64` in `constants.py` (up to 32 minutes at 30 s chunks).
- **A1 — Backend. Done.** The "Raise" rows use the constant, including `IS_CHANGED`. Legacy nodes, `EASY_MAX_CHUNKS` and the prompt cache size are unchanged. `run_storage.py` is untouched, so existing contracts hash as before.
- **A2 — Frontend. Done in source, not checked in a browser.** `regenerateOptions` and the Reference Images V3.9 helper use 64. The assignment table already pages six chunks at a time. Review/Take controls and the Project Manager panel have not been exercised with long runs.
- **A3 — Tests. Done:** `tests/test_max_chunks.py` plus the updated pinned tests.
- **A4 — Memory guidance. Done:** one display-only Status line above 16 chunks (`long_run_decode_advisory` in `v3/reliability_v38.py`).
- **A5 — Acceptance. Open.** Browser save/reload; GPU run above 16 chunks at Draft resolution in Full Run and Review; long-chain drift measurement; whether to raise the prompt cache size. Then README (EN + JA) and a version bump with the release.

Requires: no contract version bump, no new node, no workflow file change.

---

## B. Per-chunk LoRAs

### Findings

- Continuum has no LoRA code. LoRAs are applied upstream by Core or rgthree loaders and reach the Sampler as patches on the one MODEL input. The documented chain is UNET → Sage → Sol → LoRA → Spectrum → Sampler.
- The same MODEL is cloned for every group, so there is nowhere to vary it today.
- LoRA patches are inside `contract["global"]["model"]` (`_model_signature`, `run_storage.py:953`). Changing a LoRA between Queues therefore changes every chunk hash and nothing is reused. Swapping the MODEL mid-run is not a workable route to this feature.
- V3.9 Reference Routing already solves "different conditioning per chunk with exact reuse". Its pipeline is the template: helper node with chunk selectors (`v3/reference_images_v39.py`) → private request (`v3/reference_runtime.py`) → schedule per physical group with terminal-pair conflict detection (`v3/reference_routing.py:202`) → per-group storage descriptors (`v3/reference_storage_contract.py`) → `reference_group_contracts`, a per-chunk group hash and the lineage hash (`run_storage.py:1231`, `:1289`, `:1528`) → Inspector output.
- Upstream documents name this as planned, unshipped work: "LoRA Plan / V4.0", "Chunk LoRA switching is a separate feature".

### Design

**Input.** A new helper node, "H3 Continuum LoRA Plan": rows of LoRA file, model strength and chunk selector, output type `H3_CONTINUUM_LORA_PLAN`, connected to one new optional Sampler input. LoRAs that apply to every chunk (Turbo included) stay upstream on the MODEL, exactly as now.

Alternative considered: several pre-patched MODEL inputs with selectors. It avoids loading LoRA files inside Continuum, but every variant needs its own Spectrum/accelerator chain and its own graph route in the contract. Not recommended.

**Runtime.**
- Compile the plan to one LoRA set per physical group. The terminal-merged pair must resolve to one set; otherwise report a conflict, as Reference Routing does.
- Load each file once, on CPU, before sampling starts, through the public Core calls that Core's own model-only LoRA loader uses. No private or global Core patching.
- Build one patched model variant per distinct LoRA set, then clone that variant per group at the two existing call sites. Building the patches again for every chunk would likely force a full weight re-patch at every chunk even when the set has not changed. This depends on how Core tracks patch identity across clones and must be confirmed against the supported Core versions.
- Reused groups are never cloned or patched.
- Cost: a weight re-patch at each boundary where the set changes. No extra VRAM is held; the deltas stay on CPU.

**Identity.**
- Plan not connected: the contract is byte-identical to today. Golden test.
- Plan connected: add `lora_group_contracts` (per physical group: ordered list of file content SHA-256 and strength), a `physical_group_lora_sha256` field in the chunk contract of each group that has LoRAs, and the group hashes in the lineage hash.
- Keep all of it out of `contract["global"]`, and add nothing for groups with an empty set. Then attaching a plan to an existing run, or changing chunk 6's LoRA, keeps chunks 1–5 through the existing prefix-import path and regenerates from chunk 6.
- Identify files by content hash, cached by path, size and mtime, not by filename.
- Takes made with a different LoRA set are separate revisions; old Takes stay readable and are not reused.

### Constraints and open risks

- **Schedule is global.** `sigmas` and `sampler` are in `global`. A Turbo LoRA on some chunks only would need per-chunk schedules; out of scope for the first version.
- **Model-only.** H3 LoRAs in the shipped guidance are model-only; no CLIP strength in the first version.
- **Accelerators.** Patches added inside the Sampler sit after Spectrum in clone order. Weight deltas do not depend on wrapper order, but Spectrum, Sage, Sol and KJ Preview Override each need a run with the plan active, and the reuse observers (`spectrum_signature.py`, `preview_override_signature.py`) must still report a resumable contract.
- **Quantised or GGUF model loaders** may apply LoRA patches through their own path. Unverified.
- **Second Pass** takes its own MODEL and its refine context carries no LoRA plan. First version: Second Pass ignores the plan and says so in its status. Carrying group descriptors into the refine context is a follow-up.
- **Public surface.** AGENTS.md fixes the registration at twelve nodes and the V3.9 schema. This feature adds a 13th node and a Sampler input, which is a deliberate contract change to record in AGENTS.md, the surface tests and the docs.

### Phases

- **B0 — Decisions and contract note.** Settle the open decisions below. Write the LoRA group contract (fields, hashing, empty-set rule, terminal-pair rule) as a short contract document before code, in the style of `docs/N2B_BRANCH_PROVENANCE_CONTRACT_V1.md`.
- **B1 — Pure planning core.** Plan bundle type, selector parsing (reuse `parse_chunk_selector`), per-group schedule, conflict detection, descriptor and hash functions. CPU tests only.
- **B2 — Storage identity.** Thread the plan into `build_sampling_contract`, `_apply_nonce_contract` and `_nonce_lineage_hash`. Tests: golden no-plan contract; attach a plan mid-run and keep the prefix; change one group's set and regenerate from there; Review/Take selection across LoRA revisions; original v5 history still readable.
- **B3 — Runtime.** File loading and hashing, variant models, application at both call sites, status report lines. Ship as an unregistered experimental Sampler subclass plus helper, default off, following `H3ContinuumSamplerV38MemoryPolicyExperimental` (`v3/driving_nodes.py:935`).
- **B4 — GPU gate.** Record per AGENTS.md: prompts, model, LoRAs, sampler settings, dimensions, chunk timings. Cover: A/B against the same LoRA applied upstream to all chunks (should match within sampler determinism); re-patch time at boundaries; VRAM flat across chunks; Spectrum on and off; Sage/Sol; KJ Preview Override; resume after restart.
- **B5 — Frontend.** Helper node UI (LoRA picker rows, strength, per-chunk table shared with Reference Images V3.9), plan summary in the Sampler's Inspector, Project Manager display of a Take's LoRA set.
- **B6 — Promotion.** Register the helper and the Sampler change, update the public-surface tests, AGENTS.md, README/USER_GUIDE (EN + JA), CHANGELOG, WORKLOG, PROJECT_STATE, manifests. Add a new example workflow; do not edit the shipped ones.
- **B7 — Follow-ups.** Second Pass awareness, per-chunk strength curves, per-chunk schedules for Turbo.

---

## Decisions needed

1. **Chunk cap.** 32, 64 (recommended) or another value.
2. **How LoRAs are supplied.** LoRA Plan helper that takes files (recommended) or multiple MODEL inputs.
3. **Sampler surface.** A new Sampler ID that leaves V3.9 untouched, as V3.9 did to V3.8 (recommended, and matches upstream's "V4.0" wording), or an optional input appended to V3.9.
4. **Mid-run attach.** Whether adding or changing a LoRA at chunk N must keep chunks before N (recommended; it drives the identity design above).
5. **Upstream.** Whether this stays in the fork or is meant to go back to `ukr8b3g-cmyk/ComfyUI-H3-Continuum`. It affects versioning and how strictly the EN/JA docs and release files must be kept in step.

## What the work requires

- A Python environment with torch and safetensors for the CPU suite, and Node.js for the frontend suites. `tools/validate.ps1` is the entry point.
- A ComfyUI install with a GPU, the H3 model, at least two compatible non-Turbo LoRAs, and the accelerator nodes (Spectrum, Sage, Sol, KJNodes) for B4 and A5.
- A browser session against that install for save/reload acceptance.
- A verified snapshot before changes (`tools/snapshot.ps1`), and manifest regeneration after any payload change.
