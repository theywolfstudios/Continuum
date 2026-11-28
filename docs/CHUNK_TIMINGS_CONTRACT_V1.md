# Chunk Timings contract v1 (Experimental)

Status: implemented in source and checked on CPU and in the JavaScript suites. Not yet run against ComfyUI Core, in a browser or on a GPU.

Chunk Timings give selected chunks of the V3.9 Sampler their own length. This document fixes how lengths become generation identity in Run Storage. Sampling Contract v6 and Graph Contract v4 are unchanged.

## Input

`H3 Continuum Chunk Timings` has one text field and produces an ordered list of rows, each a chunk selection and a length in seconds. Two forms are accepted, not mixed:

- A list, `5, 8, 6.5`: chunk 1 gets the first value, chunk 2 the second, and so on.
- Lines, `2: 8` or `4-6 = 10`: each sets only the chunks it names. Selections use the Reference Images grammar (`all`, or numbers and inclusive ranges such as `1,3-5`, within 1–64). Where two lines name the same chunk, the later line wins.

Lines and list items may also be separated by `;`. A value may end in `s`, `sec` or `seconds`. Each value must lie within the Sampler's Chunk Seconds range, 4–30 s. Invalid text stops the helper node with a message naming the line or chunk.

## Resolution

The Sampler's `Chunk Seconds` stays the length of every chunk the rows do not name. Rows naming chunks above the Sampler's `Chunks` are kept but inactive until the run is that long. A value within 1e-6 s of Chunk Seconds is no override.

If no configured chunk differs from Chunk Seconds, the schedule resolves to nothing, and every part of the runtime takes its uniform path. This is the case for a disconnected input, empty text, and text that only repeats Chunk Seconds.

## Runtime

- Each chunk aims at its cumulative target: the frames of all chunks up to and including it, `round(sum(lengths) × 24)`. As with uniform chunks, a chunk lands within one step of the H3 17k+5 frame grid, and the next chunk absorbs the difference.
- Without an override up to a chunk, its target uses the existing `count × Chunk Seconds × 24` expression, so those chunks sample exactly as before.
- The assembly plan's `target_frames` is the sum of the assembled chunks' lengths. Finalize trims or pads to it, as it already does for uniform runs. The plan carries `chunk_durations` only when a length differs.
- A chunk's State plan records its own length as `requested_extend_seconds`.
- Timeline prompt sections (`[5-13s]`) are mapped onto the custom chunk boundaries.
- Driving Audio uses the custom total length.
- With First and Last Image, the merged final pair is defined for two 5-second chunks. If either of the last two chunks has another length, each chunk is sampled on its own, as for any Chunk Seconds other than 5. That changes the common contract, so a saved First + Last run restarts from chunk 1 when that switch happens.
- The legacy Timeline Video (VIDEO) input is refused with an override. Timeline Video Frames with Follow Timeline selects frames by the actual output range and works with any lengths.

## Storage records

- `contract["global"]["chunk_seconds"]` stays the Sampler's Chunk Seconds.
- `contract["chunk_duration_overrides"]` lists, in chunk order, one record per chunk whose length differs:

  ```json
  [{"chunk_number": 3, "duration_seconds": 8.0}]
  ```

- That chunk's contract gains `duration_seconds`.
- The lineage hash gains `chunk_duration_overrides`.
- A plain Session records the full list as `settings["chunk_durations"]`.

## What does not change

- **No override:** none of the keys above are written. Sampling Contract, Session and assembly plan are byte-identical to a run without the feature.
- **Chunks at Chunk Seconds:** their chunk contract hash is the same as without Chunk Timings.

## Reuse

Reuse stays a prefix match on per-chunk contract hashes, and adopting a prefix from another revision still requires an identical `global` hash. A chunk's length moves every later chunk's start, but the prefix rule already regenerates every chunk after the first changed one. Consequently:

| Change | Chunks kept |
|---|---|
| Give chunk N a different length | 1 … N−1 |
| Change or remove the length of chunk N | 1 … N−1 |
| Queue again with the same lengths | the whole saved prefix |
| Change Chunk Seconds | none (common contract, as before) |

A plain Session input is reused up to the first chunk whose length differs from the Session's `settings["chunk_durations"]` (or its Chunk Seconds when absent).

Takes produced with other lengths are separate revisions. They remain readable and selectable; a selected Take is imported only up to the first chunk whose contract differs from the current lengths.

Stored manifests are validated on read: records must be in order, within the run's chunks, not equal to Chunk Seconds, and must agree with the per-chunk fields.

An older package version does not write these keys. It recomputes chunk contracts without them, so it will not reuse a chunk generated with another length.

## Outside v1

- Second Pass, the V3.8 Sampler, Easy nodes and unregistered legacy Samplers.
- A table editor in the frontend. The Sampler's Total Length reads "From connected inputs" while the helper is connected.
