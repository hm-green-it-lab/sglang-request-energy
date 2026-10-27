# Handoff: per-request energy attribution in SGLang (Seam-style step trace)

Context for a new Claude Code session. Written 2026-10-16 from a Cowork session.

## Goal
Patch this SGLang fork (branch `resource-tracking`, base v0.5.21) so it emits a
scheduler-step trace that lets us attribute measured GPU (NVML) and later CPU/DRAM
(RAPL) energy to individual concurrently batched requests, following
Roy 2026 "Seam" (vLLM) and our draft paper "Runtime Energy Attribution of
Inference Requests in LLM Serving Systems" (§III-C Instrumentation Points in SGLang).

## Environment
- Edit on Mac (this repo), run on the "tower": Windows + WSL2 Ubuntu 24.04, RTX 4090,
  driver 591.86 (CUDA <= 13.1). SSH alias `tower` (port 2222, Tailscale name `tower`).
- Fork is cloned on the tower at `~/sglang`.
- Docker dev container `sglang-dev` (image `lmsysorg/sglang:v0.5.21`) mounts `~/sglang`
  at `/workspace/sglang`, fork installed with
  `SGLANG_BUILD_RUST_EXTS=none pip install -e python --no-deps`.
- Start server: `tmux new -d -s sglang "docker exec -it sglang-dev python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 30000; exec bash"`
- Alternative venv on tower: `~/venvs/sglang`, `~/start-sglang.sh [model]`.
- WSL2 has **no RAPL**; GPU energy via NVML only on the tower. RAPL work needs native Linux (A100 node).

## Review of draft §III-C against the code
- Correct: event loop selects one `ScheduleBatch` per step (`Scheduler.event_loop_normal` /
  `event_loop_overlap` in python/sglang/srt/managers/scheduler.py: `get_next_batch_to_run` ->
  `run_batch` -> `process_batch_result`); forward modes extend/decode/mixed.
- Correct: `MetricsReporter._build_scheduled_request_metrics`
  (managers/scheduler_components/metrics_reporter.py) loops over per-request data and keeps
  only sums/variances -> template for our per-owner records.
- Partly: `DeviceTimer` (utils/device_timer.py) records CUDA events but reports only elapsed
  durations, lazily, and is off by default (`SGLANG_ENABLE_METRICS_DEVICE_TIMER`). Needs
  anchoring CUDA events to `time.monotonic_ns()` for NVML alignment.
- Missing in draft:
  1. Overlap scheduler (default on): batch N+1 is launched while GPU runs N; N's result is
     processed one iteration later. Host timestamps around `run_batch`
     (`batch.launch_ts`) measure launch, not GPU execution. Validate first with
     `--disable-overlap-schedule`, then use anchored CUDA events.
  2. Radix prefix reuse: `len(req.prefix_indices)` = cached tokens, `extend_input_len` = built.
  3. Chunked prefill (`req.is_chunked`) and mixed batches (`batch.decoding_reqs`).
  4. Retraction = recompute (`ScheduleBatch.retract_decode`, `req.retraction_count`).
  5. COW: verified N/A for attention KV (see "Coverage of Seam's activity vector").
  6. Out of scope first: spec decoding, DP attention, PP, PD disaggregation.
  7. Join key: `req.rid`.

## Record contract (Seam Table 6)
- Step record (per forward pass): step_id, t_start_ns, t_end_ns (GPU-anchored monotonic),
  forward_mode, batch_size, totals (prefill/decode/cached/kv-read tokens), overlap_enabled.
- Owner record (per request per step): step_id, rid, phase, selected_tokens
  (extend_input_len or 1), context_len (seqlen), cached_tokens, is_chunked, retraction_count.
- Buffered JSONL sidecar written by the scheduler process (not via OTel/meta_info).

## Coverage of Seam's activity vector
Seam's Eq. 1 splits dynamic energy by Prefill + Decode + KV-read + Prefix + COW + Recompute
(Table 6: KV-read, prefix, COW and recompute are optional fields). State of our trace:

| Seam term | Our trace |
|---|---|
| Prefill tokens (incl. chunks) | Yes: `selected_tokens` with `phase: prefill`, `is_chunked` |
| Decode tokens | Yes: `selected_tokens` with `phase: decode` |
| KV-cache read | Yes: owner `context_len - selected_tokens`, step `kv_read_tokens` |
| Shared prefix, hit | Yes: owner `cached_tokens`, `req_cached_tokens` |
| Shared prefix, build role + prefix id | **Missing** |
| Recompute | Partly: `retraction_count` only, no recomputed-token count |
| COW | N/A for attention KV (below); only Mamba state in hybrid models |
| Callsite / application span | Client side, joined on `rid` (same as Seam) |

Follow-ups, in priority order:
1. **Prefix build role and prefix id (main trace gap).** Needed for Seam's default
   builder-pays prefix policy (prefix-build activity charged to the request that built it,
   warm hits charged nothing) and its amortized variant (build cost spread over all requests
   sharing the prefix id). At prefill, SGLang knows the matched radix node (`req.last_node`,
   a `TreeNode` with a stable `id`, mem_cache/radix_cache.py). Record that id per prefill
   owner row as the prefix group. Builder role can be derived offline: the first request
   whose computed tokens later appear as another request's cache hit. A fully online role
   would need a hook where SGLang inserts nodes into the radix tree.
2. **Recompute tokens.** Derivable offline: after a retraction, a request's prefill tokens up
   to the context length it had already reached are recomputation. Optional explicit field
   `recompute_tokens` on owner rows. Retraction has not been exercised in tests yet.
3. **Mamba COW**, only if hybrid SSM models are traced: count copies from
   `mamba_cow_src_index` / `batch.mamba_cow_src_indices`. Out of scope for now.

Why COW is N/A: vLLM shares partially filled KV blocks, so diverging requests trigger block
copies. SGLang's radix cache matches only whole pages (`RadixKey.page_aligned`; default page
size 1 has no partial pages), so diverging requests write new KV and nothing is copied.

Context from Seam Table 3: the full vector did not clearly beat token counts on magnitude;
KV-read drove the improvement on decode-heavy and phase-mix cases, prefix/COW mainly affect
ranking and policy. So the term that mattered most is already covered; the prefix role
decides who pays for shared prefixes.

## Phases
1. StepTracer component `python/sglang/srt/managers/scheduler_components/step_tracer.py`.
   Do not grow `Scheduler.__init__`; read `.claude/skills/large-class-style` and
   `.claude/skills/env-var-conventions` first. Hooks: (A) after `get_next_batch_to_run`:
   snapshot owners + demands (host data only, no GPU sync) and record CUDA start event;
   (B) after forward launch: CUDA end event; (C) in `process_batch_result`: convert events
   to monotonic ns via anchor, emit records. Start with `--disable-overlap-schedule`.
   Plus a small test client to send concurrent requests and check the JSONL.
2. NVML sampler sidecar (pynvml: power + total energy, 10-100 ms, monotonic timestamps,
   idle-floor calibration).
3. Offline attribution: overlap windows, floor by residency, dynamic by activity vector
   (Seam Eq. 1-4), conservation assert < 1e-6 J, LB/UB bounds (Eq. 5), coverage manifest.
4. Validation: overhead on/off (bench_serving), isolated vs co-scheduled, small-group Shapley.

## Optional improvements
- **Attribute speculative-decoding steps.** Currently such steps are traced with their GPU
  window but `owners_traced: false` and no owner rows, so their energy is unattributed
  (must show up in the coverage manifest, not be dropped). But the draft and verify work is
  caused by the requests in the batch, including rejected guesses. The whole draft + verify
  cycle runs inside one `run_batch`, so the step window already covers it; only the
  per-request split is missing.
  - Cheap step: emit owner rows with `rid` and phase for spec batches (`batch.reqs` is
    known at launch), token fields marked unknown, so phase 3 can at least split over the
    right requests.
  - Full version: per-request proposed/verified draft tokens (fixed by
    `--speculative-num-draft-tokens`, draft steps, top-k; known at launch) and accepted
    tokens (known only after the step, from the result).
  - Attribution principle: charge requests for what they caused (proposed + verified,
    including rejected tokens); report the accepted share separately as an efficiency
    metric. Same principle as the extra discarded decode step under overlap scheduling.
- **Attribute beam-search member rows (currently silently wrong).** With beam search,
  `batch.reqs` holds only each group's leader; the other beams are extra rows appended to
  the GPU tensors (`append_beam_tail`, python/sglang/srt/beam_search/batch_tail.py).
  `_snapshot_decode_owners` slices `seq_lens_cpu[:num_reqs]`, so those rows are dropped
  and the leader's request is recorded as 1 token although the GPU processed
  1 + (beam width - 1) rows. Unlike other out-of-scope modes this is not flagged.
  - Interim: set `owners_traced: false` when `batch.beam_tail` is present, so the energy is
    reported as unattributed instead of mis-split.
  - Fix: fold each leader's member rows (from `batch.beam_tail` entries) into its owner row
    (`selected_tokens` and context/KV-read summed over the rows). Not tested; beam search was
    not used so far.

## Next step
Implement phase 1.
