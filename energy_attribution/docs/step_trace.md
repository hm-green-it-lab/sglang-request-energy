# Step trace: per-step, per-request records for energy attribution

The scheduler can write a trace of every forward pass ("step") together with the
work each request contributed to it. Combined with a power sampler (NVML for
GPU, later RAPL for CPU/DRAM), this lets us split measured energy across
requests that were batched together. See [PLAN.md](../PLAN.md) for the overall
approach (phases 1-4).

- Implementation: `python/sglang/srt/managers/scheduler_components/step_tracer.py`
- Test client and validator: [`tools/step_trace_client.py`](../tools/step_trace_client.py)

## Status

| Configuration | Status |
|---|---|
| `--disable-overlap-schedule`, prefill / decode / chunked prefill / mixed chunk | Supported; validated on RTX 4090 |
| Overlap scheduler (SGLang default) | Not wired up yet |
| Retraction | Recorded (`retraction_count`), not yet exercised in tests |
| Speculative decoding, DP attention, PP, PD disaggregation | Out of scope: steps are written with `owners_traced: false` and no owner rows |

## Enabling it

The tracer is off by default. When it is off, the hooks are no-ops.

| Env var | Default | Meaning |
|---|---|---|
| `SGLANG_ENABLE_STEP_TRACE` | `0` | Turn tracing on. |
| `SGLANG_STEP_TRACE_DIR` | `/tmp/sglang_step_trace` | Output directory. |
| `SGLANG_STEP_TRACE_FLUSH_STEPS` | `256` | Buffered records are written after this many steps, and whenever the scheduler goes idle. |
| `SGLANG_STEP_TRACE_REANCHOR_INTERVAL_S` | `10.0` | How often (at most) to re-pair the GPU clock with the host clock. |

For a complete Docker launch command, see
[Running a traced experiment](../README.md#running-a-traced-experiment-docker)
in the README. Pointing `SGLANG_STEP_TRACE_DIR` into the mounted repo makes the
trace visible outside the container; the default `/tmp` is container-local and
needs `docker cp`.

Each scheduler process writes one file:
`step_trace.<rank_tag>.<pid>.jsonl`, e.g. `step_trace.dp0_tp0_pp0.3048.jsonl`.
A new server launch writes a new file.

## Clock

All `*_ns` fields in the trace use the host's `CLOCK_MONOTONIC_RAW`, **not**
`time.monotonic_ns()` (`CLOCK_MONOTONIC`):

- Under WSL2, `CLOCK_MONOTONIC` is slewed by 2-4% relative to the GPU event
  timer, and the rate jumps between regimes. `CLOCK_MONOTONIC_RAW` tracked the
  GPU timer to about 0.1% (measured on an RTX 4090).
- Any data you want to join with the trace (power samples, client timestamps)
  must use the same clock: `time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)`.

GPU execution times come from CUDA events. They are converted to host time with
an **anchor**: drain the GPU, record an event, and bracket its completion with
two host clock reads. Half the bracket width is the anchor's uncertainty,
typically 50-110 µs. Anchors are re-taken only when the GPU is idle, so they add
no stall to running steps.

## Trace format

JSON Lines. Every line has a `kind` field: `meta`, `anchor`, `step` or `owner`.

```python
import pandas as pd

df = pd.read_json("step_trace.dp0_tp0_pp0.3048.jsonl", lines=True)
steps = df[df.kind == "step"]
owners = df[df.kind == "owner"]
per_step = owners.merge(steps, on="step_id", suffixes=("", "_step"))
```

### `meta` (first line, once per file)

| Field | Meaning |
|---|---|
| `schema_version` | Trace format version (currently `1`). |
| `clock` | Time base of all `*_ns` fields (`CLOCK_MONOTONIC_RAW`). |
| `pid`, `rank_tag` | Scheduler process that wrote the file. `dp0_tp0_pp0` = data-parallel rank 0, tensor-parallel rank 0, pipeline rank 0. |
| `gpu_id`, `device_name` | GPU this scheduler drives. |
| `device_uuid` | Same `GPU-...` format NVML reports (`nvmlDeviceGetUUID`), so a power sampler can pick the right device. |
| `enable_overlap` | Whether the overlap scheduler was on. |
| `time_source` | `cuda_event` = step times come from GPU events; `host` = no GPU available, host timestamps used instead. |

### `anchor` (at startup, then at most every `REANCHOR_INTERVAL_S` while idle)

| Field | Meaning |
|---|---|
| `anchor_id` | Sequential ID; each `step` names the anchor its times were converted with. |
| `host_ns` | Host time paired with the anchor's GPU event. |
| `uncertainty_ns` | ± error of that pairing. |

### `step` (one per forward pass)

A step is one execution of the model over a batch.

| Field | Meaning |
|---|---|
| `step_id` | Forward-pass counter; joins to `owner` rows. |
| `forward_mode` | `EXTEND` = prefill, `DECODE` = one token per request, `MIXED` = prefill and decode in the same pass. |
| `batch_size` | Number of requests in the step. |
| `t_start_ns`, `t_end_ns` | **When the GPU executed the step.** This is the window to use for energy attribution. |
| `host_launch_begin_ns`, `host_launch_end_ns` | When the CPU started and finished launching the step. Under the overlap scheduler this window differs from the GPU window and must not be used as GPU time. |
| `host_result_ns` | When the CPU finished processing the step's result. |
| `anchor_id` | Anchor used to convert `t_start_ns` / `t_end_ns`. |
| `num_prefill_reqs`, `num_decode_reqs` | Requests in the step that were prefilling vs decoding. |
| `prefill_tokens`, `decode_tokens` | New tokens computed in each phase. |
| `cached_tokens` | Prefix tokens the prefilling requests reused instead of recomputing. |
| `kv_read_tokens` | Earlier KV-cache tokens read by attention, summed over all requests (the "context" work, as opposed to new tokens). |
| `overlap_enabled` | Whether the overlap scheduler was on. |
| `owners_traced` | `false` = unsupported mode; the step has no `owner` rows. |

### `owner` (one per request per step)

| Field | Meaning |
|---|---|
| `step_id`, `rid` | Which step and which request. `rid` is SGLang's request ID; set it yourself (`"rid"` in `/generate`) to join with client-side data. |
| `phase` | `prefill` or `decode`. |
| `selected_tokens` | New tokens computed for this request in this step: the chunk size for prefill, always 1 for decode. |
| `context_len` | KV length after the step (earlier tokens + new ones). `context_len - selected_tokens` is what attention read. |
| `cached_tokens` | Prefix tokens reused in this step instead of recomputed: radix-cache hits, or earlier chunks of a chunked prefill. |
| `req_cached_tokens` | Running total of the request's radix-cache hits. |
| `is_chunked` | `true` if the prompt is too long for one step and this is not its last chunk. |
| `retraction_count` | How often the request was evicted for memory and recomputed. Non-zero means some prefill work was repeated. |

### Example

```json
{"kind":"step","step_id":31,"forward_mode":"MIXED","batch_size":2,
 "t_start_ns":7898211629569,"t_end_ns":7898234017264, ...
 "num_prefill_reqs":1,"num_decode_reqs":1,"prefill_tokens":787,"decode_tokens":1,
 "cached_tokens":0,"kv_read_tokens":909, ...}
```

Two requests ran in one pass for 22.4 ms of GPU time. One request prefilled 787
new tokens and the other decoded one token. Together, attention read 909 earlier
KV tokens. The step's two `owner` rows (`step_id: 31`) give the per-request
split, which is the input for attributing that step's energy.

### Things that look odd but are expected

- **The first step is usually a tiny prefill** (e.g. 6 tokens). That is the
  server's own warmup or health-check request; its `rid` is not one of yours.
- **`t_start_ns` can be slightly before `host_launch_begin_ns`** (tens of µs).
  That is within the anchor's uncertainty, not real reordering.

## Validating a trace

[`tools/step_trace_client.py`](../tools/step_trace_client.py) sends a concurrent workload with
fixed `rid`s and checks the trace against what the server reported.

```bash
# Send requests; writes one entry per request to the manifest
python energy_attribution/tools/step_trace_client.py send --url http://localhost:30000 --out manifest.json \
    --num-requests 160 --concurrency 48 --max-new-tokens 512 --arrival-spread-s 20

# Check the trace against the manifest
python energy_attribution/tools/step_trace_client.py check --manifest manifest.json \
    --trace /tmp/sglang_step_trace/step_trace.dp0_tp0_pp0.<pid>.jsonl
```

Use `--enable-mixed-chunk` and a small `--chunked-prefill-size` (e.g. 1024) on
the server to exercise mixed and chunked steps.

### `manifest.json` (written by `send`)

One entry per request, as reported by the server.

| Field | Meaning |
|---|---|
| `rid` | Request ID chosen by the client (`st-<random>-<index>`); joins to `owner.rid`. |
| `prompt_tokens` | Prompt length in tokens. |
| `completion_tokens` | Tokens generated. The client sets `ignore_eos`, so this equals `max_new_tokens`. |
| `cached_tokens` | Prompt tokens served from the radix cache. Even-numbered requests share a long prefix, so they get large hits. |
| `finish_reason` | Why generation stopped (`length` here). |
| `client_send_ns`, `client_recv_ns` | Client-side send/receive times on `CLOCK_MONOTONIC_RAW`. Comparable with the trace only if the client ran on the server host. |

### What `check` verifies

- Per request (skipped if it was retracted):
  - prefill tokens computed + prefix reused = `prompt_tokens`;
  - prefix reused = server's `cached_tokens`;
  - number of decode rows = `completion_tokens - 1` (the prefill step produces the first token);
  - the last prefill chunk is not marked `is_chunked`.
- Per step:
  - positive duration;
  - number of owner rows = `batch_size`;
  - owner tokens sum to the step totals.
- Without overlap:
  - steps do not overlap in time;
  - each GPU window lies inside its host window (launch begin to result), within the anchor uncertainty plus `--clock-tolerance-ns`.

It also prints coverage (mixed steps, chunked rows, prefix-reuse rows), so you
can see which code paths a run actually exercised.
