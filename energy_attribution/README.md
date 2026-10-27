# Energy attribution for SGLang

Research work on this fork (branch `resource-tracking`): attribute measured GPU
energy (NVML), and later CPU/DRAM energy (RAPL), to individual requests that the
SGLang scheduler batched together. The approach follows Roy 2026 ("Seam") and
our draft paper "Runtime Energy Attribution of Inference Requests in LLM Serving
Systems".

| Phase | Status |
|---|---|
| 1. Step trace (scheduler steps + per-request demand) | Done without overlap scheduler; overlap pending |
| 2. NVML power sampler | Not started |
| 3. Offline attribution | Not started |
| 4. Validation (overhead, isolated vs co-scheduled) | Not started |

Details and the optional improvements are in [PLAN.md](PLAN.md).

## Layout

| Path | Contents |
|---|---|
| [`PLAN.md`](PLAN.md) | Phases, design notes, optional improvements |
| [`docs/step_trace.md`](docs/step_trace.md) | Trace format reference: env vars, record fields, clock, validation |
| [`tools/step_trace_client.py`](tools/step_trace_client.py) | Sends a concurrent workload and checks a trace against it |
| [`tools/plot_step_trace.py`](tools/plot_step_trace.py) | Plots a trace (and its manifest) for visual sanity checks; needs pandas + matplotlib |
| `runs/<date>_<name>/` | One folder per experiment (git-ignored): `trace.jsonl`, `manifest.json`, `server_cmd.txt` |
| `papers/` | Reference PDFs (git-ignored) |

The tracer itself lives in the SGLang package:
`python/sglang/srt/managers/scheduler_components/step_tracer.py`, wired into
`python/sglang/srt/managers/scheduler.py`.

## Running a traced experiment (Docker)

Requirements on the server machine: an NVIDIA GPU, Docker with the NVIDIA
Container Toolkit (`docker run --gpus all` works), and git.

**1. Clone the fork.**

```bash
git clone -b resource-tracking https://github.com/BZoennchen/sglang.git
cd sglang
```

**2. Start a traced server.** This runs the official SGLang image, mounts the
fork over it, installs the fork in editable mode and starts the server. Traces
land in `energy_attribution/runs/$RUN` on the host.

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct
RUN=$(date +%F)_my-run
mkdir -p energy_attribution/runs/$RUN

docker run --rm -it --init --name sglang-trace \
    --gpus all --shm-size 32g --ipc=host -p 30000:30000 \
    -v "$PWD":/workspace/sglang -w /workspace/sglang \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN \
    -e SGLANG_ENABLE_STEP_TRACE=1 \
    -e SGLANG_STEP_TRACE_DIR=/workspace/sglang/energy_attribution/runs/$RUN \
    -e SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python \
    lmsysorg/sglang:v0.5.21 \
    bash -c "SGLANG_BUILD_RUST_EXTS=none pip install -q -e python --no-deps && \
             python -m sglang.launch_server --model-path $MODEL \
                 --host 0.0.0.0 --port 30000 --disable-overlap-schedule"
```

- `--disable-overlap-schedule` is required for now: with the overlap scheduler
  (SGLang's default) the trace contains only its header lines.
- To exercise chunked and mixed steps, append
  `--chunked-prefill-size 1024 --enable-mixed-chunk`.
- Supported: one GPU, or plain tensor parallelism (`--tp N`, one trace file per
  rank). Not supported: speculative decoding, DP attention, pipeline
  parallelism, PD disaggregation.
- The image tag matches the fork's base release (v0.5.21), so `--no-deps`
  reuses the image's dependencies.
- `SGLANG_BUILD_RUST_EXTS=none` skips building the Rust extensions (the image
  has no Rust toolchain); `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python` then
  selects the pure-Python radix tree, without which startup fails.
- `-e HF_TOKEN` passes your token through for gated models; drop it otherwise.
- Stop the server with Ctrl-C, or `docker stop sglang-trace` from another shell.

**3. Send the workload and check the trace.** The client needs only Python 3
(standard library) on Linux, macOS or WSL; it does not run on native Windows.
It can run on the server machine or on any machine that reaches port 30000.

```bash
# Wait until the server answers
until curl -sf http://<server>:30000/health_generate >/dev/null; do sleep 5; done

python3 energy_attribution/tools/step_trace_client.py send --url http://<server>:30000 \
    --out energy_attribution/runs/$RUN/manifest.json \
    --num-requests 160 --concurrency 48 --max-new-tokens 512 --arrival-spread-s 20

python3 energy_attribution/tools/step_trace_client.py check \
    --manifest energy_attribution/runs/$RUN/manifest.json \
    --trace energy_attribution/runs/$RUN/step_trace.*.jsonl
```

If the client runs on another machine, fetch the trace first, e.g.
`rsync -av <server>:sglang/energy_attribution/runs/$RUN/ energy_attribution/runs/$RUN/`.
The tracer writes buffered records when the server goes idle, so wait a few
seconds after `send` finishes. With tensor parallelism, check each rank's file
separately.

Optionally plot it: `python3 energy_attribution/tools/plot_step_trace.py
--trace "energy_attribution/runs/$RUN/step_trace.*.jsonl" --manifest
energy_attribution/runs/$RUN/manifest.json` writes four PNGs into the run folder.

**4. Record the configuration.** Write the exact server and client commands
into `runs/$RUN/server_cmd.txt`, so every trace can be traced back to the setup
that produced it.

## Notes

- **Clock.** All trace timestamps use `CLOCK_MONOTONIC_RAW`, not
  `time.monotonic_ns()`. Under WSL2 the latter was slewed 2-4% against the GPU
  timer; on native Linux the difference is negligible. Any power sampler joined
  with the trace must use the raw clock too. Client timestamps in the manifest
  are only comparable with the trace when the client runs on the server machine.
- **File ownership.** The container runs as root, so trace files on the host are
  owned by root. Use `sudo chown -R $USER energy_attribution/runs` if needed.
- **Prompt sizes.** The client's prompts reach about 2,000-2,500 tokens plus
  `--max-new-tokens`; lower `--max-new-tokens` for models with a small context
  window.
- **Check failures that are not tracer bugs:** overlap scheduler on (no steps),
  speculative decoding (no owner rows), beam search (member rows dropped),
  aborted requests (token counts differ). Retracted requests are skipped. On
  very different hardware, `--clock-tolerance-ns` (default 0.2 ms) may need
  loosening; the printed anchor uncertainty is its floor.

## Papers

- Roy, 2026. "Scheduler-step GPU energy attribution for continuously batched LLM
  serving" (Seam).
- Draft: "Runtime Energy Attribution of Inference Requests in LLM Serving
  Systems".
