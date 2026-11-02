# Shared native and canonical tracing

This feature branch extends `resource-tracking` at
`90c8d408466081a030f1f37ace8021182f66d5c3`. Its original
`tools/step_trace_client.py` and `tools/plot_step_trace.py` are retained unchanged.

Install the matching **llm-request-energy 0.2.0** SDK wheel in the engine
environment. The SDK core has no NVML, Java or experiment-framework dependency;
Torch is imported only when CUDA tracing is enabled. The wheel is built in the
independent attribution project; do not assume that this development version is
already available on PyPI. Normal serving with tracing disabled needs no SDK.

* `SGLANG_ENABLE_STEP_TRACE=1` enables native files in `SGLANG_STEP_TRACE_DIR`.
  A native-only run needs no `REQUEST_ENERGY_*` settings.
* `REQUEST_ENERGY_TRACE_DIR` enables canonical streams. Set a unique
  `REQUEST_ENERGY_INSTANCE_ID` shared by all TP ranks. The standalone collector
  needs TP-size + 1 canonical producers.
* Set both switches to export both formats. Use **different directories**.
  Native `step_trace.<rank>.<pid>.<nonce>.jsonl` files retain schema 1 and the
  `meta/anchor/step/owner` records. The nonce prevents producer-restart collisions.
  Their format identity is `sglang.step-trace.native`; canonical metadata says
  `request-energy.canonical`. Equal schema numbers do not imply interchangeability.

One immutable CPU snapshot supplies both ownership representations. It preserves
prefix reuse, chunk boundaries, retraction counts and executed recomputation.
Native `kv_read_tokens` remains the original context-minus-selected proxy;
canonical `kv_read_token_pairs` includes its explicit triangular term. Neither
is measured memory traffic.

One shared CUDA resolver records forward work after its scheduling-stream
dependency, and ends it at the original post-`run_batch` boundary. Deferred
sampling has a separate event pair and the same explicit `logical_step_id`.
The native envelope spans these constituent intervals and lists their IDs and
times. It is diagnostic: only canonical constituents enter energy allocation.
No duplicate event pairs or per-step device synchronization are introduced.
Initial GPU/RAW-clock anchoring remains explicit; later anchors are asynchronous
and idle-gated. Bracket uncertainty and modeled drift are retained.

The common bounded writer flushes within 50 ms or the configured record count.
Sequence fields expose missing records. Native queue/writer failures invalidate
the canonical producer too. Native shutdown health is written to a separate
`.health.json`, preserving the four record types accepted by the original plotter.
Request terminals do not discard outstanding forward/sampling ownership.

Run the original checker and plotter with native output, then check that both
serializers used identical execution/demand observations:

```bash
python energy_attribution/tools/step_trace_client.py check \
  --manifest manifest.json --trace native/step_trace.<rank>.<pid>.<nonce>.jsonl
python energy_attribution/tools/validate_canonical_trace.py \
  --native native/step_trace.<rank>.<pid>.<nonce>.jsonl \
  --canonical 'canonical/*.jsonl'
python energy_attribution/tools/plot_step_trace.py \
  --trace 'native/step_trace.*.jsonl' --manifest manifest.json --out plots
python -m pytest energy_attribution/tests/test_dual_trace.py
```

The native checker has profile-specific token-count expectations, including a
trailing discarded decode under ordinary overlap. Cancellation, speculative
work and retraction must not be judged using that rule indiscriminately.
Speculation, beams, multimodal/SSM, offloading, disaggregation, DP and PP are
outside the trusted initial profile. The processor owns NVML/RAPL acquisition,
energy policies, conservation and live/replay; engines never calculate joules.

Native serving verification and its limits are recorded in the branch's
validation receipt after execution; conservation alone does not prove physical
per-request energy accuracy or low overhead.
