# Native request-energy validation

Validation date: 2 November 2026 (lab UTC logs extend into 3 November).
These are TP=1 software conformance checks, not accuracy or overhead results.
The tested runtime revision is `58e20c3095306a32ca4b7a24fd2c3dbf3429a905`.
Subsequent receipt/documentation-only commits retain those runtime bytes.
Exact source, model-file, SDK, image-config and result hashes are in the
[machine-readable receipt](validation/2026-11-02.json). Private addresses and credentials are excluded.

## Runtime and procedure

Both engines use llm-request-energy **0.2.0**, built in the independent attribution project.
The SDK is a separate development wheel, not assumed published on PyPI.
Its SHA-256 is `792954f12ccf5d8957b057552cacbd8a25ddbbeb297377e0f8fcec6067628ac6`.
Tracing disabled requires no SDK imports from these instrumentation paths.

Each full case used eight measured requests, four client threads, two short warmups,
15 seconds of loaded-engine idle calibration, 100 ms polling and 10-second pre/post windows.
The profile was eager dense text, context 4096, chunk size 128, fixed greedy output lengths,
and the computed-token allocation policy with the declared residency floor.
RTX used Qwen3-0.6B-Base with pinned file hashes; A100 used Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`.
SGLang used Triton attention and PyTorch sampling.

| Hardware | Mode | Final requests | GPU estimate range | Native steps / executions |
| --- | --- | ---: | ---: | ---: |
| RTX 4090 | overlap | 8/8 | 9.25–21.57 J | 106 / 106 |
| RTX 4090 | nonoverlap | 8/8 | 8.48–19.32 J | 99 / 99 |
| RTX 4090 | delayed | 8/8 | 8.83–20.49 J | 105 / 210 |
| A100 80GB | overlap | 8/8 | 17.46–67.66 J | 105 / 105 |
| A100 80GB | nonoverlap | 8/8 | 14.02–28.55 J | 99 / 99 |
| A100 80GB | delayed | 8/8 | 15.56–31.60 J | 105 / 210 |

All cases reconciled the measured request inventory, returned positive GPU estimates,
and passed software integrity and conservation. Live/replay request output and ledgers
were byte-identical. Equal-share and activity-weighted offline replays also passed.
The largest absolute ledger residual in these sglang cases was
`1.24345e-14 J`, below
`max(1e-6 J, 1e-9 * reconstructed domain energy)`.
The estimate ranges include baseline-floor allocations; they are not hardware/engine rankings.

## Compatibility and output observations

The contributor baseline is `90c8d408466081a030f1f37ace8021182f66d5c3`.
The original checker and plotter are byte-for-byte unchanged from that revision.
Every case passed `step_trace_client.py check` and `validate_canonical_trace.py`.
The unchanged plotter generated four PNGs per case, 24 total; hashes are in the receipt.
The delayed-sampling cases each matched 105 native envelopes to 210 constituent CUDA executions.
Native envelopes are diagnostic and never an extra energy input.

The branch-local tests are `python -m pytest energy_attribution/tests/test_dual_trace.py` (2 passed).

The independent attribution project's combined suite passed 64 tests, including these
adapter fixtures, immutable observations, asynchronous writer health and protocol-family rejection.

Both ordinary-overlap trace-on/off comparisons matched all eight response hashes and token counts.

Concurrent fixed greedy serving is not generally guaranteed bitwise reproducible;
these comparisons are observations under the recorded profiles, not universal equivalence proofs.

## Measurement limits

RTX driver 580.196.07 and A100 driver 580.178.04 remained unchanged in this merge pass.
RTX collected NVML and genuine RAPL package energy; no DRAM domain was available.
A100 collected NVML only. The other A100 retained its public service, so shared-host
interference remains possible. Two short warmups do not ensure every JIT shape is warm;
the larger A100 overlap observations are retained.

100 ms polling does not establish 100 ms sensor resolution. RTX power observations
retain the declared one-second averaging support. CPU cgroup ratios were not sampled;
CPU/DRAM model accuracy remains unvalidated. Conservation establishes accounting
consistency, not physical per-request ground truth.

Native cancellation/preemption, CUDA graphs, alternate vLLM runner coverage and TP=2
remain open. The occupied A100 pair was not used for TP=2. Reference accuracy, sensor
characterization and paired overhead still require the planned research campaign.
`hardware_validated` and `research_ready` remain false.
