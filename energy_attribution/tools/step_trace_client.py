"""Drive concurrent requests against SGLang and validate the step trace.

    # 1. server with the tracer (inside the container):
    SGLANG_ENABLE_STEP_TRACE=1 python -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-1.5B-Instruct --port 30000 --disable-overlap-schedule

    # 2. send a concurrent workload, record what the server reported per rid:
    python energy_attribution/tools/step_trace_client.py send --url http://<server>:30000 --out manifest.json

    # 3. check the trace against the manifest:
    python energy_attribution/tools/step_trace_client.py check --manifest manifest.json \
        --trace /tmp/sglang_step_trace/step_trace.dp0_tp0_pp0.<pid>.jsonl

Stdlib only, so it runs anywhere (Mac, WSL, container).
"""

import argparse
import time
import json
import random
import sys
import urllib.request
import uuid
from typing import Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

SHARED_PREFIX = (
    "You are a meticulous assistant. Answer concisely and precisely. "
    "Context: the following is a benchmark of batched inference. " * 8
)
WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()


def _now_ns() -> int:
    """Current time in ns on CLOCK_MONOTONIC_RAW, the clock the step trace uses.

    The raw clock is never slewed, unlike time.monotonic_ns(), which drifted 2-4%
    against the GPU timer under WSL2. Not available on native Windows.
    """
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def _post(url: str, payload: dict, timeout: int=600) -> Any:
    """POST payload as JSON to url and return the decoded JSON response."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _make_prompt(rng: random.Random, i: int, shared_prefix: bool) -> str:
    """Build a synthetic prompt for request i.

    The body is 20-1500 random words, so prompt lengths vary widely and long
    prompts exceed a small --chunked-prefill-size (chunked prefill). With
    shared_prefix, the prompt starts with SHARED_PREFIX, so every such request
    after the first can reuse it from the radix cache (prefix reuse).
    """
    body = " ".join(rng.choice(WORDS) for _ in range(rng.randint(20, 1500)))
    head = SHARED_PREFIX if shared_prefix else ""
    return f"{head}Request {i}: summarize this list: {body}"


def _one_request(
    url: str, rid: str, prompt: str, max_new_tokens: int, delay_s: float
) -> dict:
    """Send one /generate request after delay_s and return its manifest record.

    The request carries our own rid, so the server uses it as the request ID and
    its owner rows in the trace can be joined back to this record. ignore_eos
    stops the model from ending early at its end-of-sequence token, so the
    request generates exactly max_new_tokens (unless aborted or cut off by the
    context length); generation lengths then come from the seeded workload, not
    from the model. Greedy sampling (temperature 0) makes the output
    deterministic; it does not affect the length. The record keeps what
    the server reports in meta_info (prompt, completion and cached tokens): the
    reference the trace is checked against.
    """
    time.sleep(delay_s)
    t_send = _now_ns()
    out = _post(
        f"{url}/generate",
        {
            "rid": rid,
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        },
    )
    meta = out["meta_info"]
    return {
        "rid": rid,
        "prompt_tokens": meta["prompt_tokens"],
        "completion_tokens": meta["completion_tokens"],
        "cached_tokens": meta.get("cached_tokens", 0),
        "finish_reason": meta.get("finish_reason"),
        # Client clock; comparable to the trace only when run on the server host.
        "client_send_ns": t_send,
        "client_recv_ns": _now_ns(),
    }


def cmd_send(args: argparse.Namespace) -> None:
    """Send a concurrent workload and write the server's per-request report.

    Builds --num-requests jobs with seeded randomness (same --seed, same
    workload), each with a unique rid, a prompt (every second one with the
    shared prefix), a random max_new_tokens up to --max-new-tokens, and a random
    start delay within --arrival-spread-s. Staggered arrivals make new prefills
    land while others decode, which produces mixed batches. Up to --concurrency
    requests are in flight at once. Writes the list of records from _one_request
    to --out (manifest.json).
    """
    rng = random.Random(args.seed)
    jobs = []
    for i in range(args.num_requests):
        # These are the arguments (except url) for _one_request
        jobs.append(
            dict(
                rid=f"st-{uuid.uuid4().hex[:12]}-{i}",
                prompt=_make_prompt(rng, i, shared_prefix=i % 2 == 0),
                max_new_tokens=rng.randint(8, args.max_new_tokens),
                # Staggered arrivals produce mixed prefill/decode steps.
                delay_s=rng.uniform(0, args.arrival_spread_s),
            )
        )
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(_one_request, args.url, **job) for job in jobs]
        results = [f.result() for f in futures]
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {len(results)} request records to {args.out}")


def _load_trace(path: str) -> dict:
    """Read a trace JSONL file into lists of records keyed by their kind.

    Returns {"meta": [...], "anchor": [...], "step": [...], "owner": [...]}.
    """
    records = defaultdict(list)
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            records[rec["kind"]].append(rec)
    return records


def cmd_check(args: argparse.Namespace) -> None:
    """Validate a step trace against the manifest from the same run.

    Step checks, on the trace alone:
    - every step has a positive duration;
    - without overlap, steps do not overlap in time, and each GPU window
      [t_start_ns, t_end_ns] lies inside its host window [host_launch_begin_ns,
      host_result_ns], within the anchor uncertainty plus --clock-tolerance-ns;
    - the number of owner rows equals batch_size (for traced steps);
    - the owners' selected_tokens sum to the step's prefill + decode tokens.

    Request checks, trace against the server's report (manifest):
    - every manifest request has owner rows;
    - prefill tokens computed + prefix reused = prompt_tokens;
    - prefix reused = the server's cached_tokens;
    - decode rows = completion_tokens - 1 (the prefill step samples the first
      token, each decode step one more);
    - the last prefill chunk is not marked is_chunked.
    Retracted requests are skipped, since their prefill was partly recomputed.

    Prints the worst anchor uncertainty, the largest GPU-outside-host skew and
    how many mixed, chunked and prefix-reuse rows the run exercised. Exits with
    status 1 and lists up to 50 failures if any check fails.
    """
    with open(args.manifest) as f:
        manifest = {r["rid"]: r for r in json.load(f)}
    trace = _load_trace(args.trace)
    steps = sorted(trace["step"], key=lambda r: r["step_id"])
    owners = trace["owner"]
    failures = []

    def fail(msg):
        failures.append(msg)

    print(
        f"meta={trace['meta']}\nanchors={len(trace['anchor'])} "
        f"steps={len(steps)} owners={len(owners)}"
    )
    worst_anchor = max((a["uncertainty_ns"] for a in trace["anchor"]), default=0)
    print(f"worst anchor uncertainty: {worst_anchor / 1e3:.1f} us")

    # Step-level invariants.
    owners_by_step = defaultdict(list)
    for o in owners:
        owners_by_step[o["step_id"]].append(o)
    overlap = any(s["overlap_enabled"] for s in steps)
    for prev, cur in zip(steps, steps[1:]):
        if not overlap and cur["t_start_ns"] < prev["t_end_ns"]:
            fail(f"steps {prev['step_id']} and {cur['step_id']} overlap in time")
    anchors = {a["anchor_id"]: a for a in trace["anchor"]}
    max_skew_ns = 0
    for s in steps:
        if s["t_end_ns"] <= s["t_start_ns"]:
            fail(f"step {s['step_id']}: non-positive duration")
        if not overlap:
            # Without overlap the GPU window must sit inside the host window
            # from launch begin to result processing.
            tol = anchors[s["anchor_id"]]["uncertainty_ns"] + args.clock_tolerance_ns
            skew = max(
                s["host_launch_begin_ns"] - s["t_start_ns"],
                s["t_end_ns"] - s["host_result_ns"],
            )
            max_skew_ns = max(max_skew_ns, skew)
            if skew > tol:
                fail(f"step {s['step_id']}: GPU window outside host window by {skew} ns")
        rows = owners_by_step.get(s["step_id"], [])
        if s["owners_traced"] and len(rows) != s["batch_size"]:
            fail(f"step {s['step_id']}: {len(rows)} owners != batch_size")
        if sum(r["selected_tokens"] for r in rows) != (
            s["prefill_tokens"] + s["decode_tokens"]
        ):
            fail(f"step {s['step_id']}: token totals disagree with owners")

    # Request-level reconciliation against what the server reported.
    rows_by_rid = defaultdict(list)
    for o in owners:
        rows_by_rid[o["rid"]].append(o)
    for rid, req in manifest.items():
        rows = sorted(rows_by_rid.get(rid, []), key=lambda r: r["step_id"])
        if not rows:
            fail(f"{rid}: no owner records")
            continue
        if any(r["retraction_count"] for r in rows):
            print(f"{rid}: retracted, skipping token reconciliation")
            continue
        prefill = [r for r in rows if r["phase"] == "prefill"]
        decode = [r for r in rows if r["phase"] == "decode"]
        computed = sum(r["selected_tokens"] for r in prefill)
        reused = prefill[0]["cached_tokens"] if prefill else 0
        if computed + reused != req["prompt_tokens"]:
            fail(
                f"{rid}: prefill computed {computed} + reused {reused} "
                f"!= prompt_tokens {req['prompt_tokens']}"
            )
        if reused != req["cached_tokens"]:
            fail(f"{rid}: reused {reused} != server cached_tokens {req['cached_tokens']}")
        # The prefill step samples the first token; each decode step one more.
        if len(decode) != req["completion_tokens"] - 1:
            fail(
                f"{rid}: {len(decode)} decode steps "
                f"!= completion_tokens - 1 = {req['completion_tokens'] - 1}"
            )
        if any(r["is_chunked"] for r in prefill[-1:]):
            fail(f"{rid}: last prefill chunk marked is_chunked")

    if not overlap:
        print(f"max GPU-outside-host skew: {max_skew_ns / 1e3:.1f} us (<= 0 is ideal)")
    n_mixed = sum(1 for s in steps if s["num_prefill_reqs"] and s["num_decode_reqs"])
    n_chunked = sum(1 for o in owners if o["is_chunked"])
    n_cached = sum(1 for o in owners if o["phase"] == "prefill" and o["cached_tokens"])
    print(
        f"coverage: mixed steps={n_mixed}, chunked owner rows={n_chunked}, "
        f"prefill rows with prefix reuse={n_cached}"
    )
    if failures:
        print(f"\n{len(failures)} FAILURES")
        for msg in failures[:50]:
            print("  " + msg)
        sys.exit(1)
    print("\nOK")


def main() -> None:
    """Parse the command line and run the send or check subcommand."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("send")
    p.add_argument("--url", default="http://localhost:30000")
    p.add_argument("--out", default="manifest.json")
    p.add_argument("--num-requests", type=int, default=32)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--arrival-spread-s", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_send)
    c = sub.add_parser("check")
    c.add_argument("--manifest", required=True)
    c.add_argument("--trace", required=True)
    c.add_argument("--clock-tolerance-ns", type=int, default=200_000)
    c.set_defaults(fn=cmd_check)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
