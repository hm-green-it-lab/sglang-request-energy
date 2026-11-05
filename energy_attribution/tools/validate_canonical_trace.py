"""Compare native envelopes/owners with their canonical execution observations.

Stdlib only. This complements the unchanged step_trace_client.py checker;
it does not allocate energy or replace request/token reconciliation.
"""

import argparse
import glob
import json
from pathlib import Path


def records(path):
    path = Path(path)
    if any(
        p.name == ".env" or p.name.startswith(".env.") for p in (path, path.resolve())
    ):
        raise ValueError("Environment files are not trace inputs")
    with path.open() as handle:
        rows = [json.loads(line) for line in handle]
    if not rows or rows[0]["kind"] != "meta":
        raise ValueError("Missing producer metadata")
    if [r["sequence"] for r in rows] != list(range(len(rows))):
        raise ValueError("Lost, duplicated or reordered records")
    if any(r.get("dropped_records", 0) for r in rows):
        raise ValueError("Producer reported record loss")
    return rows


def compare(native_path, canonical_paths):
    native = records(native_path)
    meta = native[0]
    if meta.get("format_id") != "sglang.step-trace.native":
        raise ValueError("Not an integrated native trace")
    gpu = meta["device_uuid"]
    canonical = {}
    for path in canonical_paths:
        rows = records(path)
        if rows[0].get("format_id") != "request-energy.canonical":
            raise ValueError("Not a canonical stream")
        for r in rows:
            if r["kind"] == "step" and r["device_id"] == gpu:
                key = (r["logical_step_id"], r["execution_id"])
                if key in canonical:
                    raise ValueError("Duplicate canonical execution")
                canonical[key] = r
    owners = {}
    for r in native:
        if r["kind"] == "owner":
            owners.setdefault(r["step_id"], []).append(r)
    seen = set()
    steps = [r for r in native if r["kind"] == "step"]
    if len({r["step_id"] for r in steps}) != len(steps):
        raise ValueError("Duplicate native step")
    for step in steps:
        intervals = step["execution_intervals"]
        if step["t_start_ns"] != min(r["start_ns"] for r in intervals) or step[
            "t_end_ns"
        ] != max(r["end_ns"] for r in intervals):
            raise ValueError("Native envelope disagrees with its constituent intervals")
        expected_owners = {r["rid"]: r for r in owners.get(step["step_id"], [])}
        for interval in intervals:
            key = (step["logical_step_id"], interval["execution_id"])
            if key in seen or key not in canonical:
                raise ValueError("Missing or multiply charged execution reference")
            seen.add(key)
            observed = canonical[key]
            if any(observed[k] != v for k, v in interval.items()):
                raise ValueError("Native/canonical timing observations differ")
            actual_owners = {r["request_id"]: r for r in observed["owners"]}
            if actual_owners.keys() != expected_owners.keys():
                raise ValueError("Native/canonical ownership inventory differs")
            if observed["owners_traced"] != step["owners_traced"]:
                raise ValueError("Inconsistent supported ownership")
            for rid, n in expected_owners.items():
                c = actual_owners[rid]
                shared = interval["work_kind"] == "sampling"
                expected = {
                    "phase": "shared" if shared else n["phase"],
                    "computed_tokens": 0 if shared else n["selected_tokens"],
                    "context_tokens": n["context_len"],
                    "cache_reused_tokens": n["cached_tokens"],
                    "radix_cache_hits_total": n["req_cached_tokens"],
                    "retraction_count": n["retraction_count"],
                    "is_chunked": n["is_chunked"],
                    "recompute_tokens": 0 if shared else n["recompute_tokens"],
                    "kv_read_token_pairs": 0
                    if shared
                    else (
                        n["selected_tokens"] * (n["context_len"] - n["selected_tokens"])
                        + n["selected_tokens"] * (n["selected_tokens"] + 1) // 2
                    ),
                }
                if any(c[k] != v for k, v in expected.items()):
                    raise ValueError("Native/canonical demand snapshots differ")
    if seen != canonical.keys():
        raise ValueError("Canonical executions are missing from native output")
    return {"ok": True, "native_steps": len(steps), "shared_executions": len(seen)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True)
    parser.add_argument("--canonical", nargs="+", required=True)
    args = parser.parse_args()
    paths = [p for pattern in args.canonical for p in glob.glob(pattern)]
    if not paths:
        parser.error("No canonical streams matched")
    print(json.dumps(compare(args.native, paths), indent=2))
