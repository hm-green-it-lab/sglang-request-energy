"""Native SGLang and canonical traces from one frozen ownership/timing source.

Preserves PR #1's meta/anchor/step/owner wire contract and its checker/plots.
The optional shared SDK resolves one event pair per execution. Native envelopes
are diagnostics; only canonical constituent intervals enter energy allocation.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel

NATIVE_FORMAT = "sglang.step-trace.native"


@dataclass(frozen=True)
class OwnerSnapshot:
    rid: str
    phase: str
    selected_tokens: int
    context_len: int
    cached_tokens: int
    req_cached_tokens: int
    is_chunked: bool
    retraction_count: int
    recompute_tokens: int = 0

    def native(self, step_id):
        return {"kind": "owner", "step_id": step_id, **asdict(self)}

    def canonical(self):
        n = self.selected_tokens
        return {
            "request_id": self.rid,
            "phase": self.phase,
            "computed_tokens": n,
            "prefill_tokens": n if self.phase == "prefill" else 0,
            "decode_tokens": n if self.phase == "decode" else 0,
            "context_tokens": self.context_len,
            "recompute_tokens": self.recompute_tokens,
            "cache_reused_tokens": self.cached_tokens,
            "radix_cache_hits_total": self.req_cached_tokens,
            "retraction_count": self.retraction_count,
            "is_chunked": self.is_chunked,
            "kv_read_token_pairs": n * (self.context_len - n) + n * (n + 1) // 2,
            "flops_estimate": None,
            "read_bytes_estimate": None,
            "write_bytes_estimate": None,
        }


@dataclass(frozen=True)
class StepSnapshot:
    step_id: int
    forward_mode: str
    batch_size: int
    owners: tuple[OwnerSnapshot, ...]
    owners_traced: bool
    host_launch_begin_ns: int


class NoopStepTracer:
    def prepare(self, batch):
        pass

    def on_launch_begin(self, batch):
        pass

    def on_launch_end(self, batch):
        pass

    def on_sample_begin(self, batch):
        pass

    def on_sample_end(self, batch):
        pass

    def on_delayed_sample(self, batch):
        pass

    def on_result(self, batch):
        pass

    def on_idle(self):
        pass

    def arrive(self, req, is_retracted=False):
        pass

    def abort(self, req):
        pass

    def close(self):
        pass


class SchedulerStepTracer:
    def __init__(self, *, gpu_id, enable_overlap, forward_stream, supported_profile):
        from request_energy import clock
        from request_energy.instrumentation import CudaStepTracer, LifecycleTracer
        from request_energy.trace import TraceWriter

        self.clock = clock
        self._lock = threading.RLock()
        self._idle, self._closed = True, False
        self.prepared, self.recompute_ranges = {}, {}
        self.known = set()
        self.native = self.lifecycle = None
        self.enable_overlap = enable_overlap
        self.stream = forward_stream if enable_overlap else None
        self.supported_profile = supported_profile
        canonical = bool(os.environ.get("REQUEST_ENERGY_TRACE_DIR"))
        parallel = get_parallel()
        self.device = CudaStepTracer(
            "sglang",
            f"tp{parallel.tp_rank}",
            device=gpu_id,
            emit_canonical=canonical,
            on_observation=self._on_observation,
            on_anchor=self._on_anchor,
            reanchor_when=self._can_reanchor,
            reanchor_seconds=envs.SGLANG_STEP_TRACE_REANCHOR_INTERVAL_S.get(),
        )
        if canonical and parallel.tp_rank == 0:
            self.lifecycle = LifecycleTracer("sglang", "scheduler")
        if envs.SGLANG_ENABLE_STEP_TRACE.get():
            directory = Path(envs.SGLANG_STEP_TRACE_DIR.get())
            if (
                canonical
                and directory.resolve()
                == Path(os.environ["REQUEST_ENERGY_TRACE_DIR"]).resolve()
            ):
                raise RuntimeError("Native and canonical trace directories must differ")
            rank = (
                f"dp{parallel.dp_rank or 0}_tp{parallel.tp_rank}_pp{parallel.pp_rank}"
            )
            name = f"step_trace.{rank}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
            self.path = directory / (name + ".jsonl")
            self.health_path = directory / (name + ".health.json")
            self.native = TraceWriter(
                self.path,
                {
                    "format_id": NATIVE_FORMAT,
                    "producer_id": name,
                    "role": "engine_native",
                    "clock": clock.CLOCK,
                    "boot_id": clock.boot_id(),
                    "pid": os.getpid(),
                    "rank_tag": rank,
                    "gpu_id": gpu_id,
                    "device_name": self.device.device_name,
                    "device_uuid": self.device.device_id,
                    "enable_overlap": enable_overlap,
                    "time_source": "cuda_event",
                },
                flush_records=envs.SGLANG_STEP_TRACE_FLUSH_STEPS.get(),
            )
            self._on_anchor(
                {
                    "kind": "anchor",
                    "anchor_id": self.device.anchor_id,
                    "host_ns": self.device.anchor_host_ns,
                    "uncertainty_ns": self.device.anchor_uncertainty_ns,
                }
            )
        atexit.register(self.close)

    def _can_reanchor(self):
        with self._lock:
            return self._idle and not self.prepared

    def _native_emit(self, record):
        if self.native and not self.native.emit(record):
            raise RuntimeError("Native trace queue overflow; attribution is invalid")

    def _on_anchor(self, record):
        with self._lock:
            self._native_emit(record)

    def arrive(self, req, is_retracted=False):
        if self.lifecycle:
            if req.rid not in self.known:
                self.lifecycle.request(req.rid, "arrive", client_id=req.rid)
                self.known.add(req.rid)
            if is_retracted:
                self.lifecycle.request(req.rid, "release")

    def abort(self, req):
        if self.lifecycle and req.rid in self.known:
            self.lifecycle.request(req.rid, "cancel", outcome="aborted")
        self.known.discard(req.rid)
        self.recompute_ranges.pop(req.rid, None)

    def prepare(self, batch):
        if self.device.health()["error"]:
            raise RuntimeError("GPU trace resolver failed")
        if self.lifecycle:
            self.lifecycle.hold(str(batch.forward_iter))
        owners, supported = _snapshot_native_owners(batch)
        updated = []
        for owner in owners:
            rid = owner.rid
            if self.lifecycle:
                self.lifecycle.admit(rid, client_id=rid)
                self.known.add(rid)
            before, end = owner.context_len - owner.selected_tokens, owner.context_len
            ranges = self.recompute_ranges.get(rid, [])
            repeated = sum(
                max(0, min(end, right) - max(before, left)) for left, right in ranges
            )
            merged = []
            for left, right in sorted([*ranges, (before, end)]):
                if merged and left <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
                else:
                    merged.append((left, right))
            self.recompute_ranges[rid] = merged
            updated.append(replace(owner, recompute_tokens=repeated))
        snapshot = StepSnapshot(
            batch.forward_iter,
            batch.forward_mode.name,
            len(batch.reqs),
            tuple(updated),
            supported and self.supported_profile,
            self.clock.now_ns(),
        )
        with self._lock:
            if len(self.prepared) >= 8192 or batch.forward_iter in self.prepared:
                raise RuntimeError("Invalid or excessive outstanding scheduler steps")
            self._idle = False
            self.prepared[batch.forward_iter] = {
                "snapshot": snapshot,
                "requests": tuple(batch.reqs),
                "expected": {"forward"},
                "intervals": {},
                "result_ns": None,
                "launch_end_ns": None,
            }

    def _begin(self, batch, work_kind):
        from request_energy.instrumentation import sampling_owners

        with self._lock:
            entry = self.prepared[batch.forward_iter]
            entry["expected"].add(work_kind)
            snapshot = entry["snapshot"]
            owners = [o.canonical() for o in snapshot.owners]
        if work_kind == "sampling":
            owners = sampling_owners(owners)
        # Do not hold the snapshot lock while entering the timer: its resolver
        # invokes our callback under the timer lock (the opposite lock order).
        self.device.begin(
            f"{snapshot.step_id}:{work_kind}",
            owners,
            stream=self.stream,
            supported=snapshot.owners_traced,
            work_kind=work_kind,
            logical_step_id=str(snapshot.step_id),
        )

    def on_launch_begin(self, batch):
        self._begin(batch, "forward")

    def on_launch_end(self, batch):
        self.device.end(f"{batch.forward_iter}:forward", stream=self.stream)
        with self._lock:
            self.prepared[batch.forward_iter]["launch_end_ns"] = self.clock.now_ns()

    def on_sample_begin(self, batch):
        self._begin(batch, "sampling")

    def on_sample_end(self, batch):
        self.device.end(f"{batch.forward_iter}:sampling", stream=self.stream)

    def on_delayed_sample(self, batch):
        # Actual deferred work has a sampling pair; no duplicate end event.
        pass

    def _on_observation(self, observation):
        record = observation.record()
        step_id = int(record["logical_step_id"])
        with self._lock:
            self.prepared[step_id]["intervals"][record["work_kind"]] = record
            self._finish_native(step_id)

    def _finish_native(self, step_id):
        # Snapshot lock held; never calls into the CUDA timer.
        entry = self.prepared[step_id]
        if entry["result_ns"] is None or entry["expected"] != entry["intervals"].keys():
            return
        s = entry["snapshot"]
        intervals = list(entry["intervals"].values())
        forward = entry["intervals"]["forward"]
        owners = s.owners if s.owners_traced else ()
        prefill = [o for o in owners if o.phase == "prefill"]
        decode = [o for o in owners if o.phase == "decode"]
        self._native_emit(
            {
                "kind": "step",
                "step_id": step_id,
                "logical_step_id": str(step_id),
                "forward_mode": s.forward_mode,
                "batch_size": s.batch_size,
                "t_start_ns": min(r["start_ns"] for r in intervals),
                "t_end_ns": max(r["end_ns"] for r in intervals),
                "host_launch_begin_ns": s.host_launch_begin_ns,
                "host_launch_end_ns": entry["launch_end_ns"],
                "host_result_ns": entry["result_ns"],
                "anchor_id": forward["anchor_id"],
                "uncertainty_ns": max(r["uncertainty_ns"] for r in intervals),
                "num_prefill_reqs": len(prefill),
                "num_decode_reqs": len(decode),
                "prefill_tokens": sum(o.selected_tokens for o in prefill),
                "decode_tokens": sum(o.selected_tokens for o in decode),
                "cached_tokens": sum(o.cached_tokens for o in prefill),
                "kv_read_tokens": sum(
                    o.context_len - o.selected_tokens for o in owners
                ),
                "overlap_enabled": self.enable_overlap,
                "owners_traced": s.owners_traced,
                "execution_intervals": [
                    {
                        k: r[k]
                        for k in (
                            "execution_id",
                            "work_kind",
                            "start_ns",
                            "end_ns",
                            "anchor_id",
                        )
                    }
                    for r in intervals
                ],
            }
        )
        for owner in owners:
            self._native_emit(owner.native(step_id))
        del self.prepared[step_id]

    def on_result(self, batch):
        with self._lock:
            entry = self.prepared[batch.forward_iter]
            requests = entry["requests"]
            entry["result_ns"] = self.clock.now_ns()
            self._finish_native(batch.forward_iter)
        if self.lifecycle:
            self.lifecycle.release_hold(str(batch.forward_iter))
        for req in requests:
            if req.finished():
                if self.lifecycle and req.rid in self.known:
                    self.lifecycle.request(
                        req.rid, "finish", outcome=type(req.finished_reason).__name__
                    )
                    self.known.remove(req.rid)
                self.recompute_ranges.pop(req.rid, None)

    def on_idle(self):
        with self._lock:
            self._idle = True
        self.device.completed()
        if self.device.health()["error"]:
            raise RuntimeError("GPU/native trace resolver failed")

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.device.close()
        finally:
            if self.lifecycle:
                self.lifecycle.close()
            if self.native:
                try:
                    self.native.close()
                finally:
                    health = self.native.health()
                    health.update(
                        format_id=NATIVE_FORMAT,
                        outstanding_steps=len(self.prepared),
                        resolver=self.device.health(),
                        complete=not self.prepared
                        and not self.device.health()["error"]
                        and not health["error"]
                        and not health["dropped_records"],
                    )
                    self.health_path.write_text(json.dumps(health) + "\n")


def _snapshot_native_owners(batch):
    """PR #1 ownership semantics; no GPU read or mutable request references."""
    if not batch.spec_algorithm.is_none() or getattr(batch, "beam_tail", None):
        return (), False
    if any(
        getattr(r, "beam_group", None) is not None or r.multimodal_inputs is not None
        for r in batch.reqs
    ):
        return (), False
    if batch.forward_mode.is_decode():
        if batch.seq_lens_cpu is None:
            return (), False
        rows = [
            (r, "decode", 1, int(n), 0, False)
            for r, n in zip(
                batch.reqs, batch.seq_lens_cpu[: len(batch.reqs)].tolist(), strict=True
            )
        ]
    elif batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
        decode_ids = {id(r) for r in batch.decoding_reqs or []}
        rows = [
            (
                r,
                "decode" if id(r) in decode_ids else "prefill",
                int(n),
                int(p + n),
                0 if id(r) in decode_ids else int(p),
                id(r) not in decode_ids and r is batch.chunked_req,
            )
            for r, p, n in zip(
                batch.reqs, batch.prefix_lens, batch.extend_lens, strict=True
            )
        ]
    else:
        return (), False
    return tuple(
        OwnerSnapshot(
            r.rid,
            phase,
            n,
            context,
            cached,
            int(r.cached_tokens),
            chunked,
            int(r.retraction_count),
        )
        for r, phase, n, context, cached, chunked in rows
    ), True


def _snapshot_owners(batch):
    owners, supported = _snapshot_native_owners(batch)
    return [o.canonical() for o in owners], supported


def create_step_tracer(
    *, gpu_id, device, enable_overlap, forward_stream, supported_profile=True
):
    if not (
        envs.SGLANG_ENABLE_STEP_TRACE.get()
        or os.environ.get("REQUEST_ENERGY_TRACE_DIR")
    ):
        return NoopStepTracer()
    parallel = get_parallel()
    if (
        device != "cuda"
        or not supported_profile
        or parallel.pp_size != 1
        or getattr(parallel, "dp_size", 1) > 1
    ):
        raise RuntimeError(
            "Tracing requires supported single-host CUDA text generation"
        )
    return SchedulerStepTracer(
        gpu_id=gpu_id,
        enable_overlap=enable_overlap,
        forward_stream=forward_stream,
        supported_profile=True,
    )
