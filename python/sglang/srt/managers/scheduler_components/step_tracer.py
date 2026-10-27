"""Per-step scheduler trace for offline per-request energy attribution.

Each forward pass yields one ``step`` record (GPU-anchored start/end on the
host ``CLOCK_MONOTONIC_RAW`` clock plus batch totals) and one ``owner`` record
per request in the batch (its token demand in that step). Records go to a buffered JSONL
sidecar per scheduler process, keyed by ``step_id`` and joined to requests by
``rid``; an external power sampler (NVML / RAPL) on the same clock
can then be windowed onto steps and split across owners offline.

All ``*_ns`` fields are ``CLOCK_MONOTONIC_RAW``, not ``time.monotonic_ns()``:
the slewed ``CLOCK_MONOTONIC`` ran 2-4% off the GPU event timer under WSL2
(measured on an RTX 4090), while the raw clock tracked it to ~0.1%. A power
sampler must timestamp with the same clock to be joinable.

Three hooks per step, all host-only except the event records:

- ``on_launch_begin`` (after batch selection): snapshot owners and record the start CUDA event on the stream the forward runs on.
- ``on_launch_end`` (after ``run_batch`` returns): record the end CUDA event.
- ``on_result`` (in ``process_batch_result``): the forward has completed, so read both events, convert them to host ns via the clock anchor, and buffer the records.

GPU event times are mapped to the host clock with an anchor: drain the device,
record an event, and bracket its completion with the host clock; the
bracket half-width is the anchor uncertainty. Anchors are re-taken only when
that costs no overlap (non-overlap mode after a result, or when idle), and
every anchor is written to the trace so drift can be corrected offline.

Validated with ``--disable-overlap-schedule``. Under the overlap scheduler the
start event sits on ``forward_stream`` before it waits on the schedule stream,
so it marks when the previous forward drained rather than when this one began.
Speculative decoding, DP attention, PP and PD disaggregation are out of scope;
steps in those modes are written without owner records.

Enable with ``SGLANG_ENABLE_STEP_TRACE=1``; output lands in
``SGLANG_STEP_TRACE_DIR`` as ``step_trace.<rank tag>.<pid>.jsonl``.
"""

from __future__ import annotations

import atexit
import logging
import os
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)

TRACE_SCHEMA_VERSION = 1
TRACE_CLOCK = "CLOCK_MONOTONIC_RAW"


def trace_clock_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class MetaRecord(msgspec.Struct, tag_field="kind", tag="meta", frozen=True):
    schema_version: int
    clock: str
    pid: int
    rank_tag: str
    gpu_id: int
    device_name: str
    # Matches nvmlDeviceGetUUID ("GPU-..."), so a sampler can pick its device.
    device_uuid: Optional[str]
    enable_overlap: bool
    time_source: str


class AnchorRecord(msgspec.Struct, tag_field="kind", tag="anchor", frozen=True):
    anchor_id: int
    host_ns: int
    # Half-width of the host bracket around the anchor event's completion.
    uncertainty_ns: int


class StepRecord(msgspec.Struct, tag_field="kind", tag="step", frozen=True):
    step_id: int
    forward_mode: str
    batch_size: int
    # GPU execution window (device events mapped through anchor_id).
    t_start_ns: int
    t_end_ns: int
    # Host-side launch window around run_batch, for cross-checking.
    host_launch_begin_ns: int
    host_launch_end_ns: int
    host_result_ns: int
    anchor_id: int
    num_prefill_reqs: int
    num_decode_reqs: int
    prefill_tokens: int
    decode_tokens: int
    # Prefix tokens reused (radix hits and earlier chunks) by prefill owners.
    cached_tokens: int
    # KV tokens attended over by all owners (sum of context_len - selected_tokens).
    kv_read_tokens: int
    overlap_enabled: bool
    owners_traced: bool


class OwnerRecord(msgspec.Struct, tag_field="kind", tag="owner", frozen=True):
    step_id: int
    rid: str
    # "prefill" or "decode"; a mixed batch carries both.
    phase: str
    # New tokens computed for this owner in the step.
    selected_tokens: int
    # KV length after the step (prefix + selected).
    context_len: int
    # Prefix tokens reused in this step instead of recomputed.
    cached_tokens: int
    # Cumulative radix-cache hit tokens of the request.
    req_cached_tokens: int
    # True for a non-final prefill chunk.
    is_chunked: bool
    retraction_count: int


TraceRecord = Union[MetaRecord, AnchorRecord, StepRecord, OwnerRecord]


class _OpenStep(msgspec.Struct):
    forward_mode: str
    batch_size: int
    owners: List[OwnerRecord]
    owners_traced: bool
    host_launch_begin_ns: int
    start_event: Optional[torch.cuda.Event]
    host_launch_end_ns: int = 0
    end_event: Optional[torch.cuda.Event] = None


class NoopStepTracer:
    """Disabled tracer; keeps the scheduler call sites branch-free."""

    def on_launch_begin(self, batch: ScheduleBatch) -> None:
        pass

    def on_launch_end(self, batch: ScheduleBatch) -> None:
        pass

    def on_result(self, batch: ScheduleBatch) -> None:
        pass

    def on_idle(self) -> None:
        pass

    def close(self) -> None:
        pass


class SchedulerStepTracer:
    def __init__(
        self,
        *,
        output_dir: str,
        rank_tag: str,
        gpu_id: int,
        device: str,
        enable_overlap: bool,
        forward_stream: Optional[torch.cuda.Stream],
        flush_steps: int,
        reanchor_interval_s: float,
    ):
        self._enable_overlap = enable_overlap
        self._forward_stream = forward_stream
        self._flush_steps = max(1, flush_steps)
        self._reanchor_interval_ns = int(reanchor_interval_s * 1e9)
        self._use_device_events = device == "cuda" and torch.cuda.is_available()

        self._encoder = msgspec.json.Encoder()
        self._buffer: List[TraceRecord] = []
        self._steps_since_flush = 0
        self._open: Optional[_OpenStep] = None
        # Launched-but-unprocessed steps by forward_iter; >1 entry under overlap.
        self._pending: Dict[int, _OpenStep] = {}

        self._anchor_id = -1
        self._anchor_event: Optional[torch.cuda.Event] = None
        self._anchor_host_ns = 0

        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(
            output_dir, f"step_trace.{rank_tag}.{os.getpid()}.jsonl"
        )
        self._file = open(self.path, "ab")
        atexit.register(self.close)

        self._buffer.append(
            self._build_meta(rank_tag=rank_tag, gpu_id=gpu_id, device=device)
        )
        self._take_anchor()
        self._flush()
        logger.info(f"Step trace enabled, writing to {self.path}")

    # ===== Hooks =====

    def on_launch_begin(self, batch: ScheduleBatch) -> None:
        """
        A hook that runs after the scheduler has chosen the batch, before anything is launched.
            + Snapshots who is in the batch and how much work each request brings (`_snapshot_owners`). It reads only CPU-side data, so it never waits for the GPU.
            + Records the host time and a start event on the GPU stream. A CUDA event is a marker in the GPU's work queue that the GPU timestamps when it reaches it.
            + Keeps both in a temporary open-step object.
        """
        owners, owners_traced = _snapshot_owners(batch)
        self._open = _OpenStep(
            forward_mode=batch.forward_mode.name,
            batch_size=len(batch.reqs),
            owners=owners,
            owners_traced=owners_traced,
            host_launch_begin_ns=trace_clock_ns(),
            start_event=self._record_event(),
        )

    def on_launch_end(self, batch: ScheduleBatch) -> None:
        """
        A hook that is called after ``run_batch`` returns.
            + Records the end event. It sits behind all of the step's work (forward pass, sampling, copy of results to the CPU), so it fires when that work is done.
            + Files the open step under `batch.forward_iter`, the scheduler's step counter, which becomes `step_id`. The counter is only assigned inside run_batch, which is why the owner rows get their `step_id` later.
        """
        step = self._open
        self._open = None
        if step is None or batch.forward_iter is None:
            return
        step.end_event = self._record_event()
        step.host_launch_end_ns = trace_clock_ns()
        self._pending[batch.forward_iter] = step

    def on_result(self, batch: ScheduleBatch) -> None:
        """
        A hook that runs at the end of `process_batch_result`, when the GPU work has finished.
            + Looks up the open step by `forward_iter`, converts both events to host time (Clocks section), computes batch totals from the owner rows, and buffers one step record plus its owner records.
            + Without overlap, re-anchors the clock if the last anchor is older than the interval. That is cheap here because the GPU is already idle.
            + Writes the buffer to disk every `FLUSH_STEPS` steps.
        """
        step = self._pending.pop(batch.forward_iter, None)
        if step is None:
            return
        self._emit_step(step_id=batch.forward_iter, step=step)
        if not self._enable_overlap:
            # The forward has drained, so a fresh anchor costs no overlap.
            self._maybe_reanchor()
        self._steps_since_flush += 1
        if self._steps_since_flush >= self._flush_steps:
            self._flush()

    def on_idle(self) -> None:
        """
        A hook that runs when the scheduler has no batch. Re-anchors if due and flushes the buffer.
        """
        if self._pending:
            return
        self._maybe_reanchor()
        if self._buffer:
            self._flush()

    def close(self) -> None:
        if self._file.closed:
            return
        self._flush()
        self._file.close()

    # ===== Record building =====

    def _build_meta(self, *, rank_tag: str, gpu_id: int, device: str) -> MetaRecord:
        device_name, device_uuid = device, None
        if self._use_device_events:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            device_name = props.name
            device_uuid = f"GPU-{props.uuid}"
        return MetaRecord(
            schema_version=TRACE_SCHEMA_VERSION,
            clock=TRACE_CLOCK,
            pid=os.getpid(),
            rank_tag=rank_tag,
            gpu_id=gpu_id,
            device_name=device_name,
            device_uuid=device_uuid,
            enable_overlap=self._enable_overlap,
            time_source="cuda_event" if self._use_device_events else "host",
        )

    def _emit_step(self, *, step_id: int, step: _OpenStep) -> None:
        if self._use_device_events:
            # No-op once the result has been copied out; guards the overlap path.
            step.end_event.synchronize()
            t_start_ns = self._event_to_host_ns(step.start_event)
            t_end_ns = self._event_to_host_ns(step.end_event)
        else:
            t_start_ns = step.host_launch_begin_ns
            t_end_ns = step.host_launch_end_ns

        owners = [msgspec.structs.replace(o, step_id=step_id) for o in step.owners]
        prefill = [o for o in owners if o.phase == "prefill"]
        decode = [o for o in owners if o.phase == "decode"]
        self._buffer.append(
            StepRecord(
                step_id=step_id,
                forward_mode=step.forward_mode,
                batch_size=step.batch_size,
                t_start_ns=t_start_ns,
                t_end_ns=t_end_ns,
                host_launch_begin_ns=step.host_launch_begin_ns,
                host_launch_end_ns=step.host_launch_end_ns,
                host_result_ns=trace_clock_ns(),
                anchor_id=self._anchor_id,
                num_prefill_reqs=len(prefill),
                num_decode_reqs=len(decode),
                prefill_tokens=sum(o.selected_tokens for o in prefill),
                decode_tokens=sum(o.selected_tokens for o in decode),
                cached_tokens=sum(o.cached_tokens for o in prefill),
                kv_read_tokens=sum(o.context_len - o.selected_tokens for o in owners),
                overlap_enabled=self._enable_overlap,
                owners_traced=step.owners_traced,
            )
        )
        self._buffer.extend(owners)

    # ===== Clock anchoring =====

    def _record_event(self) -> Optional[torch.cuda.Event]:
        if not self._use_device_events:
            return None
        event = torch.cuda.Event(enable_timing=True)
        # Non-overlap forwards run on the scheduler's current stream.
        event.record(self._forward_stream if self._enable_overlap else None)
        return event

    def _event_to_host_ns(self, event: torch.cuda.Event) -> int:
        # elapsed_time is in ms with ~0.5 us resolution.
        return self._anchor_host_ns + round(
            self._anchor_event.elapsed_time(event) * 1e6
        )

    def _maybe_reanchor(self) -> None:
        if trace_clock_ns() - self._anchor_host_ns >= self._reanchor_interval_ns:
            self._take_anchor()

    def _take_anchor(self) -> None:
        if not self._use_device_events:
            self._anchor_id += 1
            self._anchor_host_ns = trace_clock_ns()
            self._buffer.append(
                AnchorRecord(
                    anchor_id=self._anchor_id,
                    host_ns=self._anchor_host_ns,
                    uncertainty_ns=0,
                )
            )
            return
        # Drain first so the event completes as soon as it is enqueued.
        torch.cuda.synchronize()
        event = torch.cuda.Event(enable_timing=True)
        before_ns = trace_clock_ns()
        event.record()
        event.synchronize()
        after_ns = trace_clock_ns()
        self._anchor_id += 1
        self._anchor_event = event
        self._anchor_host_ns = (before_ns + after_ns) // 2
        self._buffer.append(
            AnchorRecord(
                anchor_id=self._anchor_id,
                host_ns=self._anchor_host_ns,
                uncertainty_ns=(after_ns - before_ns + 1) // 2,
            )
        )

    # ===== Output =====

    def _flush(self) -> None:
        if not self._buffer:
            return
        self._file.write(self._encoder.encode_lines(self._buffer))
        self._file.flush()
        self._buffer = []
        self._steps_since_flush = 0


def _snapshot_owners(batch: ScheduleBatch) -> tuple[List[OwnerRecord], bool]:
    """
        This method answers, per step: which requests are in this forward pass, and how much work does each bring?
        It returns the owner rows plus a flag `owners_traced` saying whether they can be trusted.

        The owner records tell you who was in the step and in what proportion, which is what's needed to split that energy across requests.

        How it decides. It looks at the batch type and reads the right fields for it:
            + `DECODE` => every request generates one token. selected_tokens = 1, context_len = sequence length incl. this step's input token.
            + `EXTEND` => prefill. selected_tokens = tokens computed now, cached_tokens = KV reused, context_len = their sum
            + `MIXED` =>  prefill with decoding requests riding along. As `EXTEND`; requests in `batch.decoding_reqs` are labelled decode with `cached_tokens = 0`
            + anything else (Speculative decoding, IDLE, PREBUILT, SPLIT_PREFILL, DLLM_EXTEND) => No owner rows, `owners_traced: false`

        This method reads only CPU-side data. It never waits on the GPU, so it doesn't slow down the step it's measuring.

    Args:
        batch (ScheduleBatch): the batch of this forward pass

    Returns:
        tuple[List[OwnerRecord], bool]: A list of owner records (one per request), and a flag whether this list can be trusted.
    """
    mode = batch.forward_mode
    # Speculative decoding is on! In that case the owners are unknown.
    if not batch.spec_algorithm.is_none():
        return [], False
    # The whole batch is a pure decode step: every request in it generates exactly one token
    if mode.is_decode():
        return _snapshot_decode_owners(batch), True
    # Decoding paired with prefill (MIXED) or prefill (EXTEND => Compute KV cache)
    if mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
        return _snapshot_extend_owners(batch), True
    return [], False


def _snapshot_decode_owners(batch: ScheduleBatch) -> List[OwnerRecord]:
    num_reqs = len(batch.reqs)
    # prepare_for_decode already counted this step's input token (we do not have to add +1!); 
    # rows past num_reqs are a beam tail (best n output candidates), not requests.
    if batch.seq_lens_cpu is not None:
        # read the current sequence length of each request (drop beam-search tail rows before)
        seq_lens = batch.seq_lens_cpu[:num_reqs].tolist()
    else:
        # Fallback, if the info is not available on the CPU (e.g. speculative decoding)
        # This should never be called if we filter these unknowns
        # req.seqlen lags behind seq_lens_cpu in overlap mode
        seq_lens = [req.seqlen for req in batch.reqs]
    return [
        _owner_record(
            req=req,
            phase="decode",
            selected_tokens=1,
            context_len=int(seq_len),
            cached_tokens=0,
            is_chunked=False,
        )
        for req, seq_len in zip(batch.reqs, seq_lens, strict=True)
    ]


def _snapshot_extend_owners(batch: ScheduleBatch) -> List[OwnerRecord]:
    """
        Note: when SGLang builds a prefill batch, it already computes two lists aligned with batch.reqs:
            + `batch.prefix_lens[i]`: how many tokens of request i already have KV in the cache (from radix-cache hits or earlier chunks).
            + `batch.extend_lens[i]`: how many new tokens request i computes in this step.

        For a `MIXED` batch, `mix_with_running()` appends the decoding requests to the end of both lists, 
        with `prefix_len = seq_len - 1` and `extend_len = 1`. That's exactly what a decode step does, written in prefill terms. It also stores those requests in batch.decoding_reqs.
    """
    # Mixed batches (and decode batches converted to extend) carry their
    # decode requests in decoding_reqs, appended after the prefill rows.
    decode_ids = {id(req) for req in batch.decoding_reqs or []}
    owners = []
    for req, prefix_len, extend_len in zip(
        batch.reqs, batch.prefix_lens, batch.extend_lens, strict=True
    ):
        is_decode = id(req) in decode_ids
        owners.append(
            _owner_record(
                req=req,
                phase="decode" if is_decode else "prefill",
                selected_tokens=extend_len,
                context_len=prefix_len + extend_len,
                cached_tokens=0 if is_decode else prefix_len,
                # batch.chunked_req is the request whose prefill continues next step.
                is_chunked=not is_decode and req is batch.chunked_req,
            )
        )
    return owners


def _owner_record(
    *,
    req,
    phase: str,
    selected_tokens: int,
    context_len: int,
    cached_tokens: int,
    is_chunked: bool,
) -> OwnerRecord:
    return OwnerRecord(
        # Filled in on_result, once run_batch has assigned forward_iter.
        step_id=-1,
        rid=req.rid,
        phase=phase,
        selected_tokens=selected_tokens,
        context_len=context_len,
        cached_tokens=cached_tokens,
        req_cached_tokens=req.cached_tokens,
        is_chunked=is_chunked,
        retraction_count=req.retraction_count,
    )


def create_step_tracer(
    *,
    gpu_id: int,
    device: str,
    enable_overlap: bool,
    forward_stream: Optional[torch.cuda.Stream],
) -> Union[SchedulerStepTracer, NoopStepTracer]:
    if not envs.SGLANG_ENABLE_STEP_TRACE.get():
        return NoopStepTracer()
    parallel = get_parallel()
    return SchedulerStepTracer(
        output_dir=envs.SGLANG_STEP_TRACE_DIR.get(),
        rank_tag=f"dp{parallel.dp_rank or 0}_tp{parallel.tp_rank}_pp{parallel.pp_rank}",
        gpu_id=gpu_id,
        device=device,
        enable_overlap=enable_overlap,
        forward_stream=forward_stream,
        flush_steps=envs.SGLANG_STEP_TRACE_FLUSH_STEPS.get(),
        reanchor_interval_s=envs.SGLANG_STEP_TRACE_REANCHOR_INTERVAL_S.get(),
    )
