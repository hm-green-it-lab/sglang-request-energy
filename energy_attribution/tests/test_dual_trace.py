"""Native/canonical conformance without importing a serving runtime or CUDA."""

import importlib.util
import json
import sys
from dataclasses import FrozenInstanceError
from enum import Enum
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from request_energy.instrumentation import ExecutionObservation
from request_energy.trace import TraceWriter

ROOT = Path(__file__).resolve().parents[2]


def load(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def exported(tmp_path, monkeypatch):
    class Mode(Enum):
        EXTEND = 1
        MIXED = 2

        def is_decode(self):
            return False

    settings = NS(
        SGLANG_ENABLE_STEP_TRACE=NS(get=lambda: True),
        SGLANG_STEP_TRACE_DIR=NS(get=lambda: str(tmp_path / "native")),
        SGLANG_STEP_TRACE_FLUSH_STEPS=NS(get=lambda: 256),
        SGLANG_STEP_TRACE_REANCHOR_INTERVAL_S=NS(get=lambda: 10),
    )
    for name, fields in {
        "sglang.srt.environ": {"envs": settings},
        "sglang.srt.model_executor.forward_batch_info": {"ForwardMode": Mode},
        "sglang.srt.runtime_context": {
            "get_parallel": lambda: NS(tp_rank=0, dp_rank=0, pp_rank=0)
        },
    }.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("REQUEST_ENERGY_TRACE_DIR", str(tmp_path / "canonical"))
    monkeypatch.setenv("REQUEST_ENERGY_INSTANCE_ID", "fixture")
    monkeypatch.setattr("request_energy.clock.now_ns", lambda: 10**9)
    monkeypatch.setattr("request_energy.clock.boot_id", lambda: "fixture")

    class Timer:
        device_name = "fixture"
        device_id = "GPU-fixture"
        anchor_id = 0
        anchor_host_ns = 1000
        anchor_uncertainty_ns = 3

        def __init__(self, *args, **kwargs):
            self.observer = kwargs["on_observation"]
            self.pending, self.resolved = {}, []

        def begin(self, key, owners, **kwargs):
            self.pending[key] = {
                "kind": "step",
                "engine_instance_id": "fixture",
                "device_id": self.device_id,
                "execution_id": key,
                "timing_source": "synthetic",
                "anchor_id": 0,
                "uncertainty_ns": 3,
                "logical_step_id": kwargs["logical_step_id"],
                "owners_traced": kwargs["supported"],
                "owners": json.loads(json.dumps(owners)),
                "work_kind": kwargs["work_kind"],
            }

        def end(self, key, **kwargs):
            pass

        def resolve(self, key, start, end):
            record = {**self.pending.pop(key), "start_ns": start, "end_ns": end}
            self.resolved.append(record)
            self.observer(ExecutionObservation(json.dumps(record)))

        def health(self):
            return {"error": None, "pending": len(self.pending)}

        def close(self):
            pass

    monkeypatch.setattr("request_energy.instrumentation.CudaStepTracer", Timer)
    adapter = load(
        "energy_fixture_sglang",
        ROOT / "python/sglang/srt/managers/scheduler_components/step_tracer.py",
        monkeypatch,
    )
    tracer = adapter.SchedulerStepTracer(
        gpu_id=0, enable_overlap=True, forward_stream=None, supported_profile=True
    )
    req = NS(
        rid="request-a",
        beam_group=None,
        multimodal_inputs=None,
        cached_tokens=4,
        retraction_count=0,
        finished=lambda: True,
        finished_reason="length",
    )
    batch = NS(
        forward_iter=1,
        forward_mode=Mode.EXTEND,
        reqs=[req],
        spec_algorithm=NS(is_none=lambda: True),
        beam_tail=None,
        prefix_lens=[4],
        extend_lens=[1],
        decoding_reqs=[],
        chunked_req=None,
    )
    tracer.prepare(batch)
    snapshot = tracer.prepared[1]["snapshot"]
    with pytest.raises(FrozenInstanceError):
        snapshot.owners[0].selected_tokens = 999
    req.cached_tokens, batch.extend_lens[0] = 999, 999
    tracer.on_launch_begin(batch)
    tracer.on_launch_end(batch)
    tracer.on_sample_begin(batch)
    tracer.on_sample_end(batch)
    tracer.device.resolve("1:forward", 1000, 1100)
    tracer.on_result(batch)
    assert 1 in tracer.prepared  # Result alone cannot finalize pending sampling.
    tracer.device.resolve("1:sampling", 1300, 1400)
    assert not tracer.prepared
    tracer.close()
    canonical = tmp_path / "canonical" / "execution.jsonl"
    writer = TraceWriter(
        canonical,
        {
            "producer_id": "execution",
            "role": "engine_execution",
            "clock": "CLOCK_MONOTONIC_RAW",
            "boot_id": "fixture",
        },
    )
    for record in tracer.device.resolved:
        writer.emit(record)
    writer.close()
    return tracer.path, canonical, tmp_path


def test_native_envelope_and_owners_use_same_forward_sampling_observations(
    exported, monkeypatch
):
    native, canonical, _ = exported
    rows = [json.loads(line) for line in native.read_text().splitlines()]
    assert {r["kind"] for r in rows} == {"meta", "anchor", "step", "owner"}
    step = next(r for r in rows if r["kind"] == "step")
    owner = next(r for r in rows if r["kind"] == "owner")
    assert (step["t_start_ns"], step["t_end_ns"]) == (1000, 1400)
    assert step["kv_read_tokens"] == 4  # Legacy definition, not triangular KV pairs.
    assert owner["selected_tokens"] == 1 and owner["req_cached_tokens"] == 4
    comparator = load(
        "energy_fixture_compare",
        ROOT / "energy_attribution/tools/validate_canonical_trace.py",
        monkeypatch,
    )
    assert comparator.compare(native, [canonical])["shared_executions"] == 2


def test_cross_format_validator_rejects_divergent_timing(exported, monkeypatch):
    native, canonical, _ = exported
    rows = [json.loads(line) for line in canonical.read_text().splitlines()]
    next(r for r in rows if r["kind"] == "step")["end_ns"] += 1
    canonical.write_text("".join(json.dumps(r) + "\n" for r in rows))
    comparator = load(
        "energy_fixture_compare",
        ROOT / "energy_attribution/tools/validate_canonical_trace.py",
        monkeypatch,
    )
    with pytest.raises(ValueError, match="timing observations"):
        comparator.compare(native, [canonical])
