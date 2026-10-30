import copy

import pytest
from pydantic import ValidationError

from prembench.schema import Run, json_schema

# 전부 가상 값


def _record() -> dict:
    per_request = [
        {"ttft_ms": 120.0, "e2e_ms": 2600.0, "tpot_ms": 19.5, "output_tokens": 128},
        {"ttft_ms": None, "e2e_ms": 30.0, "tpot_ms": None, "output_tokens": None},  # 실패 요청
    ]
    return {
        "run_id": "example-0001",
        "submitted_at": "2026-11-08T12:00:00+09:00",
        "submitter": "example-user",
        "provenance": "self",
        "environment": "cloud_rented",
        "provider": "RunPod",
        "hardware": {
            "gpu_model": "NVIDIA L40S", "gpu_count": 1, "gpu_vram_gb": 45.0, "interconnect": "none",
            "cpu_model": None, "ram_gb": None, "driver_version": "550.54.15", "cuda_version": "12.4",
            "gpu_power_limit_w": 350.0, "gpu_clock_locked": False, "pcie_gen": 4, "pcie_lanes": 16,
            "numa_nodes": None,
        },
        "model": {
            "hf_id": "Qwen/Qwen2.5-7B-Instruct", "params_b": 7.6, "quantization": "bf16",
            "weight_bytes_gb": 15.2, "max_position_embeddings": 32768,
            "moe": False, "active_params_b": 7.6, "chat_template_applied": True,
        },
        "engine": {
            "name": "vllm", "version": "0.0.0", "tensor_parallel": 1, "pipeline_parallel": 1,
            "max_model_len": 8192, "gpu_memory_utilization": 0.9, "kv_cache_dtype": "auto",
            "enable_prefix_caching": False, "extra_args": None,
            "attention_backend": "FLASH_ATTN", "cuda_graphs": True, "chunked_prefill": True,
            "max_num_seqs": 256, "max_num_batched_tokens": 8192, "speculative_decoding": "none",
            "engine_commit_or_image_digest": None,
        },
        "load": {
            "concurrency": 1, "input_tokens": 512, "output_tokens": 128, "request_count": 2,
            "warmup_count": 1, "mode": "closed_loop", "arrival_rate_rps": None,
            "arrival_process": None, "random_seed": 0,
        },
        "result": {
            "ttft_p50_ms": 120.0, "ttft_p95_ms": 120.0, "ttft_p99_ms": 120.0,
            "tpot_p50_ms": 19.5, "tpot_p95_ms": 19.5,
            "e2e_p50_ms": 1315.0, "e2e_p95_ms": 2471.5, "e2e_p99_ms": 2574.3, "itl_p99_ms": 25.0,
            "output_tps_total": 49.0, "output_tps_per_user": 51.0, "requests_per_sec": 0.38,
            "error_count": 1, "oom": False,
            "gpu_mem_peak_gb": None, "gpu_util_avg_pct": None, "power_avg_w": None,
            "duration_s": 2.63, "energy_total_j": None, "energy_per_output_token_j": None,
            "output_tokens_actual_mean": 128.0, "token_count_source": "engine_usage",
            "per_request": per_request,
        },
    }


def _with(path: str, value) -> dict:
    rec = copy.deepcopy(_record())
    *parents, last = path.split(".")
    node = rec
    for p in parents:
        node = node[p]
    node[last] = value
    return rec


def test_valid_record():
    run = Run.model_validate(_record())
    assert run.result.per_request[1].ttft_ms is None


@pytest.mark.parametrize(
    "path",
    [
        "hardware.pcie_gen",
        "model.moe",
        "engine.engine_commit_or_image_digest",
        "load.mode",
        "result.per_request",
        "result.token_count_source",
        "result.itl_p99_ms",
    ],
)
def test_new_fields_are_required_keys(path):
    rec = copy.deepcopy(_record())
    *parents, last = path.split(".")
    node = rec
    for p in parents:
        node = node[p]
    del node[last]
    with pytest.raises(ValidationError):
        Run.model_validate(rec)


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        Run.model_validate(_with("engine.cuda_graph", True))


def test_open_loop_requires_arrival():
    rec = _with("load.mode", "open_loop")
    with pytest.raises(ValidationError, match="open_loop"):
        Run.model_validate(rec)
    rec["load"]["arrival_rate_rps"] = 2.0
    rec["load"]["arrival_process"] = "poisson"
    Run.model_validate(rec)


def test_closed_loop_rejects_arrival_rate():
    with pytest.raises(ValidationError, match="closed_loop"):
        Run.model_validate(_with("load.arrival_rate_rps", 2.0))


def test_e2e_percentiles_monotonic():
    with pytest.raises(ValidationError, match="e2e"):
        Run.model_validate(_with("result.e2e_p95_ms", 10.0))


def test_active_params_le_params():
    with pytest.raises(ValidationError, match="active_params_b"):
        Run.model_validate(_with("model.active_params_b", 100.0))


def test_per_request_length_matches_request_count():
    with pytest.raises(ValidationError, match="per_request"):
        Run.model_validate(_with("load.request_count", 3))


def test_no_token_source_means_no_token_counts():
    with pytest.raises(ValidationError, match="token_count_source"):
        Run.model_validate(_with("result.token_count_source", None))


def test_json_schema_has_new_fields():
    s = json_schema()
    assert "PerRequest" in s["$defs"]
    assert {"mode", "arrival_rate_rps", "arrival_process", "random_seed"} <= set(s["$defs"]["Load"]["required"])
    assert "per_request" in s["$defs"]["Result"]["required"]
