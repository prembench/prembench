import copy
import warnings

import pytest
from pydantic import ValidationError

from prembench.schema import Run, SchemaWarning, json_schema

# 전부 가상 값


def _record() -> dict:
    per_request = [
        {"ttft_ms": 120.0, "e2e_ms": 2600.0, "tpot_ms": 19.5, "output_tokens": 128,
         "finish_reason": "length", "reasoning_tokens": None},
        {"ttft_ms": None, "e2e_ms": 30.0, "tpot_ms": None, "output_tokens": None,
         "finish_reason": "error", "reasoning_tokens": None},  # 실패 요청
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
            "sampling": {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0,
                         "max_tokens": 128, "stop": []},
            "reasoning_enabled": False, "reasoning_budget_tokens": None,
        },
        "conditions": {
            "model_load_state": "warm", "colocated_workloads": False, "shared_tenancy_note": None,
            "gpu_temp_start_c": 41.0, "gpu_temp_end_c": 63.0, "chat_template_variant": None,
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
            "reasoning_tokens_mean": None, "content_tokens_mean": None,
            "finish_reason_counts": {"stop": 0, "length": 1, "error": 1, "other": 0},
            "content_empty_rate": 0.0, "prefix_cache_hit_rate": None,
            "per_request": per_request,
        },
    }


def _locate(rec: dict, path: str):
    """'a.b.0.c' → (부모 노드, 마지막 키). 숫자 조각은 리스트 인덱스."""
    *parents, last = path.split(".")
    node = rec
    for p in parents:
        node = node[int(p)] if isinstance(node, list) else node[p]
    return node, last


def _with(path: str, value) -> dict:
    rec = copy.deepcopy(_record())
    node, last = _locate(rec, path)
    node[last] = value
    return rec


def test_valid_record():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 정상 레코드는 경고도 없어야 한다
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
        "load.sampling",
        "load.sampling.stop",
        "load.reasoning_enabled",
        "conditions",
        "conditions.model_load_state",
        "result.finish_reason_counts",
        "result.content_empty_rate",
        "result.prefix_cache_hit_rate",
        "result.per_request.0.finish_reason",
        "result.per_request.0.reasoning_tokens",
    ],
)
def test_new_fields_are_required_keys(path):
    rec = copy.deepcopy(_record())
    node, last = _locate(rec, path)
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
    assert "conditions" in s["required"]
    assert {"Sampling", "Conditions", "FinishReasonCounts"} <= set(s["$defs"])


def test_sampling_not_nullable():
    with pytest.raises(ValidationError):
        Run.model_validate(_with("load.sampling", None))
    with pytest.raises(ValidationError):
        Run.model_validate(_with("load.sampling.temperature", None))


def test_finish_reason_counts_must_sum_to_request_count():
    rec = _with("result.finish_reason_counts", {"stop": 1, "length": 1, "error": 1, "other": 0})
    with pytest.raises(ValidationError, match="finish_reason_counts 합"):
        Run.model_validate(rec)


def test_finish_reason_counts_must_match_per_request():
    rec = _with("result.finish_reason_counts", {"stop": 1, "length": 0, "error": 1, "other": 0})
    with pytest.raises(ValidationError, match="per_request 집계"):
        Run.model_validate(rec)


def test_finish_reason_error_matches_error_count():
    with pytest.raises(ValidationError, match="error_count"):
        Run.model_validate(_with("result.error_count", 0))


def test_content_empty_rate_range():
    with pytest.raises(ValidationError):
        Run.model_validate(_with("result.content_empty_rate", 1.5))


def test_reasoning_enabled_without_split_warns_not_rejects():
    rec = _with("load.reasoning_enabled", True)
    with pytest.warns(SchemaWarning, match="reasoning_tokens_mean"):
        run = Run.model_validate(rec)
    assert run.load.reasoning_enabled is True


def test_reasoning_enabled_with_split_no_warning():
    rec = _with("load.reasoning_enabled", True)
    rec["result"]["reasoning_tokens_mean"] = 40.0
    rec["result"]["content_tokens_mean"] = 88.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Run.model_validate(rec)


def test_reasoning_budget_requires_reasoning_enabled():
    with pytest.raises(ValidationError, match="reasoning_budget_tokens"):
        Run.model_validate(_with("load.reasoning_budget_tokens", 1024))


def test_conditions_model_load_state_enum():
    with pytest.raises(ValidationError):
        Run.model_validate(_with("conditions.model_load_state", "lukewarm"))
