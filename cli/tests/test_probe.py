import httpx

from prembench import probe
from prembench.probe import HostProbe, parse_gpu_csv, parse_smi_cuda, parse_topo, probe_engine

# 아래 출력은 형식만 맞춘 가상 값


def test_parse_gpu_csv():
    gpus, driver = parse_gpu_csv("NVIDIA L40S, 46068, 550.54.15\nNVIDIA L40S, 46068, 550.54.15\n")
    assert [g.name for g in gpus] == ["NVIDIA L40S", "NVIDIA L40S"]
    assert gpus[0].vram_gb == 45.0
    assert driver == "550.54.15"


def test_parse_smi_cuda():
    assert parse_smi_cuda("| NVIDIA-SMI 550.54.15  Driver Version: 550.54.15  CUDA Version: 12.4 |") == "12.4"
    assert parse_smi_cuda("garbage") is None


def test_parse_topo():
    nvlink = "\tGPU0\tGPU1\tCPU Affinity\nGPU0\t X \tNV12\t0-31\nGPU1\tNV12\t X \t0-31\n"
    pcie = "\tGPU0\tGPU1\nGPU0\t X \tPHB\nGPU1\tPHB\t X \n"
    assert parse_topo(nvlink) == "NVLink"
    assert parse_topo(pcie) == "PCIe"
    assert parse_topo("") is None


def test_no_nvidia_smi(monkeypatch):
    monkeypatch.setattr(probe.shutil, "which", lambda _: None)
    host = probe.probe_host()
    assert host.gpus == []
    assert host.no_gpu_reason == "nvidia-smi 없음"


def test_to_hardware_single_gpu():
    host = HostProbe([probe.Gpu("NVIDIA L4", 22.5)], "550.54.15", "12.4", "nvidia-smi", "none", None, None)
    hw = host.to_hardware()
    assert hw.gpu_count == 1 and hw.interconnect == "none"


def _client(routes: dict[str, dict]) -> httpx.Client:
    def handler(req: httpx.Request) -> httpx.Response:
        body = routes.get(req.url.path)
        return httpx.Response(200, json=body) if body is not None else httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_engine_vllm():
    c = _client({
        "/v1/models": {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct", "owned_by": "vllm", "max_model_len": 8192}]},
        "/version": {"version": "0.11.0"},
    })
    e = probe_engine("http://x/v1", c)
    assert (e.name, e.version, e.max_model_len) == ("vllm", "0.11.0", 8192)


def test_engine_ollama():
    c = _client({
        "/v1/models": {"data": [{"id": "qwen2.5:7b", "owned_by": "library"}]},
        "/api/version": {"version": "0.12.3"},
    })
    e = probe_engine("http://x/v1", c)
    assert (e.name, e.version, e.models) == ("ollama", "0.12.3", ["qwen2.5:7b"])


def test_engine_tgi():
    c = _client({
        "/v1/models": {"data": [{"id": "tgi", "owned_by": "huggingface"}]},
        "/info": {"version": "3.0.0", "model_id": "meta-llama/Llama-3.1-8B-Instruct"},
    })
    assert probe_engine("http://x/v1", c).name == "tgi"


def test_engine_unreachable():
    def boom(req):
        raise httpx.ConnectError("refused")

    e = probe_engine("http://x/v1", httpx.Client(transport=httpx.MockTransport(boom)))
    assert not e.reachable and e.error
