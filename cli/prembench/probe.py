"""하드웨어·엔진 자동 감지.

- GPU: nvidia-smi 가 1차 소스. torch 는 설치돼 있을 때만 CUDA 런타임 버전 보강용 (의존성 아님).
- 엔진: OpenAI-compatible `/v1/models` + 엔진별 버전 엔드포인트.
- 감지 못 한 값은 None. 추측해서 채우지 않는다.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import httpx

from prembench.schema import EngineName, Hardware, Interconnect

_TIMEOUT_S = 10


# ---------------------------------------------------------------- GPU


@dataclass
class Gpu:
    name: str
    vram_gb: float


@dataclass
class HostProbe:
    gpus: list[Gpu]
    driver_version: str | None
    cuda_version: str | None
    cuda_version_source: str | None  # "torch" (런타임) | "nvidia-smi" (드라이버 지원 최대)
    interconnect: Interconnect | None
    cpu_model: str | None
    ram_gb: float | None
    no_gpu_reason: str | None = None

    def to_hardware(self) -> Hardware:
        if not self.gpus:
            raise ValueError(f"no GPU: {self.no_gpu_reason}")
        names = {g.name for g in self.gpus}
        if len(names) > 1:
            raise ValueError(f"서로 다른 GPU 혼재는 레코드 1건으로 표현 불가: {sorted(names)}")
        return Hardware(
            gpu_model=self.gpus[0].name,
            gpu_count=len(self.gpus),
            gpu_vram_gb=self.gpus[0].vram_gb,
            interconnect=self.interconnect or "none",
            cpu_model=self.cpu_model,
            ram_gb=self.ram_gb,
            driver_version=self.driver_version,
            cuda_version=self.cuda_version,
            # 아래는 아직 probe 하지 않음 → null
            gpu_power_limit_w=None,
            gpu_clock_locked=None,
            pcie_gen=None,
            pcie_lanes=None,
            numa_nodes=None,
        )


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_S, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def parse_gpu_csv(text: str) -> tuple[list[Gpu], str | None]:
    """`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits` 파싱."""
    gpus: list[Gpu] = []
    driver = None
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        name, mem_mib, driver = parts
        gpus.append(Gpu(name=name, vram_gb=round(float(mem_mib) / 1024, 1)))
    return gpus, driver


def parse_smi_cuda(text: str) -> str | None:
    """`nvidia-smi` 헤더의 'CUDA Version: 12.4' — 드라이버가 지원하는 최대 버전."""
    m = re.search(r"CUDA Version:\s*([\d.]+)", text)
    return m.group(1) if m else None


def parse_topo(text: str) -> Interconnect | None:
    """`nvidia-smi topo -m` 행렬에 NV# 가 있으면 NVLink, 없으면 PCIe."""
    rows = [ln for ln in text.splitlines() if ln.startswith("GPU")]
    if not rows:
        return None
    return "NVLink" if any(re.search(r"\bNV\d+\b", r) for r in rows) else "PCIe"


def _torch_cuda() -> str | None:
    if importlib.util.find_spec("torch") is None:
        return None
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return None
    return torch.version.cuda if torch.cuda.is_available() else None


def _cpu_model() -> str | None:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            return None
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        return out.strip() if out else None
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            return None
    return platform.processor() or None


def _ram_gb() -> float | None:
    if sys.platform == "win32":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(stat)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None
        total = stat.ullTotalPhys
    else:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            return None
    return round(total / 1024**3, 1)


def probe_host() -> HostProbe:
    cpu, ram = _cpu_model(), _ram_gb()
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return HostProbe([], None, None, None, None, cpu, ram, no_gpu_reason="nvidia-smi 없음")

    csv = _run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    gpus, driver = parse_gpu_csv(csv) if csv else ([], None)
    if not gpus:
        return HostProbe(
            [], driver, None, None, None, cpu, ram, no_gpu_reason="nvidia-smi 가 GPU 를 보고하지 않음"
        )

    cuda, src = _torch_cuda(), "torch"
    if cuda is None:
        header = _run([smi])
        cuda, src = (parse_smi_cuda(header) if header else None), "nvidia-smi"
    if cuda is None:
        src = None

    if len(gpus) == 1:
        interconnect: Interconnect | None = "none"
    else:
        topo = _run([smi, "topo", "-m"])
        interconnect = parse_topo(topo) if topo else None

    return HostProbe(gpus, driver, cuda, src, interconnect, cpu, ram)


def parse_temps(text: str) -> float | None:
    """`nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits` → 여러 장이면 최댓값."""
    temps = []
    for line in text.splitlines():
        try:
            temps.append(float(line.strip()))
        except ValueError:
            continue
    return max(temps) if temps else None


def gpu_temperature_c() -> float | None:
    """현재 GPU 온도(°C). nvidia-smi 가 없거나 읽지 못하면 None."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    out = _run([smi, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
    return parse_temps(out) if out else None


# ---------------------------------------------------------------- 엔진


@dataclass
class EngineProbe:
    url: str
    reachable: bool
    name: EngineName | None = None
    version: str | None = None
    models: list[str] = field(default_factory=list)
    max_model_len: int | None = None  # vLLM 만 /v1/models 에 노출
    error: str | None = None


_OWNED_BY = {"vllm": "vllm", "sglang": "sglang"}


def _root(url: str) -> str:
    url = url.rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def _get_json(client: httpx.Client, url: str) -> dict | None:
    try:
        r = client.get(url)
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _identify(client: httpx.Client, root: str, hint: str | None) -> tuple[EngineName | None, str | None]:
    """엔진별 고유 엔드포인트로 이름·버전 확정. owned_by 힌트에 해당하는 엔진부터 시도."""
    checks: list[tuple[EngineName, str, str]] = [
        ("vllm", "/version", "version"),
        ("sglang", "/get_server_info", "version"),
        ("ollama", "/api/version", "version"),
        ("tgi", "/info", "version"),
    ]
    checks.sort(key=lambda c: c[0] != hint)
    for name, path, key in checks:
        data = _get_json(client, root + path)
        if data is None or key not in data:
            continue
        # vLLM /version 은 {"version"} 하나뿐이라 다른 엔진과 구분이 약함 → TGI 는 model_id 로 확인
        if name == "tgi" and "model_id" not in data:
            continue
        return name, str(data[key])
    return None, None


def probe_engine(url: str, client: httpx.Client | None = None) -> EngineProbe:
    own = client is None
    client = client or httpx.Client(timeout=_TIMEOUT_S)
    try:
        base = url.rstrip("/")
        try:
            r = client.get(base + "/models")
            r.raise_for_status()
            payload = r.json()
        except (httpx.HTTPError, ValueError) as e:
            return EngineProbe(url=url, reachable=False, error=f"{base}/models: {e}")

        entries = payload.get("data", []) if isinstance(payload, dict) else []
        models = [e["id"] for e in entries if isinstance(e, dict) and "id" in e]
        owners = {e.get("owned_by") for e in entries if isinstance(e, dict)}
        hint = next((_OWNED_BY[o] for o in owners if o in _OWNED_BY), None)
        lens = [e["max_model_len"] for e in entries if isinstance(e, dict) and "max_model_len" in e]

        name, version = _identify(client, _root(url), hint)
        return EngineProbe(
            url=url,
            reachable=True,
            name=name,
            version=version,
            models=models,
            max_model_len=lens[0] if lens else None,
        )
    finally:
        if own:
            client.close()


def as_dict(host: HostProbe, engine: EngineProbe | None) -> dict:
    return {"host": asdict(host), "engine": asdict(engine) if engine else None}


# ---------------------------------------------------------------- Apple Silicon


def apple_hardware() -> Hardware | None:
    """Apple Silicon 내장 GPU. NVIDIA 가 없을 때만 쓴다 (개발용 맥북 측정).

    gpu_vram_gb 는 통합 메모리 전체 크기다 — CPU 와 공유하므로 전용 VRAM 과 다르다.
    """
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return None
    chip = _cpu_model()
    ram = _ram_gb()
    if not chip or not ram:
        return None
    return Hardware(
        gpu_model=chip, gpu_count=1, gpu_vram_gb=ram, interconnect="none",
        cpu_model=chip, ram_gb=ram, driver_version=None, cuda_version=None,
        gpu_power_limit_w=None, gpu_clock_locked=None, pcie_gen=None, pcie_lanes=None, numa_nodes=None,
    )


# ---------------------------------------------------------------- GPU 샘플링 (nvidia-smi 1초)

_SAMPLE_QUERY = "index,memory.used,utilization.gpu,power.draw"


@dataclass
class GpuSample:
    t_s: float  # time.perf_counter()
    gpu: int
    mem_used_gb: float | None
    util_pct: float | None
    power_w: float | None


def _num(s: str) -> float | None:
    try:
        return float(s.strip())
    except ValueError:
        return None  # "[N/A]" 등


def parse_samples(text: str, t_s: float) -> list[GpuSample]:
    """`nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits`."""
    out = []
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) != 4 or _num(parts[0]) is None:
            continue
        mem = _num(parts[1])
        out.append(GpuSample(t_s, int(_num(parts[0])), mem / 1024 if mem is not None else None,
                             _num(parts[2]), _num(parts[3])))
    return out


class GpuSampler:
    """측정 구간 동안 백그라운드 스레드로 nvidia-smi 를 1초마다 읽는다. nvidia-smi 가 없으면 create() 가 None."""

    def __init__(self, read: Callable[[], str | None], interval_s: float = 1.0) -> None:
        self._read = read
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.samples: list[GpuSample] = []

    @classmethod
    def create(cls) -> GpuSampler | None:
        smi = shutil.which("nvidia-smi")
        if smi is None:
            return None
        return cls(lambda: _run([smi, f"--query-gpu={_SAMPLE_QUERY}", "--format=csv,noheader,nounits"]))

    def _tick(self) -> None:
        t = time.perf_counter()
        text = self._read()
        if text:
            self.samples.extend(parse_samples(text, t))

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._tick()
            self._stop.wait(self._interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[GpuSample]:
        self._stop.set()
        self._thread.join(timeout=10)
        self._tick()  # 종료 시점 1회 더
        return self.samples


# ---------------------------------------------------------------- Ollama 메타데이터


@dataclass
class OllamaModel:
    parameter_count: int | None
    quantization_level: str | None  # 예: "Q4_K_M"
    context_length_max: int | None  # gguf 메타데이터 (모델 최대)
    expert_count: int | None
    template_sha256: str | None
    size_bytes: int | None  # gguf 파일 크기 (/api/tags)


def _post_json(client: httpx.Client, url: str, body: dict) -> dict | None:
    try:
        r = client.post(url, json=body)
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def probe_ollama_model(url: str, model: str, client: httpx.Client | None = None) -> OllamaModel | None:
    own = client is None
    client = client or httpx.Client(timeout=_TIMEOUT_S)
    try:
        root = _root(url)
        show = _post_json(client, root + "/api/show", {"model": model})
        if show is None:
            return None
        info = show.get("model_info") or {}
        arch = info.get("general.architecture")
        template = show.get("template")
        tags = _get_json(client, root + "/api/tags") or {}
        size = next((m.get("size") for m in tags.get("models", []) if model in (m.get("name"), m.get("model"))), None)
        return OllamaModel(
            parameter_count=info.get("general.parameter_count"),
            quantization_level=(show.get("details") or {}).get("quantization_level"),
            context_length_max=info.get(f"{arch}.context_length") if arch else None,
            expert_count=info.get(f"{arch}.expert_count") if arch else None,
            template_sha256=hashlib.sha256(template.encode()).hexdigest() if template else None,
            size_bytes=size,
        )
    finally:
        if own:
            client.close()


def ollama_loaded(url: str, model: str, client: httpx.Client | None = None) -> dict | None:
    """/api/ps 의 해당 모델 항목 (적재돼 있지 않으면 None). context_length 는 실제 런너 설정값."""
    own = client is None
    client = client or httpx.Client(timeout=_TIMEOUT_S)
    try:
        ps = _get_json(client, _root(url) + "/api/ps") or {}
        return next((m for m in ps.get("models", []) if model in (m.get("name"), m.get("model"))), None)
    finally:
        if own:
            client.close()
