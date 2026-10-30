# 맥북 개발 환경 (Apple Silicon, GPU 없음)

맥북은 **CLI 개발·테스트용**이다. 여기서 나온 수치는 실측 DB 에 넣지 않는다
(NVIDIA GPU 가 없어 `prembench probe` 는 "no GPU" 를 출력하는 게 정상).
Ollama 는 `load.py` 를 실제 OpenAI-compatible 엔진에 대고 end-to-end 로 확인하는 용도.

## 0. 사전 준비

```sh
xcode-select --install          # git, clang (이미 있으면 건너뜀)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install uv
```

## 1. 리포 받기

```sh
git clone https://github.com/prembench/prembench.git
cd prembench                    # ← 이후 모든 명령은 리포 루트에서
```

## 2. Python 3.12 가상환경 (uv)

> **주의: `uv venv` 는 반드시 리포 루트(`prembench/`)에서 실행한다.**
> `cli/` 나 홈 디렉토리 등 다른 곳에서 만들면 `.venv` 가 엉뚱한 위치에 생기고,
> uv 는 현재 디렉토리부터 위로 올라가며 `.venv` 를 찾기 때문에
> 이후 `uv pip install` / `uv run` 이 다른 환경을 집거나 환경을 못 찾고 실패한다.
> 리포 루트의 `.venv` 하나만 쓴다 (`.gitignore` 에 이미 포함).

```sh
pwd                              # .../prembench 인지 확인 (ROADMAP.md 가 보여야 함)
uv python install 3.12
uv venv --python 3.12            # → ./.venv
source .venv/bin/activate
python --version                 # Python 3.12.x
```

잘못된 곳에 만들었다면 그 `.venv` 디렉토리를 지우고 루트에서 다시 만든다.

## 3. CLI 설치 (editable + dev 의존성)

```sh
uv pip install -e "cli[dev]"     # 리포 루트에서. cli/pyproject.toml 을 가리킨다
prembench --help
prembench probe                  # 맥북에서는 "no GPU — NVIDIA GPU 미검출 (nvidia-smi 없음)"
```

`dev` 엑스트라는 pytest, pytest-asyncio, 그리고 `load.py` 테스트용 mock 서버(fastapi, uvicorn)다.
런타임 의존성은 typer, httpx, pydantic, rich 뿐이다.

테스트:

```sh
cd cli && pytest -q && cd ..
```

## 4. 로컬 엔진: Ollama + qwen2.5:1.5b

```sh
brew install ollama              # 또는 https://ollama.com/download 의 macOS 앱
ollama serve                     # 앱으로 설치했으면 이미 떠 있음 (메뉴바 아이콘)
ollama pull qwen2.5:1.5b         # 약 1GB
```

OpenAI-compatible 엔드포인트 확인:

```sh
curl -s http://localhost:11434/v1/models
prembench probe --url http://localhost:11434/v1     # engine: ollama <버전>
```

`load.py` 를 실제 Ollama 에 대고 도는 통합 테스트 (환경변수가 없으면 skip):

```sh
cd cli && PREMBENCH_OLLAMA_URL=http://localhost:11434/v1 pytest -q tests/test_load_ollama.py
```

참고: Ollama 는 Metal(GPU) 로 돌지만 레코드 스키마의 하드웨어는 NVIDIA 기준이라
맥북 결과는 DB 대상이 아니다.

## 5. Claude Code

```sh
curl -fsSL https://claude.ai/install.sh | bash      # 네이티브 설치
# 또는: brew install --cask claude-code
claude --version
```

리포 루트에서 실행한다 (루트의 `CLAUDE.md` 를 읽어야 하므로):

```sh
cd prembench
claude
```

첫 실행 시 `/login` 으로 로그인. 세션 시작 프롬프트는 "ROADMAP.md 의 현재 단계를 읽고 시작해".
