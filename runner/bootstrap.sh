#!/usr/bin/env bash
# 클라우드 GPU 인스턴스: 설치 → 조합(matrix.toml) 실행 → 결과 보관 → 인스턴스 종료.
#
#   bash bootstrap.sh MATRIX.toml               실제 실행. 끝나면(실패해도) 인스턴스를 끈다
#   bash bootstrap.sh MATRIX.toml --dry-run     계획·예상 시간·비용만 (아무것도 띄우지 않음)
#   bash bootstrap.sh MATRIX.toml --no-shutdown 디버깅용: 끝나도 끄지 않음
#
# 환경변수 (기본값)
#   WORKDIR=/workspace        네트워크 볼륨 (RunPod 기본 마운트). HF 캐시·venv·결과가 여기 남는다
#   PREMBENCH_REPO=https://github.com/prembench/prembench.git
#   PREMBENCH_REF=main        설치할 prembench git ref
#   SKIP_INSTALL=0            1 이면 설치 생략 (PATH 의 prembench·엔진을 그대로 씀)
#   PYTHON                    matrix.toml 을 읽을 python (3.11+). 기본: 설치한 venv 의 python
#   SHUTDOWN_CMD              기본: RunPod 이면 `runpodctl stop pod $RUNPOD_POD_ID`, 아니면 `shutdown -h now`
#   UPLOAD_CMD                결과 디렉터리를 마지막 인자로 받는 업로드 명령 (선택). 실패해도 종료는 진행
#   HF_TOKEN                  gated 모델용
#
# 레코드는 셀이 끝날 때마다 $WORKDIR/results/<시각>/records/ 에 쓰이므로, 중간에 꺼져도 끝난 셀은 남는다.

set -euo pipefail

usage() { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

MATRIX=""
DRY_RUN=0
NO_SHUTDOWN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --no-shutdown) NO_SHUTDOWN=1 ;;
    -h|--help) usage ;;
    -*) echo "unknown option: $arg" >&2; usage ;;
    *) MATRIX="$arg" ;;
  esac
done
[[ -n "$MATRIX" && -f "$MATRIX" ]] || { echo "error: matrix 파일이 필요함: ${MATRIX:-<없음>}" >&2; usage; }
MATRIX="$(cd "$(dirname "$MATRIX")" && pwd)/$(basename "$MATRIX")"

WORKDIR="${WORKDIR:-/workspace}"
PREMBENCH_REPO="${PREMBENCH_REPO:-https://github.com/prembench/prembench.git}"
PREMBENCH_REF="${PREMBENCH_REF:-main}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
export HF_HOME="${HF_HOME:-$WORKDIR/hf}"
export PREMBENCH_CACHE="${PREMBENCH_CACHE:-$WORKDIR/prembench-cache}"

log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

toml_get() {  # toml_get <python 식> — m 은 파싱된 matrix
  "$PY" -c 'import sys, tomllib; m = tomllib.load(open(sys.argv[1], "rb")); print(eval(sys.argv[2]))' "$MATRIX" "$1"
}

install_prembench() {
  if [[ "$SKIP_INSTALL" == 1 ]]; then
    PY="${PYTHON:-python3}"
    return
  fi
  command -v uv >/dev/null || { log "uv 설치"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  [[ -d "$WORKDIR/venv" ]] || uv venv --python 3.12 "$WORKDIR/venv"
  # shellcheck disable=SC1091
  source "$WORKDIR/venv/bin/activate"
  log "prembench 설치 ($PREMBENCH_REF)"
  uv pip install --quiet "prembench[tokenizer] @ git+${PREMBENCH_REPO}@${PREMBENCH_REF}#subdirectory=cli"
  PY="${PYTHON:-$WORKDIR/venv/bin/python}"
}

install_engine() {
  [[ "$SKIP_INSTALL" == 1 ]] && return
  local name version
  name="$(toml_get 'm["engine"].get("name", "vllm")')"
  version="$(toml_get 'm["engine"]["version"]')"
  log "엔진 설치: $name==$version"
  uv pip install --quiet "${name}==${version}"
}

# ---------------------------------------------------------------- dry-run: 띄우지 않고 끄지도 않는다

if [[ "$DRY_RUN" == 1 ]]; then
  install_prembench
  exec prembench batch "$MATRIX" --dry-run
fi

# ---------------------------------------------------------------- 실제 실행

OUT="$WORKDIR/results/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/bootstrap.log") 2>&1

if [[ -z "${SHUTDOWN_CMD:-}" ]]; then
  if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null; then
    SHUTDOWN_CMD="runpodctl stop pod $RUNPOD_POD_ID"
  else
    SHUTDOWN_CMD="shutdown -h now"
  fi
fi

BATCH_PID=""
WATCHDOG_PID=""
finish() {
  local code=$?
  trap - EXIT INT TERM
  if [[ -n "$WATCHDOG_PID" ]]; then
    # 감시 서브셸을 먼저 죽인다 (sleep 을 먼저 죽이면 서브셸이 깨어나 이 스크립트에 TERM 을 보낼 수 있다)
    local kids
    kids="$(pgrep -P "$WATCHDOG_PID" || true)"
    kill "$WATCHDOG_PID" 2>/dev/null || true
    # shellcheck disable=SC2086  # PID 여러 개를 단어로 나눠 넘긴다
    [[ -n "$kids" ]] && kill $kids 2>/dev/null || true  # 남은 sleep
  fi
  [[ -s "$OUT/watchdog.log" ]] && cat "$OUT/watchdog.log"
  if [[ -n "$BATCH_PID" ]] && kill -0 "$BATCH_PID" 2>/dev/null; then
    log "batch 중단"
    kill -TERM "$BATCH_PID" 2>/dev/null || true
    pkill -TERM -P "$BATCH_PID" 2>/dev/null || true
  fi
  log "종료 코드 $code — 결과: $OUT"
  if [[ -n "${UPLOAD_CMD:-}" ]]; then
    log "업로드: $UPLOAD_CMD $OUT"
    eval "$UPLOAD_CMD \"\$OUT\"" || log "업로드 실패 (결과는 볼륨에 남아 있음)"
  fi
  if [[ "$NO_SHUTDOWN" == 1 ]]; then
    log "--no-shutdown: 인스턴스를 끄지 않음"
  else
    log "인스턴스 종료: $SHUTDOWN_CMD"
    eval "$SHUTDOWN_CMD" || log "종료 명령 실패 — 수동으로 꺼야 함"
  fi
  exit "$code"
}
trap finish EXIT
trap 'log "신호 수신"; exit 130' INT TERM

log "matrix: $MATRIX  workdir: $WORKDIR"
cp "$MATRIX" "$OUT/matrix.toml"

install_prembench

# 비용 상한: max_hours 가 지나면 이 스크립트를 끝내고(→ finish → 종료), 그래도 안 끝나면 강제 종료
MAX_S="$(toml_get 'max(1, int(-(-float(m["instance"]["max_hours"]) * 3600 // 1)))')"
# 감시자는 stdout 파이프를 물지 않게 로그 파일로만 쓴다 (남은 sleep 이 파이프를 잡아 호출자가 멈추는 것 방지)
(
  sleep "$MAX_S"
  echo "[bootstrap] watchdog: max_hours 초과 — 중단" >>"$OUT/watchdog.log"
  kill -TERM $$ 2>/dev/null || true
  sleep 120
  eval "$SHUTDOWN_CMD"
) </dev/null >>"$OUT/watchdog.log" 2>&1 &
WATCHDOG_PID=$!
log "watchdog: ${MAX_S}초 뒤 강제 종료"

if [[ "$SKIP_INSTALL" != 1 ]]; then
  command -v nvidia-smi >/dev/null || { log "error: nvidia-smi 없음 — GPU 인스턴스가 아님"; exit 1; }
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi
install_engine

prembench batch "$MATRIX" --out-dir "$OUT" &
BATCH_PID=$!
wait "$BATCH_PID"
