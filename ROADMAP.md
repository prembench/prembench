# 온프레미스 LLM 실측 DB — 실행 로드맵

> 목표: 2027-01-31 공개. 주말 10개, 예산 100만 원.
> 이 파일은 리포지토리 루트에 두고, Claude Code 세션마다 "ROADMAP.md의 현재 단계를 읽고 시작해" 로 시작한다.

---

## 0. 결정 사항 (바꾸지 말 것)

| 항목 | 결정 | 이유 |
|---|---|---|
| 측정 CLI | Python 3.11+, `typer` + `httpx`(async) | 운영자 대부분 Python. pip 한 줄로 설치. |
| 부하 생성 | 직접 구현 (OpenAI-compatible `/v1/chat/completions` 대상) | vLLM·SGLang·TGI·Ollama 전부 같은 API. 엔진 독립. |
| 1차 엔진 | vLLM | 온프레미스 서빙의 사실상 표준. |
| 2차 엔진 | SGLang, Ollama | 비교용. v1 이후. |
| DB | Supabase (Postgres) | 무료 티어, REST 자동, 인증 내장. |
| 사이트 | Next.js (App Router) + Vercel | 무료, 빠름, SSR로 검색 노출. |
| 원본 보관 | 모든 제출 JSON을 공개 GitHub 리포 `data/` 에도 커밋 | 출처 추적. "이 숫자 어디서 났냐"에 영수증. |
| 라이선스 | CLI: MIT / 데이터: CC BY 4.0 | 벤더가 인용하게 만들려면 열려 있어야 함. |
| 측정 장소 | 개인 클라우드 계정(RunPod/Vast/Lambda) + 외부 제출 | 회사 자원 절대 사용 안 함. |

---

## 1. 측정 설계 (스키마가 제품이다)

### 1.1 레코드 필수 필드 (schema_version 1.1)

```
schema_version ("1.1"), prembench_version, run_id, batch_id (격자 1회 실행 묶음|null),
submitted_at, submitter (handle), provenance ("self" | "community" | "scraped" | "vendor")
environment ("cloud_rented" | "on_prem"), provider (자유 문자열 예: RunPod|Vast|Lambda, on_prem 이면 null)
  # 사이트의 모든 셀에 environment 를 표시한다. "온프레미스에서 쟀다"고 말하지 않는다.
  # 온프레미스는 대상 독자를 뜻하는 말이지 측정 장소를 주장하는 말이 아니다.

hardware:
  gpu_model, gpu_count, gpu_vram_gb, interconnect (PCIe|NVLink|none),
  cpu_model, ram_gb, driver_version, cuda_version,
  gpu_power_limit_w, gpu_clock_locked (bool), pcie_gen, pcie_lanes, numa_nodes

model:
  hf_id, params_b,
  quantization (자유 문자열, 양자화 안 했으면 "none". 정규화는 사이트 몫),
  weight_dtype (bf16|fp16|fp8 … quantization="none" 이면 필수),
  weight_bytes_gb (safetensors 실측), max_position_embeddings,
  moe (bool), active_params_b (<= params_b), chat_template_applied (bool)

engine:
  name (vllm|sglang|ollama|tgi), version, engine_commit_or_image_digest,
  tensor_parallel, pipeline_parallel, data_parallel, expert_parallel,
  # tensor × pipeline × (data or 1) <= gpu_count
  max_model_len, gpu_memory_utilization, kv_cache_dtype, enable_prefix_caching,
  attention_backend, cuda_graphs (bool), chunked_prefill (bool),
  max_num_seqs, max_num_batched_tokens, speculative_decoding ("none"|방식명),
  server_command (서버 시작 명령 원문), extra_args (server_command 중 파싱 안 된 나머지)

load:
  concurrency, input_tokens, output_tokens, request_count, warmup_count,
  # input_tokens = 엔진이 본 전체 프롬프트 토큰 (chat template 포함) = usage.prompt_tokens
  mode (closed_loop|open_loop), arrival_rate_rps, arrival_process (poisson|constant),
  random_seed,
  sampling: {temperature, top_p, repetition_penalty, max_tokens, stop: [str], seed (int|null)}   # null 불가
  reasoning_enabled (bool|null), reasoning_budget_tokens (int|null), ignore_eos (bool|null)
  # closed_loop 이면 arrival_rate_rps·arrival_process 는 null, open_loop 이면 필수
  # reasoning_budget_tokens 는 reasoning_enabled=true 일 때만
  # output_tokens == sampling.max_tokens

conditions:                           # 측정 당시 조건
  measured_from (same_host|remote),
  # remote 면 GPU 호스트를 못 보므로 gpu_mem_peak_gb·gpu_util_avg_pct·power_avg_w·energy_*·gpu_temp_*·
  # colocated_workloads 가 null 인 게 정상
  model_load_state (cold|warm), colocated_workloads (bool|null), shared_tenancy_note,
  gpu_temp_start_c, gpu_temp_end_c (측정 구간 시작·종료, nvidia-smi, 다중 GPU 면 장당 최대),
  chat_template_variant (템플릿 이름 또는 해시)

result:
  ttft_p50_ms, ttft_p95_ms, ttft_p99_ms,
  tpot_p50_ms, tpot_p95_ms            # time per output token
  e2e_p50_ms, e2e_p95_ms, e2e_p99_ms,
  itl_p50_ms, itl_p95_ms, itl_p99_ms  # 백분위는 모두 단조 증가
  output_tps_total, output_tps_per_user (= output_tps_total / concurrency),
  requests_per_sec, error_count, oom (bool),
  gpu_mem_peak_gb (다중 GPU 면 장당 최대), gpu_util_avg_pct (전 GPU 평균),
  power_avg_w (전 GPU 합)             # nvidia-smi 1초 샘플링
  duration_s,
  energy_total_j (전 GPU 합), energy_per_output_token_j,
  output_tokens_actual_mean, token_count_source (engine_usage|client_tokenizer|null),
  reasoning_tokens_mean, content_tokens_mean (엔진이 분리해 주지 않으면 null),
  finish_reason_counts: {stop, length, error, other}   # 합 == request_count, error == error_count
  content_empty_rate (= per_request 의 content_empty 비율),
  prefix_cache_hit_rate (엔진 메트릭, 없으면 null),
  per_request: [{start_s (측정 시작 기준 초), status, error (200자 절단), prompt_tokens,
                 ttft_ms, e2e_ms (>= ttft_ms), tpot_ms, itl_max_ms, output_tokens,
                 finish_reason, reasoning_tokens, content_empty (bool)}]
    # 측정 구간 요청 전부(실패 포함), 길이 = request_count, finish_reason 집계 = finish_reason_counts

repetition: {index (1부터), count, representative (bool)}
  # 같은 셀 반복 측정의 모든 회차를 레코드로 남긴다. 대표 = output_tps_total 의 nearest-rank 중앙값에
  # 해당하는 실제 회차 (3회면 가운데, 2회면 낮은 쪽). 평균 내지 않는다. 사이트 표는 대표 회차만.

# 모든 키는 필수. 재지 못한 값은 null (생략 아님). token_count_source 가 null 이면 토큰 수 필드도 null.
# 경고(SchemaWarning, 거부 아님 — 사이트에서 구분 표시):
#   - reasoning_enabled=true 인데 reasoning_tokens_mean=null ("추론 토큰 미분리")
#   - concurrency > max_num_seqs (초과분은 엔진 대기열)
#   - ignore_eos=true 인데 finish_reason_counts.stop > 0 (엔진이 ignore_eos 를 적용 안 했을 수 있음)
#   - prefix_cache_hit_rate > 0.05 (캐시가 결과를 부풀렸을 수 있음)
```

### 1.2 측정 격자 (v1)

- 동시성: `1, 4, 16, 64`
- 입력/출력 토큰: `(512,128)`, `(2048,512)`, `(8192,512)`, `(32768,512)`
- → GPU×모델 조합당 16셀. OOM 나는 셀은 `oom=true`로 기록 (그게 정보다).

### 1.3 v1 대상

**하드웨어 (대여 가능)**: RTX 4090, L4, L40S, RTX A6000, A100 80GB, H100 80GB
**다중 GPU**: A100×2 (TP=2), L40S×4 (TP=4) — 70B급 전용

**모델 (온프레미스 수요 기준, 한국 맥락 포함)**:
- Qwen 계열 7B / 14B / 32B / 72B (bf16, awq-int4)
- Llama 3.x 8B / 70B
- Gemma 최신 중형
- EXAONE 최신 (LG) — 한국 공공·기업 수요
- Kanana / HyperCLOVA X SEED / A.X 등 국산 공개 모델 중 서빙 가능한 것
- gpt-oss 20B/120B

**엔진**: vLLM 최신 안정 버전 1개로 고정. 버전은 레코드에 박힌다.

### 1.4 측정 규칙
- 워밍업 전에 실제 생성 요청 1건으로 준비 확인. `/v1/models` 200 은 준비 확인이 아니다. 실패하면 측정 시작 안 함.
- 워밍업 요청 N개 후 측정 시작.
- 측정 구간 시작·종료 시 `nvidia-smi` 로 GPU 온도 기록 (없으면 null).
- 입력 프롬프트는 고정 시드 랜덤 토큰 (캐시 효과 제거). 한국어 토큰 비율 옵션 (`--lang ko`).
  - 모델 HF 토크나이저로 본문을 정확히 N 토큰으로 만들고, chat template 오버헤드는 엔진 usage.prompt_tokens 로 실측
    (길이 2개로 재서 다르면 토크나이저 불일치 → 측정 중단). 본문 = input_tokens − 오버헤드.
  - 시드는 레코드에 기록(random_seed). 기본은 실행마다 새로 뽑는다 — 같은 시드를 다시 쓰면 이전 실행의
    프롬프트가 엔진 prefix 캐시에 남아 결과가 부풀려진다. 반복 측정(2·3회째)도 시드를 바꾼다.
  - 측정 후 모든 요청의 usage.prompt_tokens == input_tokens 를 확인. 다르면(엔진 context 한도로 잘림 등) 레코드를 내지 않는다.
- 백분위는 nearest-rank (보간 없음). 지연·처리량은 성공 요청만으로 계산하고 실패는 error_count·finish_reason 으로 남긴다.
- 각 셀 최소 `request_count = concurrency × 5` 또는 60초 중 긴 쪽.
- `nvidia-smi --query-gpu` 1초 샘플링을 백그라운드로.
- 같은 셀 2회 반복, 편차 10% 넘으면 3회째 돌리고 중앙값.
  - 편차 = (max−min)/min, output_tps_total 과 ttft_p95_ms 둘 다 본다. 하나라도 10% 초과 또는 값 없음 → 3회째.
  - 중앙값은 계산하지 않고 nearest-rank 중앙값에 해당하는 실제 회차를 대표로 표시 (repetition.representative).
  - OOM 셀은 반복하지 않는다. 프롬프트 토큰 불일치 회차는 새 시드로 1번만 다시 잰다.
  - 엔진이 죽으면 남은 셀은 재지 않는다 (레코드 없음). 입력+출력 > max_model_len 인 셀은 건너뛴다.
- OOM 감지: 요청 오류 문구 + 엔진 로그(`--engine-log`, 회차 동안 새로 쓰인 줄만).

---

## 2. 리포지토리 구조

```
prembench/                      # 확정 (2026-10-30)
├── ROADMAP.md
├── CLAUDE.md                  # 아래 3절
├── cli/                       # pip 패키지
│   ├── pyproject.toml
│   └── prembench/
│       ├── __main__.py        # typer 앱
│       ├── probe.py           # 하드웨어·엔진 자동 감지
│       ├── load.py            # async 부하 생성
│       ├── prompts.py         # 고정 시드 랜덤 프롬프트 (정확한 토큰 수)
│       ├── metrics.py         # p50/p95 계산 (nearest-rank)
│       ├── cell.py            # 셀 1개: probe → 프롬프트 → 부하 → 지표 → Run 조립
│       ├── config.py          # ~/.prembench/config (TOML)
│       ├── hf.py              # HF Hub 파일·메타데이터 (httpx)
│       ├── schema.py          # pydantic 모델 (1.1)
│       ├── grid.py            # 격자 실행
│       └── submit.py          # POST to API
├── runner/                    # 클라우드 인스턴스용 자동화
│   ├── bootstrap.sh           # 인스턴스 부팅 → vllm 설치 → 모델 캐시 → 격자 실행 → 제출 → 종료
│   └── matrix.yaml            # 이번 주말에 돌릴 조합
├── data/                      # 제출 원본 JSON (git)
│   └── 2026/11/...
├── web/                       # Next.js
│   ├── app/
│   │   ├── page.tsx           # 검색: GPU × 모델 선택 → 격자 테이블
│   │   ├── compare/           # 2~3개 구성 나란히
│   │   ├── r/[run_id]/        # 레코드 상세 + 원본 JSON 링크
│   │   ├── submit/            # 제출 방법 안내
│   │   └── methodology/       # 1절 그대로. 신뢰의 근거.
│   └── lib/supabase.ts
└── supabase/
    ├── migrations/            # runs 테이블, 뷰
    └── functions/submit/      # 토큰 검증 → insert → data/ 에 커밋하는 GitHub API 호출
```

---

## 3. CLAUDE.md (리포 루트에 둘 것)

```
이 프로젝트는 온프레미스 LLM 서빙 실측 DB다. ROADMAP.md 가 단일 진실.
- 측정 스키마는 ROADMAP.md 1.1 과 cli/prembench/schema.py 가 일치해야 한다. 바꾸면 둘 다.
- 회사 코드·데이터·서버 이름을 절대 넣지 않는다. 예시 값은 전부 가상.
- CLI 는 외부 의존성을 최소화한다 (typer, httpx, pydantic, rich 만).
  예외: 선택 extra `prembench[tokenizer]` 의 `tokenizers` (프롬프트 토큰 수 맞추기). 기본 설치에는 넣지 않는다.
- 모든 수치는 "실측"만. 추정·보간·외삽 금지. 값이 없으면 null.
- 커밋 메시지는 영어, 코드 주석은 한국어 가능.
- 테스트: pytest. load.py 는 mock 서버(기본 FastAPI)로 테스트.
```

---

## 4. 주차별 계획 (주말 기준, 평일 저녁은 보너스)

### W1 (11/1~2) — 스키마 + 뼈대
- [ ] 리포 생성, 구조 세팅, CLAUDE.md
- [ ] `schema.py` — 1.1 전부 pydantic으로. JSON Schema 내보내기.
- [ ] `probe.py` — nvidia-smi / torch 로 GPU·드라이버·CUDA 감지, `/v1/models` 로 엔진 감지
- [ ] `methodology.md` 초안 — 1절을 사람 읽을 문서로
- 완료 기준: `prembench probe` 가 맥북에서 "GPU 없음"을 정확히 출력

Claude Code 세션 프롬프트 예:
"ROADMAP.md 1.1 의 필드를 pydantic v2 모델로 만들어. 각 필드에 단위를 docstring 으로. JSON Schema 를 schema/run.schema.json 으로 내보내는 명령도 추가."

### W2 (11/8~9) — 부하 생성기
- [x] `load.py` — asyncio + httpx, 스트리밍 응답에서 첫 토큰 시각·토큰 간격 기록
- [x] 고정 시드 랜덤 프롬프트 생성 (tokenizer로 정확한 토큰 수)
- [x] `metrics.py` — p50/p95/p99, per-user tps
- [x] mock OpenAI 서버로 pytest
- [x] 맥북에서 Ollama 작은 모델 띄워 end-to-end 1셀 성공
- 완료 기준: `prembench run --url http://localhost:11434/v1 --model X --concurrency 4 --in 512 --out 128` 이 JSON 1개를 뱉음

### W3 (11/15~16) — 격자 + 자동화
- [x] `grid.py` — 1.2 격자 순회, OOM 감지(5xx + 엔진 로그), 반복·중앙값
- [ ] `runner/bootstrap.sh` — RunPod 템플릿: vllm 설치 → HF 캐시 볼륨 → `vllm serve` → 격자 → 결과 업로드 → `shutdown`
- [ ] `matrix.yaml` 로 "이번 주말 조합" 선언
- [ ] RunPod 계정, 네트워크 볼륨 생성, 4090 1시간 빌려서 7B 1개 격자 완주
- 완료 기준: 인스턴스 켜고 손 안 대고 16셀 결과 JSON 16개 받음. **첫 실측 레코드.**

### W4 (11/22~23) — 측정 1차 (소형)
- [ ] 4090, L4, L40S, A6000 × 7B/8B/14B 모델 × bf16/awq
- [ ] 결과 `data/` 커밋
- [ ] 측정하면서 스키마에 빠진 필드 발견 → 1.1 업데이트
- [ ] **현장 시드 섭외 시작** — 온프레미스 GPU 서버 가진 지인·연구실·중소 SI·스타트업 목록 10곳, "CLI 한 번만 돌려주세요" 메시지 초안. 회사 서버는 제외.
- 예산 소진 예상: 10~15만 원
- 완료 기준: 레코드 150개+, 현장 섭외 응답 2곳+

### W5 (11/29~30) — DB + API
- [ ] Supabase 프로젝트, `runs` 테이블 (스키마 그대로 + jsonb 원본)
- [ ] `data/` → DB 적재 스크립트
- [ ] `functions/submit` — API 토큰 검증, insert, GitHub API 로 `data/` 커밋
- [ ] `submit.py` — `prembench submit result.json` / `prembench run --submit`
- 완료 기준: CLI 에서 제출 → DB 에 보이고 → GitHub 에 커밋됨

### W6 (12/6~7) — 측정 2차 (중·대형)
- [ ] A100, H100 × 32B/72B/70B × bf16/fp8/awq
- [ ] A100×2 TP=2, L40S×4 TP=4
- [ ] EXAONE, 국산 모델 1~2개
- 예산 소진 예상: 30~40만 원
- 완료 기준: 레코드 400개+

### W7 (12/13~14) — 사이트 v0
- [ ] Next.js 세팅, Supabase 연결
- [ ] `/` — GPU 선택 + 모델 선택 → 동시성×컨텍스트 격자 테이블 (셀 = output tps / TTFT p95, OOM 은 회색)
- [ ] `/r/[run_id]` — 전체 필드 + 원본 JSON 링크
- [ ] `/methodology`
- 디자인은 나중에. 표가 읽히는 게 전부.
- 완료 기준: 배포된 URL 에서 "L40S + Qwen 32B AWQ" 격자가 보임

### W8 (12/20~21) — 사이트 v1
- [ ] `/compare` — 2~3개 구성 나란히
- [ ] 필터: 양자화, 엔진 버전, 제출 출처(self/community)
- [ ] `/submit` — 설치 3줄 + 토큰 발급 (GitHub OAuth 로그인)
- [ ] 제출자 이름 노출 (리더보드는 아직 X — 숫자 경쟁 유도하면 측정이 왜곡됨)
- [ ] OG 이미지, 기본 SEO
- 완료 기준: 민우 아닌 사람이 문서만 보고 제출 성공 (친구 1명에게 테스트)

### W9 (1/3~4) — 다듬기 + 글
- [ ] CLI PyPI 배포 (`pip install prembench`)
- [ ] README 영/한
- [ ] 공개용 글 2편 초안:
  1. "단일 사용자 tok/s 표가 사이징에 쓸모없는 이유 — 동시성 격자로 본 L40S vs A100"
  2. "온프레미스 LLM 사양 결정, 뭘 재야 하나" (methodology 풀어쓴 것)
- [ ] 버그 수정, 측정 재현성 확인 (같은 셀 다른 날 재측정)

### W10 (1/10~11) — 소프트 런치
- [ ] 글 1편 발행 (GeekNews, 개발자 커뮤니티, LinkedIn)
- [ ] 폐쇄망·온프레미스 LLM 하는 사람 10명에게 직접 DM: "측정 CLI 돌려서 올려줄 수 있나요"
- [ ] vLLM 디스코드/깃허브 Discussions 에 소개
- 완료 기준: 외부 제출 1건, **그중 on_prem 레코드 5건+** (공개 시점에 "현장 데이터 0건"이면 안 된다)

### W11~12 (1/17~31) — 공개 런치
- [ ] 글 2편째 발행
- [ ] Show HN / r/LocalLLaMA 포스팅 (영문 README 필수)
- [ ] 들어온 피드백으로 격자·스키마 조정
- 완료 기준: 외부 제출 5건, 주간 방문 500

---

## 5. 판정 기준

| 시점 | 기준 | 미달 시 |
|---|---|---|
| W3 | 자동화 완주 | 수동으로라도 W4 진행, 자동화는 W5로 |
| W8 | 타인 제출 성공 | 제출 UX 재설계 전까지 런치 보류 |
| 2027-04-30 | 외부 제출 10건 | **접는다.** 데이터는 공개 유지, CLI 는 아카이브. |
| 2027-06-30 | 컨설팅 문의 1건 | 1단계 수익 모델 재검토 |

---

## 6. 하지 않을 것 (v1)

- 리더보드·랭킹 (측정 왜곡 유도)
- 추정·보간 (실측만)
- 학습·파인튜닝 벤치마크 (서빙만)
- 클라우드 API (OpenAI 등) 비교 (온프레미스만)
- 품질 평가 (속도·용량만. 품질은 다른 사람들이 잘함)
- 예쁜 디자인 (표가 읽히면 됨)

---

## 7. 첫 Claude Code 세션 시작 프롬프트

```
ROADMAP.md 를 읽어. 지금 W1 이다.
1. 2절 구조대로 디렉토리와 빈 파일을 만들고
2. CLAUDE.md 를 3절 내용으로 생성하고
3. cli/pyproject.toml 을 typer+httpx+pydantic+rich 로 세팅하고
4. 1.1 스키마를 cli/prembench/schema.py 에 pydantic v2 로 구현해.
끝나면 `prembench --help` 가 뜨는지 확인해.
```

> **메모 — 사이트 표시 정책**
> 운영자 기본 뷰에서는 다음 필드를 숨기고, "전체 필드 보기" 토글로 노출한다.
> - `gpu_util_avg_pct`
> - 에너지 3종: `power_avg_w`, `energy_total_j`, `energy_per_output_token_j`
> - `pcie_gen`, `pcie_lanes`, `numa_nodes`, `gpu_clock_locked`
>
> 숨기는 것이지 빼는 것이 아니다. 레코드·원본 JSON 에는 그대로 남는다.
