이 프로젝트는 온프레미스 LLM 서빙 실측 DB다. ROADMAP.md 가 단일 진실.
- 측정 스키마는 ROADMAP.md 1.1 과 cli/prembench/schema.py 가 일치해야 한다. 바꾸면 둘 다.
- 회사 코드·데이터·서버 이름을 절대 넣지 않는다. 예시 값은 전부 가상.
- CLI 는 외부 의존성을 최소화한다 (typer, httpx, pydantic, rich 만).
  예외: 선택 extra `prembench[tokenizer]` 의 `tokenizers` (프롬프트 토큰 수 맞추기). 기본 설치에는 넣지 않는다.
- 모든 수치는 "실측"만. 추정·보간·외삽 금지. 값이 없으면 null.
- 커밋 메시지는 영어, 코드 주석은 한국어 가능.
- 테스트: pytest. load.py 는 mock 서버(기본 FastAPI)로 테스트.
