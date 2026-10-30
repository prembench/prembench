"""load.py 테스트용 mock OpenAI-compatible 서버 (FastAPI + uvicorn, 실제 HTTP).

httpx.ASGITransport 는 스트리밍 응답을 끝까지 모은 뒤 돌려주므로 토큰 간격을 잴 수 없다.
그래서 uvicorn 을 별도 스레드에서 실제 포트로 띄운다.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


@dataclass
class MockConfig:
    first_token_delay_s: float = 0.05
    inter_token_delay_s: float = 0.01
    tokens: int = 8  # 본문 토큰 수. 추론 토큰과 합해 max_tokens 를 넘으면 잘린다
    reasoning_tokens: int = 0  # 본문 앞에 reasoning_content 로 보낼 토큰 수
    fail_requests: frozenset[int] = frozenset()  # 500 을 돌려줄 요청 번호 (1부터, 준비 확인 포함)
    error_message: str = "mock failure"
    send_usage: bool = True
    # 본문 문자열 → usage.prompt_tokens. 엔진의 토크나이저 + chat template 역할
    prompt_tokens: Callable[[str], int] = lambda content: 5
    # Ollama 흉내용 (/v1/models, /api/*)
    models: tuple[str, ...] = ("mock",)
    ollama_version: str = "0.0.0-mock"


@dataclass
class MockState:
    requests: int = 0
    in_flight: int = 0
    max_in_flight: int = 0
    bodies: list[dict] = field(default_factory=list)


def make_app(cfg: MockConfig, state: MockState) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "library"} for m in cfg.models]}

    @app.get("/api/version")
    async def version():
        return {"version": cfg.ollama_version}

    @app.post("/api/show")
    async def show(req: Request):
        name = (await req.json())["model"]
        if name not in cfg.models:
            return JSONResponse({"error": "not found"}, status_code=404)
        return {
            "template": "{{ .Prompt }}",
            "details": {"format": "gguf", "quantization_level": "Q4_K_M"},
            "model_info": {
                "general.architecture": "mockarch",
                "general.parameter_count": 1_500_000_000,
                "mockarch.context_length": 32768,
            },
        }

    @app.get("/api/tags")
    async def tags():
        return {"models": [{"name": m, "model": m, "size": 1_000_000_000} for m in cfg.models]}

    @app.get("/api/ps")
    async def ps():
        # 생성 요청을 한 번이라도 받으면 적재된 것으로 본다
        loaded = cfg.models if state.requests else ()
        return {"models": [{"name": m, "model": m, "context_length": 4096} for m in loaded]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        state.requests += 1
        state.bodies.append(body)
        if state.requests in cfg.fail_requests:
            return JSONResponse({"error": {"message": cfg.error_message}}, status_code=500)

        limit = body.get("max_tokens") or (cfg.reasoning_tokens + cfg.tokens)
        n_reason = min(cfg.reasoning_tokens, limit)
        n_content = min(cfg.tokens, limit - n_reason)
        n = n_reason + n_content
        finish = "length" if cfg.reasoning_tokens + cfg.tokens > limit else "stop"
        p_tok = cfg.prompt_tokens(body["messages"][-1]["content"])

        if not body.get("stream"):
            resp = {
                "id": "mock-1", "object": "chat.completion", "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "t0"},
                             "finish_reason": finish}],
            }
            if cfg.send_usage:
                resp["usage"] = {"prompt_tokens": p_tok, "completion_tokens": n, "total_tokens": p_tok + n}
            return JSONResponse(resp)

        async def gen():
            state.in_flight += 1
            state.max_in_flight = max(state.max_in_flight, state.in_flight)
            try:
                base = {"id": "mock-1", "object": "chat.completion.chunk", "model": body["model"]}
                # 첫 청크는 role 만 (vLLM 과 같은 형태) — 내용 없음, TTFT 에 포함되면 안 된다
                yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}}]})}\n\n"
                await asyncio.sleep(cfg.first_token_delay_s)
                for i in range(n):
                    if i:
                        await asyncio.sleep(cfg.inter_token_delay_s)
                    key = "reasoning_content" if i < n_reason else "content"
                    chunk = {**base, "choices": [{"index": 0, "delta": {key: f"t{i} "}}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]})}\n\n"
                if cfg.send_usage and (body.get("stream_options") or {}).get("include_usage"):
                    usage = {"prompt_tokens": p_tok, "completion_tokens": n, "total_tokens": p_tok + n}
                    if cfg.reasoning_tokens:
                        usage["completion_tokens_details"] = {"reasoning_tokens": n_reason}
                    yield f"data: {json.dumps({**base, 'choices': [], 'usage': usage})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                state.in_flight -= 1

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


class MockServer:
    def __init__(self, cfg: MockConfig | None = None):
        self.cfg = cfg or MockConfig()
        self.state = MockState()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        config = uvicorn.Config(
            make_app(self.cfg, self.state), host="127.0.0.1", port=self.port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> MockServer:
        self._thread.start()
        deadline = time.monotonic() + 5
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("mock server did not start")
            time.sleep(0.01)
        return self

    def __exit__(self, *exc) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)
