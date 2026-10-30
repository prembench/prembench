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
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


@dataclass
class MockConfig:
    first_token_delay_s: float = 0.05
    inter_token_delay_s: float = 0.01
    tokens: int = 8  # max_tokens 가 더 작으면 그쪽을 따른다
    fail_every: int = 0  # n 번째 요청마다 500. 0 이면 실패 없음
    send_usage: bool = True


@dataclass
class MockState:
    requests: int = 0
    in_flight: int = 0
    max_in_flight: int = 0
    bodies: list[dict] = field(default_factory=list)


def make_app(cfg: MockConfig, state: MockState) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        state.requests += 1
        state.bodies.append(body)
        if cfg.fail_every and state.requests % cfg.fail_every == 0:
            return JSONResponse({"error": {"message": "mock failure"}}, status_code=500)

        n = min(cfg.tokens, body.get("max_tokens") or cfg.tokens)

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
                    chunk = {**base, "choices": [{"index": 0, "delta": {"content": f"t{i} "}}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}]})}\n\n"
                if cfg.send_usage and (body.get("stream_options") or {}).get("include_usage"):
                    usage = {"prompt_tokens": 5, "completion_tokens": n, "total_tokens": 5 + n}
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
