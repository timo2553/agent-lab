"""本机 HTTP 桩服务：用真实 socket 替掉真实模型。

和 tests/test_llm_client.py 的分工：
- 那边注入的是假 opener，测"拿到响应之后怎么归类、怎么重试"（纯逻辑，毫秒级）
- 这边 socket 是真的，测"请求真的发出去之后长什么样、超时是不是真的生效"

不联网、不需要 API Key，CI 里照跑。
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass(frozen=True)
class RequestRecord:
    """桩服务收到的一次请求。"""

    path: str
    headers: dict[str, str]
    body: bytes

    @property
    def json_body(self) -> dict:
        return json.loads(self.body.decode("utf-8"))


@dataclass(frozen=True)
class StubReply:
    """桩服务要回什么。别直接构造，用下面的 ok / http_error / raw。"""

    status: int = 200
    body: bytes = b""
    content_type: str = "application/json"
    delay_s: float = 0.0


def ok(payload: dict, *, delay_s: float = 0.0) -> StubReply:
    """200 + JSON 正常返回。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return StubReply(body=body, delay_s=delay_s)


def http_error(status: int, text: str = "stub error", *, delay_s: float = 0.0) -> StubReply:
    """非 2xx：让 client 从状态码去归类（401 / 429 / 503 ...）。"""
    body = json.dumps({"error": text}).encode("utf-8")
    return StubReply(status=status, body=body, delay_s=delay_s)


def raw(body: str | bytes, *, status: int = 200, content_type: str = "text/html") -> StubReply:
    """原样返回：用来造"200 但不是 JSON"这类畸形响应。"""
    data = body.encode("utf-8") if isinstance(body, str) else body
    return StubReply(status=status, body=data, content_type=content_type)


class StubServer:
    """配合 with 使用的本机桩服务。

        with StubServer(lambda record: ok(PAYLOAD)) as stub:
            ...
            stub.base_url     # http://127.0.0.1:<随机端口>/v1
            stub.requests     # 收到的请求列表（RequestRecord）
            stub.call_count    # 一共收到几次

    回调在服务线程里执行，决定这次回什么；想"第几次回不同的东西"就在闭包里自己记数。
    """

    def __init__(self, responder: Callable[[RequestRecord], StubReply]) -> None:
        self._responder = responder
        self.requests: list[RequestRecord] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self))
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def respond(self, record: RequestRecord) -> StubReply:
        return self._responder(record)

    def __enter__(self) -> StubServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def _make_handler(stub: StubServer) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 - 方法名由基类约定
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            record = RequestRecord(
                path=self.path,
                headers={key: value for key, value in self.headers.items()},
                body=body,
            )
            stub.requests.append(record)
            reply = stub.respond(record)
            if reply.delay_s:
                time.sleep(reply.delay_s)
            try:
                self.send_response(reply.status)
                self.send_header("Content-Type", reply.content_type)
                self.send_header("Content-Length", str(len(reply.body)))
                self.end_headers()
                self.wfile.write(reply.body)
            except OSError:
                # 客户端已经超时走人，这里的写入必然失败，属于预期而非故障
                pass

        def log_message(self, *args: object) -> None:
            pass  # 别让桩服务把测试输出刷满

    return _Handler