from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import requests

from .util import log


HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length",
}


@dataclass
class CompactionSettings:
    context_tokens: int = 4096
    reserve_output_tokens: int = 768
    safety_tokens: int = 192
    keep_recent_messages: int = 6
    summary_max_tokens: int = 320
    summary_chunk_chars: int = 7000

    @property
    def prompt_budget(self) -> int:
        return max(1024, self.context_tokens - self.reserve_output_tokens - self.safety_tokens)


class BackendClient:
    def __init__(self, port: int):
        self.base = f"http://127.0.0.1:{port}"
        self.session = requests.Session()

    def post_json(self, path: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
        r = self.session.post(self.base + path, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def rendered_token_count(self, messages: list[dict[str, Any]]) -> int:
        try:
            templ = self.post_json("/apply-template", {"messages": messages}, timeout=60.0)
            prompt = str(templ.get("prompt") or "")
            tok = self.post_json("/tokenize", {"content": prompt}, timeout=60.0)
            tokens = tok.get("tokens")
            if isinstance(tokens, list):
                return len(tokens)
            n = tok.get("n_tokens")
            if isinstance(n, int):
                return n
        except Exception:
            pass
        # Conservative fallback for text-only messages if token endpoints fail.
        chars = 0
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, str):
                chars += len(c)
            else:
                chars += len(json.dumps(c, ensure_ascii=False))
        return max(1, chars // 3)

    def summarize(self, text: str, max_tokens: int) -> str:
        payload = {
            "model": "routecache-local",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Compress the supplied earlier conversation into durable working memory. "
                        "Preserve concrete facts, requirements, names, file paths, code identifiers, "
                        "decisions, constraints, numeric values, unresolved tasks, and user preferences. "
                        "Remove repetition and conversational filler. Do not answer the conversation."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0.1,
            "max_tokens": int(max_tokens),
            "stream": False,
            "reasoning_effort": "none",
        }
        data = self.post_json("/v1/chat/completions", payload, timeout=600.0)
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except Exception:
            return ""


def _message_text(m: dict[str, Any]) -> str:
    role = str(m.get("role") or "unknown")
    content = m.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    extras = []
    if m.get("name"):
        extras.append(f"name={m['name']}")
    if m.get("tool_calls"):
        extras.append("tool_calls=" + json.dumps(m["tool_calls"], ensure_ascii=False))
    suffix = (" [" + ", ".join(extras) + "]") if extras else ""
    return f"{role}{suffix}: {content}"


def _split_chunks(messages: list[dict[str, Any]], max_chars: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for m in messages:
        s = _message_text(m)
        if cur and n + len(s) + 2 > max_chars:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += len(s) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def compact_messages(
    messages: list[dict[str, Any]],
    backend: BackendClient,
    settings: CompactionSettings,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compact oldest chat history while preserving the active end of the conversation.

    The exact llama.cpp chat template + tokenizer are used for budget checks. Leading
    system messages and the newest N messages are kept verbatim. Older messages are
    summarized into a compact system memory. If summarization fails, deterministic
    oldest-message dropping is used as a last-resort safety mechanism.
    """
    original_tokens = backend.rendered_token_count(messages)
    if original_tokens <= settings.prompt_budget:
        return messages, {"compacted": False, "tokens_before": original_tokens, "tokens_after": original_tokens}

    leading_system: list[dict[str, Any]] = []
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        leading_system.append(messages[i])
        i += 1

    tail_count = min(settings.keep_recent_messages, max(1, len(messages) - i))
    recent = messages[-tail_count:] if tail_count else []
    old_end = max(i, len(messages) - tail_count)
    old = messages[i:old_end]

    summaries: list[str] = []
    for chunk in _split_chunks(old, settings.summary_chunk_chars):
        try:
            s = backend.summarize(chunk, settings.summary_max_tokens)
        except Exception as e:
            log(f"Context compaction summary chunk failed: {e}")
            s = ""
        if s:
            summaries.append(s)

    if summaries:
        memory = "\n\n".join(summaries)
        if len(summaries) > 1 and len(memory) > settings.summary_chunk_chars:
            try:
                memory2 = backend.summarize(memory, settings.summary_max_tokens)
                if memory2:
                    memory = memory2
            except Exception:
                pass
        compacted = leading_system + [{
            "role": "system",
            "content": "[RouteCache compacted conversation memory]\n" + memory,
        }] + recent
    else:
        compacted = leading_system + recent

    # Guarantee the request fits. Keep the latest user turn unless it alone exceeds
    # context (in which case the upstream server will return the normal explicit error).
    while len(compacted) > len(leading_system) + 1 and backend.rendered_token_count(compacted) > settings.prompt_budget:
        drop_at = len(leading_system)
        # Prefer dropping compacted memory before recent messages if necessary.
        compacted.pop(drop_at)

    after = backend.rendered_token_count(compacted)
    return compacted, {
        "compacted": True,
        "tokens_before": original_tokens,
        "tokens_after": after,
        "messages_before": len(messages),
        "messages_after": len(compacted),
        "budget": settings.prompt_budget,
    }


class ContextProxy:
    def __init__(self, listen_port: int, backend_port: int, settings: CompactionSettings):
        self.listen_port = int(listen_port)
        self.backend_port = int(backend_port)
        self.settings = settings
        self.backend = BackendClient(self.backend_port)
        self.httpd: ThreadingHTTPServer | None = None

    def start(self) -> ThreadingHTTPServer:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self) -> None:
                super().setup()
                # Local token streaming should not wait for TCP packet coalescing.
                try:
                    self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except OSError:
                    pass

            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def _write_http_chunk(self, chunk: bytes) -> None:
                if not chunk:
                    return
                self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

            def _forward(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                path = self.path
                modified = False
                request_streaming = False
                if self.command == "POST" and path.split("?", 1)[0] in {
                    "/v1/chat/completions", "/chat/completions"
                } and body:
                    try:
                        data = json.loads(body.decode("utf-8"))
                        request_streaming = bool(data.get("stream", False))
                        messages = data.get("messages")
                        if isinstance(messages, list) and all(isinstance(x, dict) for x in messages):
                            compacted, meta = compact_messages(messages, proxy.backend, proxy.settings)
                            if meta.get("compacted"):
                                data["messages"] = compacted
                                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                                modified = True
                                log(
                                    "Context compacted automatically: "
                                    f"{meta.get('tokens_before')} -> {meta.get('tokens_after')} tokens; "
                                    f"{meta.get('messages_before')} -> {meta.get('messages_after')} messages."
                                )
                    except Exception as e:
                        log(f"Context compaction skipped for one request: {e}")

                headers = {
                    k: v for k, v in self.headers.items()
                    if k.lower() not in HOP_HEADERS and k.lower() != "host"
                }
                headers["Host"] = f"127.0.0.1:{proxy.backend_port}"
                headers["Connection"] = "close"
                # Keep SSE uncompressed so newline/event boundaries can be forwarded
                # immediately instead of waiting for decompression buffers.
                if request_streaming:
                    headers["Accept-Encoding"] = "identity"
                try:
                    r = requests.request(
                        self.command,
                        f"http://127.0.0.1:{proxy.backend_port}{path}",
                        data=body if self.command not in {"GET", "HEAD"} else None,
                        headers=headers,
                        stream=True,
                        timeout=(10, 3600),
                    )
                    content_type = str(r.headers.get("Content-Type") or "").lower()
                    is_sse = "text/event-stream" in content_type or request_streaming

                    self.send_response(r.status_code)
                    for k, v in r.headers.items():
                        if k.lower() in HOP_HEADERS:
                            continue
                        self.send_header(k, v)
                    if is_sse and self.command != "HEAD":
                        # BaseHTTPRequestHandler does not add transfer framing for us.
                        # Explicit chunked framing lets browsers consume each SSE line
                        # immediately instead of buffering a close-delimited response.
                        self.send_header("Transfer-Encoding", "chunked")
                        self.send_header("Cache-Control", "no-cache, no-transform")
                        self.send_header("X-Accel-Buffering", "no")
                    self.send_header("Connection", "close")
                    if modified:
                        self.send_header("X-RouteCache-Compacted", "1")
                    self.end_headers()

                    if self.command != "HEAD":
                        if is_sse:
                            # llama.cpp streams OpenAI chat responses as SSE. Read a
                            # complete SSE line at a time and flush it immediately. This
                            # avoids the old 8192-byte proxy buffer that made tokens appear
                            # much more slowly than they were actually decoded.
                            while True:
                                line = r.raw.readline()
                                if not line:
                                    break
                                self._write_http_chunk(line)
                            self.wfile.write(b"0\r\n\r\n")
                            self.wfile.flush()
                        else:
                            for chunk in r.iter_content(chunk_size=65536):
                                if chunk:
                                    self.wfile.write(chunk)
                                    self.wfile.flush()
                    self.close_connection = True
                except Exception as e:
                    payload = json.dumps({"error": {"message": f"RouteCache proxy error: {e}"}}).encode("utf-8")
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(payload)
                    self.close_connection = True

            do_GET = _forward
            do_POST = _forward
            do_PUT = _forward
            do_DELETE = _forward
            do_PATCH = _forward
            do_HEAD = _forward
            do_OPTIONS = _forward

        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.listen_port), Handler)
        thread = threading.Thread(target=self.httpd.serve_forever, name="routecache-context-proxy", daemon=True)
        thread.start()
        return self.httpd

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None


def wait_for_health(port: int, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1.5)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(0.4)
    return False


def warm_backend(port: int, rounds: int = 2, tokens: int = 96) -> list[float]:
    """Warm mmap/CPU-offloaded tensors before the first real chat request."""
    out: list[float] = []
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    for i in range(max(0, rounds)):
        payload = {
            "model": "routecache-local",
            "messages": [{"role": "user", "content": "Write a compact technical note about deterministic local inference."}],
            "temperature": 0.0,
            "max_tokens": int(tokens),
            "stream": False,
            "reasoning_effort": "none",
        }
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=payload, timeout=600)
            r.raise_for_status()
            data = r.json()
            elapsed = max(1e-6, time.perf_counter() - t0)
            usage = data.get("usage") or {}
            n = int(usage.get("completion_tokens") or 0)
            if n <= 0:
                # Some llama.cpp OpenAI responses expose native timings instead.
                timings = data.get("timings") or {}
                v = timings.get("predicted_per_second")
                if v:
                    out.append(float(v))
                    log(f"Warm-up {i+1}: {float(v):.2f} tok/s")
                    continue
            tps = n / elapsed if n else 0.0
            out.append(tps)
            log(f"Warm-up {i+1}: {tps:.2f} tok/s ({n} output tokens)")
        except Exception as e:
            log(f"Warm-up {i+1} skipped: {e}")
            break
    return out
