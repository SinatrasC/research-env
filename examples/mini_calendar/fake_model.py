"""A scripted OpenAI-compatible chat-completions endpoint for the mini-calendar demo.

    python fake_model.py --port 8765

It lets `run-subject` produce real verifiers traces locally, with no paid endpoint. The requested
model name picks a policy:

  careful  books the slot the prompt names                      (solves every valid task)
  sloppy   lists the slots, then books the first free one        (fails when that is not the slot asked for)

Both policies call the task's own MCP tools through verifiers' interception server, so the reward
comes from the real tool state, exactly as it would for a real model. Not a model: a test subject.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from uuid import uuid4

PROMPT = re.compile(r"Book the (\d\d:\d\d) slot for (\w+)")
POLICIES = ("careful", "sloppy")


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def tool_name(tools: list[dict], suffix: str) -> str:
    for tool in tools or []:
        name = (tool.get("function") or {}).get("name") or tool.get("name") or ""
        if name.endswith(suffix):
            return name
    raise ValueError(
        f"no tool ending in {suffix!r} offered; got {[t.get('function', t).get('name') for t in tools or []]}"
    )


def first_free(listing: str) -> str | None:
    try:
        slots = json.loads(listing)
        if isinstance(slots, dict):
            free = [slot for slot, holder in slots.items() if holder is None]
            return sorted(free)[0] if free else None
    except json.JSONDecodeError:
        pass
    free = re.findall(r'"(\d\d:\d\d)"\s*:\s*null', listing)
    return sorted(free)[0] if free else None


def decide(body: dict) -> tuple[str | None, dict | None]:
    """(final text, tool call) for the next assistant turn: exactly one of them is set."""
    policy = body.get("model", "")
    if policy not in POLICIES:
        raise ValueError(f"model must be one of {POLICIES}; got {policy!r}")
    messages = body.get("messages") or []
    tools = body.get("tools") or []
    prompt = next((text_of(m.get("content")) for m in messages if m.get("role") == "user"), "")
    match = PROMPT.search(prompt)
    if not match:
        return "I could not find a booking request.", None
    slot, person = match.groups()
    results = [text_of(m.get("content")) for m in messages if m.get("role") == "tool"]
    if policy == "careful":
        if not results:
            return None, {"name": tool_name(tools, "book_slot"), "arguments": {"slot": slot, "person": person}}
        return "Done.", None
    if not results:
        return None, {"name": tool_name(tools, "list_slots"), "arguments": {}}
    if len(results) == 1:
        free = first_free(results[0])
        if free is None:
            return "No free slot.", None
        return None, {"name": tool_name(tools, "book_slot"), "arguments": {"slot": free, "person": person}}
    return "Done.", None


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            text, call = decide(body)
        except ValueError as exc:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": str(exc)}}).encode())
            return
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if call:
            message["tool_calls"] = [
                {
                    "index": 0,
                    "id": f"call_{uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
                }
            ]
        finish = "tool_calls" if call else "stop"
        prompt_chars = sum(len(text_of(m.get("content"))) for m in body.get("messages") or [])
        usage = {
            "prompt_tokens": max(1, prompt_chars // 4),
            "completion_tokens": 8,
            "total_tokens": max(1, prompt_chars // 4) + 8,
        }
        base = {"id": f"chatcmpl-{uuid4().hex[:10]}", "created": int(time.time()), "model": body.get("model")}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [
                {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": message, "finish_reason": None}],
                },
                {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                },
            ]
            if (body.get("stream_options") or {}).get("include_usage"):
                events.append({**base, "object": "chat.completion.chunk", "choices": [], "usage": usage})
            for event in events:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        if call:
            message["tool_calls"][0].pop("index")
        payload = {
            **base,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage,
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"fake_model: {fmt % args}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
