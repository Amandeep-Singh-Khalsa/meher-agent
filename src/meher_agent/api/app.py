"""HTTP surface for the Meher Sweets agent.

Four routes and no more: the task spec fixes the `POST /chat` body exactly, so
every diagnostic this layer produces travels in a response header and the
payload keeps the four contracted keys and nothing else.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ..agent.services import Services, get_services
from ..config import get_config
from ..logging_utils import setup_logging
from ..safety.pii import mask_text, mask_value
from ..types import AgentOutcome, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

_SERVICE_NAME = "meher-agent"
_SERVICE_VERSION = "1.0.0"

#: A blank field is rejected with this text; the client needs to know which of
#: the two inputs it forgot.
_BLANK_CONVERSATION_ID = "conversation_id must not be blank"
_BLANK_MESSAGE = "message must not be blank"

#: A turn that dies in the model or a tool still owes the customer an answer.
#: It is written to pass the reply guard on its own: honest, AI-disclosed, no
#: invented number, and it promises the follow-up that a handoff actually gives.
_TURN_FAILED_REPLY = (
    "I am sorry - something went wrong on my side while I was working out that answer, and I "
    "will not guess. I am the AI assistant for Meher Sweets & Namkeen. I have passed this to "
    "our shop team, and they will reply to you by email within one working day."
)

#: `LLMClient.health` is a blocking call carrying its own retry budget, which can
#: outlast any HTTP client's timeout. A liveness poll must not inherit that, so
#: it runs on a worker thread with a deadline. The deadline is generous enough for
#: a warm local model to answer its one-token probe and honest enough to report
#: "unreachable" rather than to hang a probe.
LLM_HEALTH_TTL_S = 30.0
LLM_HEALTH_TIMEOUT_S = 5.0


def _log_level() -> str:
    try:
        return get_config().runtime.log_level
    except Exception:  # pragma: no cover - a bad config must not stop the service
        return "INFO"


_LOG_LEVEL = _log_level()
setup_logging(_LOG_LEVEL)


class LlmHealthProbe:
    """Caches `services.llm.health()` so a polling loop cannot hammer the model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._checked_at = 0.0
        self._reachable = False
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-health")

    def reachable(self, services: Services) -> bool:
        with self._lock:
            if time.monotonic() - self._checked_at < LLM_HEALTH_TTL_S:
                return self._reachable
        reachable = self._probe(services)
        with self._lock:
            self._checked_at = time.monotonic()
            self._reachable = reachable
        return reachable

    def reset(self) -> None:
        with self._lock:
            self._checked_at = 0.0
            self._reachable = False

    def _probe(self, services: Services) -> bool:
        try:
            return bool(self._pool.submit(services.llm.health).result(timeout=LLM_HEALTH_TIMEOUT_S))
        except Exception:
            return False


_llm_health = LlmHealthProbe()


def reset_health_cache() -> None:
    """Drop the cached LLM verdict. Used by tests and by a manual smoke test."""
    _llm_health.reset()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    setup_logging(_LOG_LEVEL)
    application.state.services = get_services()
    yield


app = FastAPI(title="Meher Sweets & Namkeen agent", version=_SERVICE_VERSION, lifespan=lifespan)


def get_services_dep() -> Services:
    """Overridable seam: production reads `app.state`, tests supply a stub."""
    services = getattr(app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=503, detail="the agent service is still starting up")
    return services


ServicesDep = Annotated[Services, Depends(get_services_dep)]

_CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meher Sweets &amp; Namkeen — AI Assistant</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #faf7f2; color: #2d2a26; height: 100vh; display: flex; flex-direction: column; }
  header { background: #8b1e1e; color: #fff; padding: 14px 20px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.1rem; font-weight: 600; }
  header span { font-size: .8rem; opacity: .8; }
  #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 75%; padding: 10px 14px; border-radius: 12px; line-height: 1.5; font-size: .95rem; white-space: pre-wrap; }
  .user { align-self: flex-end; background: #8b1e1e; color: #fff; border-bottom-right-radius: 4px; }
  .bot { align-self: flex-start; background: #fff; border: 1px solid #e8e0d8; border-bottom-left-radius: 4px; }
  .bot .sources { margin-top: 8px; font-size: .75rem; color: #8b1e1e; }
  .bot .actions { margin-top: 6px; font-size: .75rem; color: #666; }
  .bot .handoff { margin-top: 6px; font-size: .75rem; color: #c0392b; font-weight: 600; }
  #input-bar { display: flex; gap: 8px; padding: 14px 20px; background: #fff; border-top: 1px solid #e8e0d8; }
  #input-bar input { flex: 1; padding: 10px 14px; border: 1px solid #d4c9bc; border-radius: 8px; font-size: .95rem; }
  #input-bar button { padding: 10px 20px; background: #8b1e1e; color: #fff; border: none; border-radius: 8px; font-size: .95rem; cursor: pointer; }
  #input-bar button:disabled { opacity: .5; cursor: default; }
  .typing { font-style: italic; color: #999; font-size: .85rem; }
</style>
</head>
<body>
<header><h1>Meher Sweets &amp; Namkeen</h1><span>AI Assistant</span></header>
<div id="chat"></div>
<div id="input-bar">
  <input id="msg" type="text" placeholder="Ask about prices, policies, orders..." autofocus />
  <button id="send" onclick="send()">Send</button>
</div>
<script>
let convId = 'web-' + Math.random().toString(36).slice(2, 10);
const chat = document.getElementById('chat');
const input = document.getElementById('msg');
const btn = document.getElementById('send');

function addMsg(text, who, extra) {
  const d = document.createElement('div');
  d.className = 'msg ' + who;
  d.textContent = text;
  if (extra) d.innerHTML += extra;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return d;
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMsg(text, 'user');
  btn.disabled = true;
  const botMsg = addMsg('', 'bot');
  const typing = document.createElement('div');
  typing.className = 'typing';
  typing.textContent = 'Typing...';
  botMsg.appendChild(typing);

  try {
    const resp = await fetch('/chat/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({conversation_id: convId, message: text})
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let full = '';
    typing.remove();
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {stream: true});
      const lines = buf.split('\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6);
        if (data === '[DONE]') continue;
        const obj = JSON.parse(data);
        if (obj.delta) { full += obj.delta; botMsg.textContent = full; chat.scrollTop = chat.scrollHeight; }
        if (obj.sources) {
          let extra = '';
          if (obj.sources.length) extra += '<div class="sources">Sources: ' + obj.sources.join(', ') + '</div>';
          if (obj.actions && obj.actions.length) extra += '<div class="actions">Actions: ' + obj.actions.map(a => a.type).join(', ') + '</div>';
          if (obj.handoff) extra += '<div class="handoff">Handed off to the shop team</div>';
          botMsg.innerHTML = full + extra;
        }
      }
    }
  } catch (e) {
    typing.remove();
    botMsg.textContent = 'Sorry, something went wrong. Please try again.';
  }
  btn.disabled = false;
  input.focus();
}

input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
</script>
</body>
</html>
"""

_EVAL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meher Sweets — Evaluation Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #faf7f2; color: #2d2a26; padding: 20px; }
  h1 { font-size: 1.3rem; margin-bottom: 16px; }
  .stats { display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
  .stat { background: #fff; border: 1px solid #e8e0d8; border-radius: 8px; padding: 12px 20px; min-width: 120px; }
  .stat .num { font-size: 1.8rem; font-weight: 700; }
  .stat .lbl { font-size: .8rem; color: #888; }
  .pass { color: #27ae60; }
  .fail { color: #e74c3c; }
  #cases { display: flex; flex-direction: column; gap: 6px; max-height: 60vh; overflow-y: auto; }
  .case { display: flex; align-items: center; gap: 10px; padding: 8px 12px; background: #fff; border: 1px solid #e8e0d8; border-radius: 6px; font-size: .85rem; }
  .case .id { font-weight: 600; min-width: 100px; }
  .case .cat { color: #888; min-width: 80px; }
  .case .lat { color: #666; margin-left: auto; }
  .case .verdict { font-weight: 700; }
  .case.pass { border-left: 4px solid #27ae60; }
  .case.fail { border-left: 4px solid #e74c3c; }
  .spark { display: inline-block; width: 60px; height: 16px; background: linear-gradient(90deg, #27ae60 0%, #e74c3c 100%); border-radius: 3px; margin-left: 8px; }
</style>
</head>
<body>
<h1>Evaluation Dashboard</h1>
<div class="stats">
  <div class="stat"><div class="num" id="passed">0</div><div class="lbl">Passed</div></div>
  <div class="stat"><div class="num" id="failed">0</div><div class="lbl">Failed</div></div>
  <div class="stat"><div class="num" id="total">0</div><div class="lbl">Total</div></div>
  <div class="stat"><div class="num" id="p50">—</div><div class="lbl">p50 latency</div></div>
</div>
<div id="cases"></div>
<script>
const cases = document.getElementById('cases');
const passedEl = document.getElementById('passed');
const failedEl = document.getElementById('failed');
const totalEl = document.getElementById('total');
const p50El = document.getElementById('p50');
let passed = 0, failed = 0, latencies = [];

function sparkline(lat) {
  const max = Math.max(...lat, 1);
  return lat.map(l => {
    const h = Math.round((l / max) * 16);
    return `<span style="display:inline-block;width:3px;height:${h}px;background:#8b1e1e;margin:0 1px;vertical-align:bottom"></span>`;
  }).join('');
}

function addCase(data) {
  const div = document.createElement('div');
  div.className = 'case ' + (data.passed ? 'pass' : 'fail');
  const failedChecks = data.checks.filter(c => !c.passed && !c.skipped).map(c => c.name);
  div.innerHTML = `<span class="id">${data.case_id}</span><span class="cat">${data.category}</span><span class="verdict">${data.passed ? 'PASS' : 'FAIL'}</span>${failedChecks.length ? `<span style="color:#e74c3c">${failedChecks.join(', ')}</span>` : ''}<span class="lat">${data.latency_s.toFixed(1)}s</span>`;
  cases.appendChild(div);
  cases.scrollTop = cases.scrollHeight;
  if (data.passed) passed++; else failed++;
  latencies.push(data.latency_s);
  passedEl.textContent = passed;
  failedEl.textContent = failed;
  totalEl.textContent = passed + failed;
  const sorted = [...latencies].sort((a,b) => a-b);
  const p50 = sorted[Math.floor(sorted.length * 0.5)] || 0;
  p50El.textContent = p50.toFixed(1) + 's';
}

async function run() {
  const resp = await fetch('/eval/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({})
  });
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buf += decoder.decode(value, {stream: true});
    const lines = buf.split('\\n');
    buf = lines.pop();
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);
      if (data === '[DONE]') continue;
      const obj = JSON.parse(data);
      if (obj.error) { console.error(obj.error); continue; }
      addCase(obj);
    }
  }
}
run();
</script>
</body>
</html>
"""


def _run_turn(message: str, conversation_id: str, services: Services) -> AgentOutcome:
    """The one call into the agent loop.

    Imported here rather than at module scope, and kept behind a module-level
    name: importing lazily keeps this module importable with nothing behind it,
    and the indirection is what lets a test replace the turn with a stub.
    """
    from ..agent.loop import run_turn

    return run_turn(message, conversation_id, services)


def _model_name(services: Services) -> str:
    model = getattr(getattr(services.config, "llm", None), "model", None)
    if isinstance(model, str) and model:
        return model
    fallback = getattr(getattr(services, "llm", None), "model", None)
    return fallback if isinstance(fallback, str) and fallback else "unknown"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the chat UI."""
    return _CHAT_HTML


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, response: Response, services: ServicesDep) -> ChatResponse:
    conversation_id = payload.conversation_id.strip()
    message = payload.message.strip()
    if not conversation_id:
        raise HTTPException(status_code=422, detail=_BLANK_CONVERSATION_ID)
    if not message:
        raise HTTPException(status_code=422, detail=_BLANK_MESSAGE)

    try:
        outcome = _run_turn(message, conversation_id, services)
    except Exception as exc:
        logger.error(
            "chat turn failed for conversation %s: %s",
            conversation_id,
            mask_text(f"{type(exc).__name__}: {exc}"),
        )
        response.headers["X-Model-Calls"] = "0"
        response.headers["X-Turn-Error"] = "1"
        return ChatResponse(reply=_TURN_FAILED_REPLY, sources=[], actions=[], handoff=True)

    response.headers["X-Model-Calls"] = str(outcome.model_calls)
    response.headers["X-Tool-Errors"] = str(len(outcome.tool_errors))
    return ChatResponse(
        reply=outcome.reply,
        sources=list(outcome.sources),
        actions=[action.to_public() for action in outcome.actions if action.ok],
        handoff=outcome.handoff,
    )


@app.post("/chat/stream")
def chat_stream(payload: ChatRequest, services: ServicesDep) -> StreamingResponse:
    """Stream the reply as Server-Sent Events.

    The turn runs to completion first (the model is called once), then the reply
    is streamed word-by-word so the UI can render it as it arrives. This is a
    showcase convenience, not true token streaming — the LLM client does not
    support it yet.
    """
    conversation_id = payload.conversation_id.strip()
    message = payload.message.strip()
    if not conversation_id:
        raise HTTPException(status_code=422, detail=_BLANK_CONVERSATION_ID)
    if not message:
        raise HTTPException(status_code=422, detail=_BLANK_MESSAGE)

    try:
        outcome = _run_turn(message, conversation_id, services)
    except Exception as exc:
        logger.error(
            "stream turn failed for conversation %s: %s",
            conversation_id,
            mask_text(f"{type(exc).__name__}: {exc}"),
        )
        outcome = None

    def generate():
        if outcome is None:
            yield f"data: {json.dumps({'error': 'turn failed'})}\n\n"
            yield "data: [DONE]\n\n"
            return
        reply = outcome.reply
        words = reply.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'delta': chunk})}\n\n"
        yield f"data: {json.dumps({'sources': list(outcome.sources), 'actions': [a.to_public() for a in outcome.actions if a.ok], 'handoff': outcome.handoff})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/eval/stream")
def eval_stream(
    cases: str = "evals/cases.jsonl",
    repeats: int = 3,
    concurrency: int = 2,
    services: ServicesDep = None,
) -> StreamingResponse:
    """Run the evaluation and stream live per-case results as SSE.

    Unlike a simple progress bar, this streams each case's verdict, failing
    checks, and latency as it completes, so the UI can show a live pass/fail
    breakdown and latency sparkline — not just a percentage.
    """
    import threading

    from pathlib import Path

    from evals.runner import run_cases

    def generate():
        def on_case(result, done, total):
            payload = {
                "case_id": result.case_id,
                "category": result.category,
                "passed": result.passed,
                "checks": [
                    {"name": c.name, "passed": c.passed, "skipped": c.skipped}
                    for c in result.checks
                ],
                "latency_s": sum(result.latencies_s),
                "done": done,
                "of": total,
            }
            yield f"data: {json.dumps(payload)}\n\n"

        try:
            run_cases(
                Path(cases),
                base_url="http://127.0.0.1:8000",
                repeats=repeats,
                out_dir=Path("reports"),
                concurrency=concurrency,
                on_case=on_case,
            )
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")



@app.get("/eval", response_class=HTMLResponse)
def eval_page() -> str:
    return _EVAL_HTML


@app.get("/leads", response_model=list[dict[str, Any]])
def leads(services: ServicesDep) -> list[dict[str, Any]]:
    return [lead.to_public_masked(mask_value) for lead in services.leads.all()]


@app.get("/health", response_model=dict[str, Any])
def health(services: ServicesDep) -> dict[str, Any]:
    return {
        "status": "ok",
        "model": _model_name(services),
        "llm_reachable": _llm_health.reachable(services),
        "max_steps": getattr(services.config.llm, "max_steps", None),
        "leads": len(services.leads.all()),
    }

@app.exception_handler(RequestValidationError)
def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    problems: list[str] = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        problems.append(f"{where or 'body'}: {error.get('msg', 'is invalid')}")
    return JSONResponse(
        status_code=422,
        content={"detail": "; ".join(problems) or "the request body could not be read"},
    )


@app.exception_handler(Exception)
def on_unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    # The traceback is rendered here and masked before it reaches a handler,
    # because a stack frame quoting the failing line is the most likely place for
    # a contact detail to appear in a log.
    rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(
        "unhandled error on %s %s: %s", request.method, request.url.path, mask_text(rendered)
    )
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


def main() -> None:
    import uvicorn

    config = get_config()
    uvicorn.run("meher_agent.api.app:app", host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
