"""A filesystem MCP server that gates every tool call through a real
admission decision, using the SDK's own `ServerMiddleware` -- a genuine
pre-execution veto point (its own runner.py names it a "middleware veto"),
not the observational audit-logging the SDK's own middleware story example
shows. There's no existing approve/deny middleware example in this repo;
the only human-in-the-loop pattern (`stories/refund_desk`) uses elicitation
for mid-call parameter confirmation, a different, complementary mechanism.

The gate itself is driven by `tulip-agents` -- a real dependency of this
example only (`pyproject.toml`), not the rest of the SDK -- via
`tulip.models.native.openai.OpenAIModel`, whose `base_url` override is
explicitly documented for vLLM endpoints
(https://tulipagents.ai's own README lists it alongside Azure/Portkey/
LiteLLM/together.ai/fireworks/groq). The backing model here is Clusiana, a
real (currently unreleased) checkpoint trained specifically for this
three-word decision -- but any tulip-compatible chat model works the same
way; swap TULIP_GATE_URL/TULIP_GATE_OPENAI_MODEL for anything else.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_REQUEST

from tulip.core.messages import Message
from tulip.models.native.openai import OpenAIModel

POLICY = (
    "Before executing any action that could destroy data or overwrite an existing "
    "file outside a scratch/temp directory, you must obtain explicit user "
    "confirmation. Reading files never requires confirmation. Writing a NEW file "
    "inside a scratch or temp directory does not require confirmation."
)

SYSTEM_PROMPT = (
    "You are an admission gate for an AI agent. Given a written policy and a proposed "
    "action, decide what the policy requires:\n"
    "  allow          — the policy permits this action to proceed\n"
    "  require_human  — the policy requires a person to approve before it proceeds\n"
    "  deny           — the policy forbids this action\n"
    "Answer with exactly one of those three words and nothing else."
)

VERDICTS = {"allow", "require_human", "deny"}

TULIP_GATE_SSH_HOST = os.environ.get("TULIP_GATE_SSH_HOST")
TULIP_GATE_LOCAL_PORT = int(os.environ.get("TULIP_GATE_LOCAL_PORT", "18010"))
TULIP_GATE_REMOTE_URL = os.environ.get("TULIP_GATE_URL", "http://127.0.0.1:8010")
TULIP_GATE_MODEL = os.environ.get("TULIP_GATE_MODEL", "clusiana-admit-v4")
OPENAI_GATE_MODEL = os.environ.get("TULIP_GATE_OPENAI_MODEL")

_tunnel_process: subprocess.Popen[bytes] | None = None


def _ensure_ssh_tunnel() -> str:
    """Opens (once) a local port-forward to a private model behind SSH, so
    the real tulip-agents OpenAIModel client -- which needs a directly
    reachable base_url -- can talk to it like any other vLLM endpoint.
    Returns the locally-forwarded base URL.
    """
    global _tunnel_process  # noqa: PLW0603
    remote_host_port = TULIP_GATE_REMOTE_URL.split("://", 1)[-1]
    local_url = f"http://127.0.0.1:{TULIP_GATE_LOCAL_PORT}"
    if _tunnel_process is not None and _tunnel_process.poll() is None:
        return local_url
    _tunnel_process = subprocess.Popen(  # noqa: S603
        [
            "ssh", "-N",
            "-L", f"{TULIP_GATE_LOCAL_PORT}:{remote_host_port}",
            "-o", "ConnectTimeout=5",
            TULIP_GATE_SSH_HOST,  # type: ignore[list-item]
        ],
    )
    time.sleep(1.5)  # let the forward come up before the first real request
    return local_url


# Qwen3 (Clusiana's base) defaults to thinking mode on -- without this the
# first generated token is `<think>`, not the verdict. Not a standard
# OpenAI param; forwarded via OpenAIModel's own `extra_body` passthrough,
# which vLLM (unlike the real OpenAI API) understands. Only set on the
# private/self-hosted path -- the real OpenAI API doesn't have this
# parameter and there's no reason to send it there.
_VLLM_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def _gate_model() -> OpenAIModel | None:
    if TULIP_GATE_SSH_HOST:
        base_url = _ensure_ssh_tunnel()
        return OpenAIModel(
            model=TULIP_GATE_MODEL, base_url=f"{base_url}/v1", api_key="unused",
            max_tokens=6, temperature=0, extra_body=_VLLM_EXTRA_BODY,
        )
    if TULIP_GATE_REMOTE_URL and not TULIP_GATE_SSH_HOST and os.environ.get("TULIP_GATE_URL"):
        return OpenAIModel(
            model=TULIP_GATE_MODEL, base_url=f"{TULIP_GATE_REMOTE_URL}/v1", api_key="unused",
            max_tokens=6, temperature=0, extra_body=_VLLM_EXTRA_BODY,
        )
    if OPENAI_GATE_MODEL and os.environ.get("OPENAI_API_KEY"):
        return OpenAIModel(model=OPENAI_GATE_MODEL, api_key=os.environ["OPENAI_API_KEY"], max_tokens=6, temperature=0)
    return None


async def classify(action_text: str) -> tuple[str, str]:
    model = _gate_model()
    if model is None:
        return "require_human", "no admission model configured -- failing closed"
    try:
        response = await model.complete(
            messages=[
                Message.system(SYSTEM_PROMPT),
                Message.user(f"POLICY:\n{POLICY}\n\nPROPOSED ACTION:\n{action_text}\n\nVerdict?"),
            ]
        )
        text = (response.message.content or "").strip()
        predicted = text.split()[0].rstrip(".,:") if text.split() else text
        if predicted not in VERDICTS:
            return "require_human", f"off-schema response: {text!r}"
        return predicted, text
    except Exception as exc:  # noqa: BLE001 -- fail closed, never open
        return "require_human", f"gate unreachable ({exc}) -- failing closed"


async def admission_gate(ctx: ServerRequestContext[Any], call_next: CallNext) -> HandlerResult:
    """The real veto: does NOT call `call_next` when the gate denies, per
    runner.py's own contract ("a middleware that short-circuited without
    call_next is trusted to return its own well-formed result"). Raises
    MCPError instead, the same mechanism `entry.handler` errors already use.
    """
    if ctx.method != "tools/call":
        return await call_next(ctx)

    params = ctx.params or {}
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})
    if tool_name not in ("write_file", "delete_file"):
        return await call_next(ctx)

    action_text = f"{tool_name}({json.dumps(arguments)})"
    verdict, raw = await classify(action_text)
    if verdict == "allow":
        return await call_next(ctx)

    raise MCPError(
        code=INVALID_REQUEST,
        message=f"denied by tulip admission gate: verdict={verdict!r} raw={raw!r}",
    )


server = MCPServer("admission-gate-example", middleware=[admission_gate])


@server.tool()
def read_file(path: str) -> str:
    """Read a file's contents. Never gated -- read-only."""
    return Path(path).expanduser().read_text()


@server.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file. Gated by admission_gate above."""
    resolved = Path(path).expanduser()
    resolved.write_text(content)
    return f"wrote {len(content)} bytes to {path}"


@server.tool()
def delete_file(path: str) -> str:
    """Delete a file. Gated by admission_gate above."""
    Path(path).expanduser().unlink()
    return f"deleted {path}"


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
