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

Design notes added in response to real review feedback on the tracking
issue (https://github.com/modelcontextprotocol/python-sdk/issues/3272,
from Ecocitizenz): an admission model's response is a policy *input*, not
proof an action is safe. To make that distinction concrete rather than
just asserted:

  - Every decision is bound to an `args_hash` of the exact arguments it
    was made about (see `DecisionRecord`), not just the tool name.
  - The policy text is versioned (`POLICY_VERSION`); a version bump is
    visible in every subsequent decision record.
  - Failure modes are deterministic and enumerated (`ReasonCode`), not
    freeform strings: an unconfigured gate, an unreachable one, a timeout,
    and an off-schema model response are each their own code, and each
    one escalates (never silently allows).
  - A minimal, secret-redacted decision record is emitted for every call
    -- `request_id`, `tool_id`, `capability_class`, `args_hash`,
    `policy_version`, `model_id`, `decision`, `reason_code`, `issued_at`
    -- logged structurally and kept in `decision_log()` for the tests in
    `tests/test_admission_gate.py` to assert against directly, independent
    of any real model call.

What's still deliberately out of scope for an example this size, disclosed
rather than silently absent:

  - **Replay of a stale approval against different arguments** isn't a
    live risk here structurally, not just by policy: nothing caches or
    reuses a decision across calls, so every `tools/call` gets a fresh
    classification against its own `args_hash`. There is no stored
    approval to replay.
  - **Materially changed tool schemas** and **model/checkpoint changes**
    are not detected or pinned -- the model identity is recorded in every
    `DecisionRecord` (so a checkpoint swap is at least visible after the
    fact), but nothing here enforces re-approval on a schema or checkpoint
    change. A real deployment gating anything higher-stakes than this
    example's toy filesystem tools would need that; it's real, non-trivial
    work this example doesn't attempt.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_REQUEST
from tulip.core.messages import Message
from tulip.models.native.openai import OpenAIModel

logger = logging.getLogger("mcp_admission_gate")

POLICY_VERSION = "policy-v1"

POLICY = (
    "Before executing any action that could destroy data or overwrite an existing "
    "file outside a scratch/temp directory, you must obtain explicit user "
    "confirmation. Reading files never requires confirmation. Writing a NEW file "
    "inside a scratch or temp directory does not require confirmation."
)

# Which capability each gated tool exercises -- recorded on every decision
# so an audit record reads as "filesystem.write was escalated", not just
# "write_file was escalated", the distinction that matters once a server
# exposes more than one tool per capability.
CAPABILITY_CLASS = {
    "read_file": "filesystem.read",
    "write_file": "filesystem.write",
    "delete_file": "filesystem.delete",
}

# How long a single admission call is allowed to take before it counts as
# unreachable. Bounded on purpose -- an unbounded call is itself a
# fail-open risk if the backing model hangs instead of erroring.
GATE_TIMEOUT_SECONDS = float(os.environ.get("TULIP_GATE_TIMEOUT_SECONDS", "10"))

SYSTEM_PROMPT = (
    "You are an admission gate for an AI agent. Given a written policy and a proposed "
    "action, decide what the policy requires:\n"
    "  allow          — the policy permits this action to proceed\n"
    "  require_human  — the policy requires a person to approve before it proceeds\n"
    "  deny           — the policy forbids this action\n"
    "Answer with exactly one of those three words and nothing else."
)

VERDICTS = {"allow", "require_human", "deny"}


class ReasonCode(str, Enum):
    """Why a decision came out the way it did -- a fixed, deterministic set
    instead of a freeform string, so a caller (or a test) can branch on it
    without parsing prose. `POLICY_*` codes came from a real model verdict;
    the others are gate-infrastructure failures, and every one of them
    resolves to `require_human`, never to `allow` -- uncertainty escalates,
    it is never interpreted as permission.
    """

    POLICY_ALLOW = "policy_allow"
    POLICY_DENY = "policy_deny"
    POLICY_ESCALATE = "policy_escalate"
    GATE_UNCONFIGURED = "gate_unconfigured"
    GATE_TIMEOUT = "gate_timeout"
    GATE_UNREACHABLE = "gate_unreachable"
    OFF_SCHEMA_RESPONSE = "off_schema_response"


@dataclasses.dataclass(frozen=True, slots=True)
class DecisionRecord:
    """A minimal, secret-redacted record of one admission decision.

    Deliberately does NOT include the raw tool arguments -- `write_file`'s
    `content` could be arbitrary, possibly sensitive, file data. `args_hash`
    binds the decision to exactly the arguments it was made about (so a
    stored record can be checked against a later claim of "the same call")
    without the record itself carrying that payload.
    """

    request_id: str
    tool_id: str
    capability_class: str
    args_hash: str
    policy_version: str
    model_id: str
    decision: str
    reason_code: ReasonCode
    issued_at: str

    def as_log_dict(self) -> dict[str, str]:
        d = dataclasses.asdict(self)
        d["reason_code"] = self.reason_code.value
        return d


# Every decision this process has made, in order -- kept in memory so
# `tests/test_admission_gate.py` can assert against real decision records
# directly, independent of whether a real admission model is reachable in
# the test environment.
_decision_log: list[DecisionRecord] = []


def decision_log() -> list[DecisionRecord]:
    """The real decisions made so far, oldest first. A copy -- callers
    can't mutate the log through the returned list."""
    return list(_decision_log)


def _args_hash(tool_name: str, arguments: dict[str, Any]) -> str:
    """A stable hash binding a decision to the exact call it was about.

    `sort_keys=True` makes this independent of argument ordering; this is
    a binding/audit hash, not a security boundary on its own -- two
    semantically-different-but-JSON-equal payloads would collide, which is
    fine for "was this the call the decision was made about", not fine as
    a cryptographic commitment to untrusted input.
    """
    canonical = json.dumps({"tool": tool_name, "arguments": arguments}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
            "ssh",
            "-N",
            "-L",
            f"{TULIP_GATE_LOCAL_PORT}:{remote_host_port}",
            "-o",
            "ConnectTimeout=5",
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


def _gate_model() -> tuple[OpenAIModel | None, str | None]:
    """Returns (model, model_id) -- the id is recorded on every decision so
    a checkpoint swap is visible in the audit trail after the fact, even
    though nothing here pins or enforces a particular checkpoint."""
    if TULIP_GATE_SSH_HOST:
        base_url = _ensure_ssh_tunnel()
        return (
            OpenAIModel(
                model=TULIP_GATE_MODEL,
                base_url=f"{base_url}/v1",
                api_key="unused",
                max_tokens=6,
                temperature=0,
                extra_body=_VLLM_EXTRA_BODY,
            ),
            TULIP_GATE_MODEL,
        )
    if TULIP_GATE_REMOTE_URL and not TULIP_GATE_SSH_HOST and os.environ.get("TULIP_GATE_URL"):
        return (
            OpenAIModel(
                model=TULIP_GATE_MODEL,
                base_url=f"{TULIP_GATE_REMOTE_URL}/v1",
                api_key="unused",
                max_tokens=6,
                temperature=0,
                extra_body=_VLLM_EXTRA_BODY,
            ),
            TULIP_GATE_MODEL,
        )
    if OPENAI_GATE_MODEL and os.environ.get("OPENAI_API_KEY"):
        return (
            OpenAIModel(model=OPENAI_GATE_MODEL, api_key=os.environ["OPENAI_API_KEY"], max_tokens=6, temperature=0),
            OPENAI_GATE_MODEL,
        )
    return None, None


async def classify(action_text: str) -> tuple[str, str, ReasonCode, str]:
    """Returns (verdict, raw_model_text, reason_code, model_id).

    Every non-`POLICY_ALLOW` path returns verdict `"require_human"` -- an
    unconfigured gate, a timeout, an unreachable backend, and an
    off-schema response are all distinct `ReasonCode`s, but none of them
    is ever interpreted as permission. Uncertainty escalates.
    """
    model, model_id = _gate_model()
    if model is None or model_id is None:
        return "require_human", "no admission model configured", ReasonCode.GATE_UNCONFIGURED, "none"
    try:
        response = await asyncio.wait_for(
            model.complete(
                messages=[
                    Message.system(SYSTEM_PROMPT),
                    Message.user(f"POLICY:\n{POLICY}\n\nPROPOSED ACTION:\n{action_text}\n\nVerdict?"),
                ]
            ),
            timeout=GATE_TIMEOUT_SECONDS,
        )
        text = (response.message.content or "").strip()
        predicted = text.split()[0].rstrip(".,:") if text.split() else text
        if predicted not in VERDICTS:
            return "require_human", text, ReasonCode.OFF_SCHEMA_RESPONSE, model_id
        reason = {
            "allow": ReasonCode.POLICY_ALLOW,
            "deny": ReasonCode.POLICY_DENY,
            "require_human": ReasonCode.POLICY_ESCALATE,
        }[predicted]
        return predicted, text, reason, model_id
    except asyncio.TimeoutError:  # noqa: UP041 -- distinct from builtin TimeoutError before py3.11
        return (
            "require_human",
            f"no response within {GATE_TIMEOUT_SECONDS}s",
            ReasonCode.GATE_TIMEOUT,
            model_id,
        )
    except Exception as exc:  # noqa: BLE001 -- fail closed, never open
        return "require_human", f"gate unreachable ({exc})", ReasonCode.GATE_UNREACHABLE, model_id


async def admission_gate(ctx: ServerRequestContext[Any], call_next: CallNext) -> HandlerResult:
    """The real veto: does NOT call `call_next` when the gate denies, per
    runner.py's own contract ("a middleware that short-circuited without
    call_next is trusted to return its own well-formed result"). Raises
    MCPError instead, the same mechanism `entry.handler` errors already use.

    Every call through here -- allowed or not -- gets a `DecisionRecord`,
    bound to that call's own `args_hash`, appended to `decision_log()` and
    logged structurally. Nothing is cached or reused across calls: the
    next call with the same arguments gets its own fresh classification
    and its own fresh record, not a replayed decision.
    """
    if ctx.method != "tools/call":
        return await call_next(ctx)

    params = ctx.params or {}
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})
    if tool_name not in ("write_file", "delete_file"):
        return await call_next(ctx)

    action_text = f"{tool_name}({json.dumps(arguments, sort_keys=True)})"
    verdict, raw, reason_code, model_id = await classify(action_text)

    record = DecisionRecord(
        request_id=str(ctx.request_id) if ctx.request_id is not None else str(uuid.uuid4()),
        tool_id=tool_name,
        capability_class=CAPABILITY_CLASS[tool_name],
        args_hash=_args_hash(tool_name, arguments),
        policy_version=POLICY_VERSION,
        model_id=model_id,
        decision=verdict,
        reason_code=reason_code,
        issued_at=datetime.now(timezone.utc).isoformat(),
    )
    _decision_log.append(record)
    logger.info("admission decision", extra={"admission_decision": record.as_log_dict()})

    if verdict == "allow":
        return await call_next(ctx)

    raise MCPError(
        code=INVALID_REQUEST,
        message=(
            f"denied by tulip admission gate: verdict={verdict!r} reason={reason_code.value!r} "
            f"raw={raw!r} request_id={record.request_id!r}"
        ),
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
