"""Real, independently-testable exercise of the admission-gate middleware.

Directly requested in review feedback on the tracking issue
(https://github.com/modelcontextprotocol/python-sdk/issues/3272): "The
middleware veto deserves a first-class, independently testable example."

These tests exercise `admission_gate` itself -- the veto logic, the
`DecisionRecord` it produces, and the deterministic `ReasonCode` for each
failure mode -- by monkeypatching `classify()` to return a controlled
verdict, so none of this needs a live admission model reachable in CI.
The separate, already-published live-server verification (linked from
README.md) covers the real end-to-end transport + real model path; this
file covers the middleware's own decision-and-record logic in isolation.

`ctx` is a `SimpleNamespace`, not a real `ServerRequestContext` -- the
middleware only reads `.method`, `.params`, and `.request_id` off it, and
building the real thing needs a live transport/session this test doesn't
need. That's a deliberate, disclosed scope choice, not an oversight.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from mcp.shared.exceptions import MCPError

import mcp_admission_gate.server as server_module
from mcp_admission_gate.server import (
    CAPABILITY_CLASS,
    POLICY_VERSION,
    DecisionRecord,
    ReasonCode,
    _args_hash,
    admission_gate,
    decision_log,
)


async def _call_next_stub(ctx: Any) -> str:
    return "tool executed"


def _ctx(tool_name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        method="tools/call",
        params={"name": tool_name, "arguments": arguments},
        request_id="req-1",
    )


@pytest.fixture(autouse=True)
def _clear_decision_log() -> None:
    server_module._decision_log.clear()
    yield
    server_module._decision_log.clear()


def test_args_hash_is_order_independent() -> None:
    a = _args_hash("write_file", {"path": "/tmp/x", "content": "hi"})
    b = _args_hash("write_file", {"content": "hi", "path": "/tmp/x"})
    assert a == b


def test_args_hash_differs_for_different_arguments() -> None:
    a = _args_hash("write_file", {"path": "/tmp/x", "content": "hi"})
    b = _args_hash("write_file", {"path": "/tmp/x", "content": "bye"})
    assert a != b


@pytest.mark.anyio
async def test_read_file_is_never_gated() -> None:
    """Reads bypass the gate entirely -- no classify() call, no record."""
    ctx = _ctx("read_file", {"path": "/tmp/x"})
    result = await admission_gate(ctx, _call_next_stub)
    assert result == "tool executed"
    assert decision_log() == []


@pytest.mark.anyio
async def test_allow_verdict_runs_the_real_tool_and_records_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server_module,
        "classify",
        AsyncMock(return_value=("allow", "allow", ReasonCode.POLICY_ALLOW, "test-model")),
    )
    ctx = _ctx("write_file", {"path": "/tmp/scratch/x", "content": "hi"})
    result = await admission_gate(ctx, _call_next_stub)

    assert result == "tool executed"
    [record] = decision_log()
    assert record.decision == "allow"
    assert record.reason_code == ReasonCode.POLICY_ALLOW
    assert record.tool_id == "write_file"
    assert record.capability_class == CAPABILITY_CLASS["write_file"]
    assert record.policy_version == POLICY_VERSION
    assert record.model_id == "test-model"
    assert record.args_hash == _args_hash("write_file", {"path": "/tmp/scratch/x", "content": "hi"})
    # The record never carries the raw arguments -- only their hash.
    assert "content" not in json.dumps(vars(record) if not isinstance(record, DecisionRecord) else record.as_log_dict())


@pytest.mark.anyio
async def test_deny_verdict_vetoes_before_call_next(monkeypatch: pytest.MonkeyPatch) -> None:
    call_next = AsyncMock(return_value="tool executed")
    monkeypatch.setattr(
        server_module,
        "classify",
        AsyncMock(return_value=("deny", "deny", ReasonCode.POLICY_DENY, "test-model")),
    )
    ctx = _ctx("delete_file", {"path": "/etc/passwd"})

    with pytest.raises(MCPError):
        await admission_gate(ctx, call_next)

    call_next.assert_not_called()
    [record] = decision_log()
    assert record.decision == "deny"
    assert record.reason_code == ReasonCode.POLICY_DENY


@pytest.mark.anyio
async def test_require_human_verdict_vetoes_before_call_next(monkeypatch: pytest.MonkeyPatch) -> None:
    call_next = AsyncMock(return_value="tool executed")
    monkeypatch.setattr(
        server_module,
        "classify",
        AsyncMock(return_value=("require_human", "require_human", ReasonCode.POLICY_ESCALATE, "test-model")),
    )
    ctx = _ctx("write_file", {"path": "/etc/important.conf", "content": "overwritten"})

    with pytest.raises(MCPError):
        await admission_gate(ctx, call_next)

    call_next.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "reason_code",
    [
        ReasonCode.GATE_UNCONFIGURED,
        ReasonCode.GATE_TIMEOUT,
        ReasonCode.GATE_UNREACHABLE,
        ReasonCode.OFF_SCHEMA_RESPONSE,
    ],
)
async def test_every_gate_failure_mode_escalates_never_allows(
    monkeypatch: pytest.MonkeyPatch, reason_code: ReasonCode
) -> None:
    """The point Ecocitizenz's review raised: an admission model being
    unavailable, timing out, or answering off-schema must never be
    interpreted as permission. Every one of these must still veto."""
    call_next = AsyncMock(return_value="tool executed")
    monkeypatch.setattr(
        server_module,
        "classify",
        AsyncMock(return_value=("require_human", "n/a", reason_code, "test-model")),
    )
    ctx = _ctx("delete_file", {"path": "/tmp/scratch/x"})

    with pytest.raises(MCPError):
        await admission_gate(ctx, call_next)

    call_next.assert_not_called()
    [record] = decision_log()
    assert record.reason_code == reason_code
    assert record.decision != "allow"


@pytest.mark.anyio
async def test_classify_without_a_configured_gate_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """No TULIP_GATE_* / OPENAI_API_KEY env set -- classify() itself, not
    just admission_gate's caller, must still escalate rather than allow."""
    monkeypatch.delenv("TULIP_GATE_URL", raising=False)
    monkeypatch.delenv("TULIP_GATE_SSH_HOST", raising=False)
    monkeypatch.delenv("TULIP_GATE_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    verdict, _raw, reason_code, model_id = await server_module.classify("write_file({})")

    assert verdict == "require_human"
    assert reason_code == ReasonCode.GATE_UNCONFIGURED
    assert model_id == "none"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
