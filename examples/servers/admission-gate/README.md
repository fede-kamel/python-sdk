# Admission Gate

An MCP server exposing filesystem tools, gating `write_file`/`delete_file`
through a real admission decision using the SDK's own `ServerMiddleware` --
a genuine pre-execution veto (the runner's own comment calls this a
"middleware veto"), not observation. No existing example in this repo
denies a call; the closest pattern, `stories/refund_desk`, uses elicitation
for mid-call parameter confirmation -- a different, complementary
mechanism to a pre-call approve/deny gate.

The gate is driven by real [tulip-agents](https://tulipagents.ai) code
(`tulip.models.native.openai.OpenAIModel`, whose `base_url` override is
documented for vLLM endpoints) -- a dependency of this example only. The
backing model is [Clusiana](https://tulipagents.ai), a real, currently
unreleased checkpoint trained for exactly this three-word decision; any
tulip-compatible chat model works the same way.

## Try it

```bash
pip install -e .
export TULIP_GATE_SSH_HOST=your-model-host   # if the model needs an SSH hop
export TULIP_GATE_URL=http://127.0.0.1:8010   # or reachable directly
export TULIP_GATE_MODEL=your-model-name
mcp-admission-gate
```

Without a `TULIP_GATE_*` variable set, `TULIP_GATE_OPENAI_MODEL` +
`OPENAI_API_KEY` works with any real OpenAI model instead -- no private
infrastructure required to try the pattern.

## An admission decision is a policy input, not proof of safety

Real review feedback on the tracking issue
([#3272](https://github.com/modelcontextprotocol/python-sdk/issues/3272), from
[Ecocitizenz](https://www.ecocitizenz.com)) made a distinction worth
building around rather than just asserting: what the model returns is one
input to a policy decision, not a guarantee the action is safe. In
response, this example now makes that concrete:

- **Every decision is bound to its own `args_hash`** -- a hash of the exact
  tool + arguments the decision was about (`_args_hash`), not just the
  tool's name.
- **The policy is versioned** (`POLICY_VERSION`) -- a version bump is
  visible on every decision made after it.
- **Failure modes are a fixed, deterministic set** (`ReasonCode`), not
  freeform strings: an unconfigured gate, an unreachable one, a timeout, and
  an off-schema model response are each their own code -- and every one of
  them resolves to `require_human`, never `allow`. Tested directly in
  `tests/test_admission_gate.py::test_every_gate_failure_mode_escalates_never_allows`,
  parametrized over all four.
- **A minimal, secret-redacted `DecisionRecord`** is emitted for every
  gated call -- `request_id`, `tool_id`, `capability_class`, `args_hash`,
  `policy_version`, `model_id`, `decision`, `reason_code`, `issued_at` --
  logged structurally and kept in `decision_log()`. It deliberately never
  carries the raw arguments (`write_file`'s `content` could be arbitrary
  file data); `args_hash` is what a caller checks a claim against.
- **A bounded timeout** (`GATE_TIMEOUT_SECONDS`, default 10s) on the
  admission call itself -- an unbounded call is its own fail-open risk if
  the backing model hangs instead of erroring.

What's still explicitly out of scope, disclosed rather than silently
absent -- this is an example, not a production authorization framework:

- **Replay of a stale approval against different arguments** isn't a live
  risk here structurally, not just by policy: nothing caches or reuses a
  decision across calls, so every `tools/call` gets its own fresh
  classification against its own `args_hash`. There's no stored approval
  to replay.
- **Materially changed tool schemas** and **model/checkpoint changes**
  aren't detected or pinned. `model_id` is recorded on every decision (so a
  checkpoint swap is visible after the fact), but nothing here enforces
  re-approval on a schema or checkpoint change -- real, non-trivial work a
  higher-stakes deployment would need and this example doesn't attempt.

## Verified

Real MCP client, real stdio transport, real gate calls -- 4/4 correct on a
representative probe (read succeeds ungated, a new write inside scratch is
allowed, an overwrite of an existing file outside scratch is held, a
delete outside scratch is held) -- confirmed by checking the actual
filesystem state afterward, not the tool's own claimed result, and
re-confirmed after the changes above against the same live model. Full
methodology: [gist](https://gist.github.com/fede-kamel/561c06c455f418cdf3996c614276276c).

The middleware's own veto logic -- independent of whether any admission
model is reachable -- is exercised directly in
`tests/test_admission_gate.py` (`pytest tests/`): the allow/deny/escalate
paths, the `DecisionRecord` each one produces, and that every gate-failure
`ReasonCode` vetoes rather than allows.
