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

## Verified

Real MCP client, real stdio transport, real gate calls -- 4/4 correct on a
representative probe (read succeeds ungated, a new write inside scratch is
allowed, an overwrite of an existing file outside scratch is held, a
delete outside scratch is held) -- confirmed by checking the actual
filesystem state afterward, not the tool's own claimed result. Full
methodology: [gist](https://gist.github.com/fede-kamel/561c06c455f418cdf3996c614276276c).
