# Building Production AI Agents

## Core Loop: Plan → Act → Observe
```
while not done and iterations < max:
    plan = llm("What tool to call next?", history + available_tools)
    action = parse_tool_call(plan)
    observation = execute_tool(action)
    history.append(action, observation)
```
Key: iteration cap (usually 5–8), cycle detection, explicit "final answer" tool.

## Tool Design
- **Schema-first**: Define tools with Pydantic/JSON Schema; the LLM sees the schema.
- **Idempotent reads**: `search_knowledge`, `fetch_url` should be safe to retry.
- **Side effects isolated**: `write_note`, `send_email` behind confirmation or audit log.
- **Time-bounded**: every tool has a timeout (default 10s); long tools return streaming updates.

## Memory
| Type | Scope | TTL | Use Case |
|---|---|---|---|
| Short-term (window) | Conversation | Last N turns | Immediate context |
| Long-term (summaries) | User/session | Months | Preferences, facts, weaknesses |
| Episodic (traces) | System | 90 days | Debugging, eval data |

Implementation: store summaries in SQLite/Postgres with embedding index for semantic recall.

## Streaming & UX
- **Token streaming**: SSE tokens to UI for perceived latency.
- **Tool-status events**: "searching...", "reading file...", "computing..." keep user informed.
- **Interrupt/cancel**: AbortController on client; server cancels in-flight tool/LLM call.

## Observability
- **Structured traces**: every step (plan, tool call, result, latency) as JSONL.
- **OpenTelemetry**: spans for LLM, tool, DB; attributes: model, tokens, cost.
- **Per-conversation cost**: $ per turn; budget alerts.

## Guardrails
- **Input sanitization**: length, encoding, injection patterns.
- **Output validation**: JSON schema, citation presence, refusal on low confidence.
- **Faithfulness gate**: LLM judge on final answer; if < threshold, escalate or refuse.
- **Rate limits**: per-user, per-tool, global.

## Hand-Rolled vs. Frameworks
| Approach | Pros | Cons |
|---|---|---|
| Hand-rolled loop | Full control, visible logic, no magic | Boilerplate, reinventing state mgmt |
| LangGraph | State machine, checkpointer, streaming, multi-agent | Abstraction layer, harder to debug internals |

**Recommendation**: Build the loop yourself first. Understand the failure modes. Then adopt LangGraph for production features (persistence, human-in-the-loop).