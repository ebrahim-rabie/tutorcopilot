from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from app.generate import Generator
from app.memory.manager import MemoryManager
from app.settings import Settings
from app.tools.registry import ToolRegistry, ToolResult
from app.tracing import Trace


@dataclass
class AgentStep:
    """Single iteration of the agent loop."""

    step_id: str
    iteration: int
    thought: str
    tool_name: str | None
    tool_args: dict | None
    tool_result: ToolResult | None
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0


@dataclass
class AgentResult:
    """Final result of agent execution."""

    answer: str
    steps: list[AgentStep]
    trace_id: str
    total_latency_ms: float
    tokens_used: dict
    tools_called: list[str]
    success: bool
    error: str | None = None


class AgentState:
    """Agent lifecycle states."""

    PLANNING = "planning"
    ACTING = "acting"
    OBSERVING = "observing"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    ERROR = "error"


SYSTEM_PROMPT = """You are TutorCopilot Agent, an AI engineering assistant with tools.

## Available Tools
{tool_schemas}

## Operating Rules
1. **Think before acting**: For any substantive question — RAG, agents, chunking,
   retrieval, evaluation, fine-tuning, or anything this knowledge base covers —
   call search_knowledge FIRST and answer from its results. Never answer domain
   questions from memory. Only greetings and meta questions about the agent itself
   may be answered directly.
2. **One tool per step**: Call ONE tool, observe result, then decide next step.
3. **Code examples**: when the user asks for code, search for the relevant material
   and also use run_python to execute a working snippet when possible.
4. **Cite sources**: When using search_knowledge, cite results with [1], [2] in your final answer.
5. **Iterate**: You have up to {max_iterations} iterations. Use them wisely.
6. **Final answer**: When you have enough information, output a JSON with "final_answer" key.

## Response Format (JSON)
{{
  "thought": "Your reasoning about what to do next",
  "tool": "tool_name" | null,
  "args": {{}} | null,
  "final_answer": "Your complete answer with citations" | null
}}

- If tool is not null, also provide args.
- If final_answer is not null, the loop terminates.
- Be concise in thought; focus on decision-making.
- STRICT: only these four keys are ever allowed. Native tool-call syntax
  like {{"name": ..., "arguments": ...}} is rejected by the API and fails the
  whole request — tool arguments go under "args", never "arguments".
  To call a tool emit: {{"thought": "...", "tool": "search_knowledge", "args": {{}}, "final_answer": null}}.
"""


def build_system_prompt(registry: ToolRegistry, max_iterations: int) -> str:
    schemas = []
    for t in registry._tools.values():
        s = t.schema
        params = ", ".join(
            [f"{k}: {v.get('type', 'any')}" for k, v in s.parameters.get("properties", {}).items()]
        )
        schemas.append(f"- {s.name}({params}): {s.description}")
    tool_list = "\n".join(schemas)
    return SYSTEM_PROMPT.format(tool_schemas=tool_list, max_iterations=max_iterations)


def _plan_once(generator: Generator, settings: Settings, messages: list[dict], *, json_mode: bool = True):
    """One planning call. Returns (response, error) — never raises.

    Empty content counts as a failure: gpt-oss reasoning occasionally stops
    after the reasoning channel, and without a check that slips through as a
    silent JSON parse error.
    """
    kwargs: dict = dict(
        model=settings.llm_model,
        temperature=0.1,
        max_tokens=settings.max_tokens,
        messages=messages,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = generator.client.chat.completions.create(**kwargs)
    except Exception as e:  # any API/transport failure must degrade, not crash
        return None, str(e)
    if not (resp.choices[0].message.content or "").strip():
        return None, "empty_content: model returned no content"
    return resp, None


_FORMAT_REMINDER = (
    "FORMAT ERROR: your previous reply was rejected by the API. Reply with ONLY this JSON object: "
    '{"thought": "...", "tool": "tool_name or null", "args": {...} or null, "final_answer": "..." or null}. '
    'Never use native tool-call syntax {"name": ..., "arguments": ...} — it is invalid here.'
)


def _retry_delay(err: str | None) -> float:
    """Backoff before re-asking: rate limits need real time to recover."""
    low = (err or "").lower()
    return 12.0 if "429" in low or "rate limit" in low else 1.0


def _is_quota_exhausted(err: str | None) -> bool:
    """Daily token quota: retrying within the same day is futile."""
    low = (err or "").lower()
    return "per day" in low or "tpd" in low


def _plan(generator: Generator, settings: Settings, messages: list[dict]):
    """Planning call with corrective retries on API rejection.

    Groq + gpt-oss intermittently rejects json-mode replies
    (``json_validate_failed``, often with empty ``failed_generation``) and
    sometimes returns empty content outright. Retry up to 6 times, alternating
    json mode with plain mode (no server-side validation) so a validation
    quirk can't sink the whole turn. Daily-quota errors fast-fail.
    """
    sequence = [True, True, False, True, False, False]  # json/plain per attempt
    err = None
    for i, json_mode in enumerate(sequence):
        if i:
            time.sleep(_retry_delay(err))
        attempt = messages if i == 0 else [*messages, {"role": "user", "content": _FORMAT_REMINDER}]
        resp, err = _plan_once(generator, settings, attempt, json_mode=json_mode)
        if resp is not None:
            return resp, None
        if _is_quota_exhausted(err):
            return None, err
    return None, err


def _extract_json(text: str) -> dict | None:
    """Pull the first balanced JSON object out of arbitrary text."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break  # malformed pair; try the next '{'
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    return None


def _parse_decision(raw: str) -> dict:
    """Parse a plan reply: strict JSON, then tolerant extraction, then nudge."""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    extracted = _extract_json(raw)
    if isinstance(extracted, dict):
        return extracted
    return {"thought": "Parse error, retrying", "tool": None, "args": None, "final_answer": None}


_TRIVIAL_QUESTIONS = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "hi there",
        "hey there",
        "thanks",
        "thank you",
        "thank you so much",
        "ty",
        "bye",
        "goodbye",
        "ok",
        "okay",
        "yes",
        "no",
        "help",
        "who are you",
        "what are you",
        "what can you do",
        "what do you do",
        "tell me about yourself",
    }
)

_CODE_HINTS = ("code", "python", "snippet", "script", "implement", "write a", "run it")
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def _is_trivial(question: str) -> bool:
    """Greetings/meta pings only — everything else must be grounded."""
    normalized = re.sub(r"[^a-z0-9 ]", "", question.lower()).strip()
    return normalized in _TRIVIAL_QUESTIONS


def _wants_code(question: str) -> bool:
    low = question.lower()
    return any(hint in low for hint in _CODE_HINTS)


def _extract_code(answer: str) -> str | None:
    m = _CODE_BLOCK.search(answer or "")
    code = m.group(1).strip() if m else ""
    return code or None


def _apply_grounding(
    decision: dict, attempted: set[str | None], question: str, trace: Trace
) -> None:
    """Deterministic grounding: the model must search before answering a
    domain question, and run code when asked for code.

    Prompt rules alone are not enough — live runs answered from memory with
    fabricated citations. This turns the spec into loop behaviour: a
    ``final_answer`` is only accepted once ``search_knowledge`` was attempted
    (and ``run_python`` for code requests). Greetings are exempt.
    """
    if not decision.get("final_answer") or _is_trivial(question):
        return
    if "search_knowledge" not in attempted:
        trace.log("grounding_enforced", {"tool": "search_knowledge", "reason": "answer_without_search"})
        thought = str(decision.get("thought") or "")
        decision["thought"] = f"{thought} | grounding: search before answering".strip(" |")
        decision["tool"] = "search_knowledge"
        decision["args"] = {"query": question}
        decision["final_answer"] = None
        return
    if _wants_code(question) and "run_python" not in attempted:
        code = _extract_code(str(decision.get("final_answer") or ""))
        if code:
            trace.log("grounding_enforced", {"tool": "run_python", "reason": "code_request"})
            decision["tool"] = "run_python"
            decision["args"] = {"code": code}
            decision["final_answer"] = None


async def run_agent_loop(
    question: str,
    generator: Generator,
    registry: ToolRegistry,
    memory: MemoryManager,
    settings: Settings,
    trace: Trace,
    max_iterations: int = 6,
    conv_id: str | None = None,
) -> AgentResult:
    """Hand-rolled Plan → Act → Observe loop."""
    start = time.monotonic()
    conv_id = conv_id or uuid.uuid4().hex[:12]
    system_prompt = build_system_prompt(registry, max_iterations)

    # Build initial context
    context = memory.short_term.get_context()
    if context:
        context = f"Recent conversation:\n{context}\n\n"
    user_prompt = f"{context}Current question: {question}"

    steps: list[AgentStep] = []
    tools_called: list[str] = []
    tokens_used = {"prompt": 0, "completion": 0}
    state = AgentState.PLANNING
    final_answer = None
    error = None

    for iteration in range(1, max_iterations + 1):
        trace.log(f"iteration_{iteration}_start", {"state": state})

        # PLANNING: Ask LLM what to do
        step_start = time.monotonic()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp, plan_err = _plan(generator, settings, messages)
        if resp is None:
            error = f"LLM call failed: {plan_err}"
            state = AgentState.ERROR
            trace.log("llm_error", {"error": plan_err})
            break

        tokens_used["prompt"] += resp.usage.prompt_tokens if resp.usage else 0
        tokens_used["completion"] += resp.usage.completion_tokens if resp.usage else 0

        raw = (resp.choices[0].message.content or "").strip()
        decision = _parse_decision(raw)
        _apply_grounding(
            decision, {s.tool_name for s in steps}, question, trace
        )

        thought = decision.get("thought", "")
        tool_name = decision.get("tool")
        tool_args = decision.get("args") or {}
        final_answer = decision.get("final_answer")

        step = AgentStep(
            step_id=uuid.uuid4().hex[:8],
            iteration=iteration,
            thought=thought,
            tool_name=tool_name,
            tool_args=tool_args if tool_name else None,
            tool_result=None,
        )

        # If final answer, we're done
        if final_answer:
            trace.log("final_answer", {"answer": final_answer, "iteration": iteration})
            state = AgentState.COMPLETE
            steps.append(step)
            break

        # ACTING: Execute tool
        if tool_name:
            state = AgentState.ACTING
            tool = registry.get(tool_name)
            if not tool:
                step.tool_result = ToolResult(success=False, error=f"Unknown tool: {tool_name}")
                trace.log("tool_error", {"tool": tool_name, "error": "unknown"})
            else:
                valid, err = tool.validate_args(tool_args)
                if not valid:
                    step.tool_result = ToolResult(success=False, error=err)
                    trace.log("tool_error", {"tool": tool_name, "error": err})
                else:
                    trace.log("tool_call", {"tool": tool_name, "args": tool_args})
                    step_start = time.monotonic()
                    result = await tool.execute(**tool_args)
                    step.duration_ms = (time.monotonic() - step_start) * 1000
                    step.tool_result = result
                    tools_called.append(tool_name)
                    trace.log(
                        "tool_result",
                        {
                            "tool": tool_name,
                            "success": result.success,
                            "latency_ms": step.duration_ms,
                        },
                    )

                    # OBSERVING: Add result to context for next iteration
                    state = AgentState.OBSERVING
                    result_text = json.dumps(result.to_dict(), ensure_ascii=False)
                    user_prompt = f"{user_prompt}\n\nTool result ({tool_name}):\n{result_text}\n\nDecide next step."
        else:
            # No tool called, no final answer -> nudge
            user_prompt = f"{user_prompt}\n\nYou didn't call a tool or give a final answer. Please decide next step."

        steps.append(step)

        # Cycle detection: if last 3 steps same tool with similar args, break
        if len(steps) >= 3:
            recent = steps[-3:]
            if all(s.tool_name == recent[0].tool_name for s in recent):
                trace.log("cycle_detected", {"tool": recent[0].tool_name})
                if not final_answer:
                    final_answer = (
                        "I noticed I was repeating the same action. Here's what I found so far."
                    )
                break

    else:
        # Max iterations reached
        trace.log("max_iterations_reached", {"iterations": max_iterations})
        if not final_answer:
            final_answer = "I couldn't reach a conclusion within the iteration limit. Please try rephrasing your question."

    total_latency = (time.monotonic() - start) * 1000
    success = error is None and final_answer is not None

    # Save turn to memory
    if memory.current_conv_id:
        memory.add_turn(
            user_msg=question,
            assistant_msg=final_answer or error or "",
            tools_used=tools_called,
            success=success,
            trace_id=trace.trace_id,
        )

    return AgentResult(
        answer=final_answer or error or "Agent execution failed.",
        steps=steps,
        trace_id=trace.trace_id,
        total_latency_ms=total_latency,
        tokens_used=tokens_used,
        tools_called=tools_called,
        success=success,
        error=error,
    )


# Streaming variant (yields steps as they happen)
async def run_agent_stream(
    question: str,
    generator: Generator,
    registry: ToolRegistry,
    memory: MemoryManager,
    settings: Settings,
    trace: Trace,
    max_iterations: int = 6,
    conv_id: str | None = None,
) -> AsyncGenerator[dict, None]:
    """Streaming version yields events: step_start, step_end, final."""
    conv_id = conv_id or uuid.uuid4().hex[:12]
    system_prompt = build_system_prompt(registry, max_iterations)
    context = memory.short_term.get_context()
    if context:
        context = f"Recent conversation:\n{context}\n\n"
    user_prompt = f"{context}Current question: {question}"

    tools_called: list[str] = []
    attempted: set[str | None] = set()
    tokens_used = {"prompt": 0, "completion": 0}
    final_answer = None
    error = None

    for iteration in range(1, max_iterations + 1):
        yield {"type": "step_start", "iteration": iteration, "state": "planning"}

        # Planning
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp, plan_err = _plan(generator, settings, messages)
        if resp is None:
            error = f"LLM call failed: {plan_err}"
            trace.log("llm_error", {"error": plan_err})
            yield {"type": "error", "error": error}
            break
        tokens_used["prompt"] += resp.usage.prompt_tokens if resp.usage else 0
        tokens_used["completion"] += resp.usage.completion_tokens if resp.usage else 0

        raw = (resp.choices[0].message.content or "").strip()
        decision = _parse_decision(raw)
        _apply_grounding(decision, attempted, question, trace)

        thought = decision.get("thought", "")
        tool_name = decision.get("tool")
        tool_args = decision.get("args") or {}
        final_answer = decision.get("final_answer")

        yield {"type": "thought", "iteration": iteration, "thought": thought}

        if final_answer:
            yield {"type": "final_answer", "answer": final_answer}
            break

        if tool_name:
            yield {"type": "tool_call", "tool": tool_name, "args": tool_args}
            attempted.add(tool_name)
            tool = registry.get(tool_name)
            if tool:
                valid, err = tool.validate_args(tool_args)
                if not valid:
                    result = ToolResult(success=False, error=err)
                else:
                    result = await tool.execute(**tool_args)
                    tools_called.append(tool_name)
            else:
                result = ToolResult(success=False, error=f"Unknown tool: {tool_name}")

            yield {"type": "tool_result", "tool": tool_name, "result": result.to_dict()}

            # Update context
            result_text = json.dumps(result.to_dict(), ensure_ascii=False)
            user_prompt = (
                f"{user_prompt}\n\nTool result ({tool_name}):\n{result_text}\n\nDecide next step."
            )
        else:
            yield {"type": "nudge", "message": "No tool or final answer provided"}
            user_prompt = f"{user_prompt}\n\nPlease call a tool or provide final_answer."

    else:
        yield {"type": "max_iterations", "message": "Iteration limit reached"}

    if not final_answer:
        final_answer = "I couldn't reach a conclusion. Please try rephrasing."

    # Save to memory
    if memory.current_conv_id:
        memory.add_turn(
            user_msg=question,
            assistant_msg=final_answer,
            tools_used=tools_called,
            success=error is None,
            trace_id=trace.trace_id,
        )

    yield {
        "type": "done",
        "answer": final_answer,
        "tools_called": tools_called,
        "tokens_used": tokens_used,
        "conversation_id": conv_id,
        "trace_id": trace.trace_id,
    }
