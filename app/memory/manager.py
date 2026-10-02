from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ConversationTurn:
    """One user-assistant exchange."""

    turn_id: str
    timestamp: float
    user_message: str
    assistant_message: str
    tools_used: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    success: bool = True
    trace_id: str = ""


@dataclass
class TopicSummary:
    """Aggregated knowledge about a topic."""

    topic: str
    summary: str
    turn_count: int = 0
    last_updated: float = field(default_factory=time.time)
    weakness_score: float = 0.0  # 0=strong, 1=weak


class ShortTermMemory:
    """Sliding window of recent conversation turns."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.turns: list[ConversationTurn] = []

    def add(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def get_context(self, include_tools: bool = True) -> str:
        """Format recent turns for LLM context."""
        lines = []
        for t in self.turns:
            lines.append(f"User: {t.user_message}")
            lines.append(f"Assistant: {t.assistant_message}")
            if include_tools and t.tools_used:
                lines.append(f"[Tools: {', '.join(t.tools_used)}]")
        return "\n".join(lines)

    def get_messages(self) -> list[dict]:
        """OpenAI-compatible message list."""
        msgs = []
        for t in self.turns:
            msgs.append({"role": "user", "content": t.user_message})
            msgs.append({"role": "assistant", "content": t.assistant_message})
        return msgs


class LongTermMemory:
    """Persistent topic summaries + weak-area tracking."""

    def __init__(self, storage_path: Path | str = "data/memory"):
        self.path = Path(storage_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.topics: dict[str, TopicSummary] = {}
        self._load()

    def _load(self) -> None:
        file = self.path / "topics.json"
        if file.exists():
            data = json.loads(file.read_text(encoding="utf-8"))
            self.topics = {k: TopicSummary(**v) for k, v in data.items()}

    def _save(self) -> None:
        file = self.path / "topics.json"
        data = {k: v.__dict__ for k, v in self.topics.items()}
        file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def update_from_turn(self, turn: ConversationTurn) -> None:
        """Extract topics from turn and update summaries."""
        # Simple topic extraction: use tools used + key nouns from user message
        topics = set(turn.tools_used)
        # Could add NER/keyphrase extraction here
        for topic in topics:
            if topic not in self.topics:
                self.topics[topic] = TopicSummary(topic=topic, summary="")
            ts = self.topics[topic]
            ts.turn_count += 1
            ts.last_updated = time.time()
            # Update weakness: if turn failed or assistant was uncertain
            if not turn.success:
                ts.weakness_score = min(1.0, ts.weakness_score + 0.1)
        self._save()

    def get_weakest_topic(self) -> str | None:
        if not self.topics:
            return None
        return max(self.topics.values(), key=lambda t: t.weakness_score).topic

    def get_topic_summary(self, topic: str) -> str | None:
        ts = self.topics.get(topic)
        return ts.summary if ts else None

    def set_topic_summary(self, topic: str, summary: str) -> None:
        if topic not in self.topics:
            self.topics[topic] = TopicSummary(topic=topic, summary=summary)
        else:
            self.topics[topic].summary = summary
        self._save()


class EpisodicMemory:
    """Full conversation traces for debugging/training (Phase 3)."""

    def __init__(self, storage_path: Path | str = "data/episodic"):
        self.path = Path(storage_path)
        self.path.mkdir(parents=True, exist_ok=True)

    def save_conversation(
        self, conv_id: str, turns: list[ConversationTurn], metadata: dict | None = None
    ) -> None:
        file = self.path / f"{conv_id}.json"
        data = {
            "conversation_id": conv_id,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "turns": [t.__dict__ for t in turns],
        }
        file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_conversation(self, conv_id: str) -> dict | None:
        file = self.path / f"{conv_id}.json"
        if not file.exists():
            return None
        return json.loads(file.read_text(encoding="utf-8"))


class MemoryManager:
    """Unified memory interface for the agent."""

    def __init__(
        self,
        short_term_max: int = 10,
        long_term_path: Path | str = "data/memory",
        episodic_path: Path | str = "data/episodic",
    ):
        self.short_term = ShortTermMemory(max_turns=short_term_max)
        self.long_term = LongTermMemory(long_term_path)
        self.episodic = EpisodicMemory(episodic_path)
        self.current_conv_id: str | None = None
        self.current_turns: list[ConversationTurn] = []

    def start_conversation(self, conv_id: str | None = None) -> str:
        self.current_conv_id = conv_id or uuid.uuid4().hex[:12]
        self.current_turns = []
        return self.current_conv_id

    def add_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        tools_used: list[str],
        success: bool = True,
        trace_id: str = "",
    ) -> ConversationTurn:
        turn = ConversationTurn(
            turn_id=uuid.uuid4().hex[:8],
            timestamp=time.time(),
            user_message=user_msg,
            assistant_message=assistant_msg,
            tools_used=tools_used,
            success=success,
            trace_id=trace_id,
        )
        self.short_term.add(turn)
        self.current_turns.append(turn)
        self.long_term.update_from_turn(turn)
        return turn

    def end_conversation(self, metadata: dict | None = None, *, persist: bool = True) -> None:
        # persist=False: eval/e2e runs are synthetic (fake LLM) and must never
        # land in data/episodic — that directory is training data (P3.1 input).
        if persist and self.current_conv_id and self.current_turns:
            self.episodic.save_conversation(self.current_conv_id, self.current_turns, metadata)
        self.current_conv_id = None
        self.current_turns = []

    def get_weakest_topic(self) -> str | None:
        return self.long_term.get_weakest_topic()
