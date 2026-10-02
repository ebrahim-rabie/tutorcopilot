"""Hermetic scripted fake OpenAI client (single home).

Phase 2's e2e checks and Phase 3's ``FinetuneEvalRunner`` tool_faithfulness
replay all need an LLM-shaped client that answers from a fixed script. This
module is the one place those fakes live; the scripts import
:func:`build_fake_generator` from here instead of redefining their own copies.

No network, no API key: ``Generator.available()`` returns True as soon as a
client is injected, which is exactly the hermetic contract the eval gates rely
on.
"""

from __future__ import annotations

from app.generate import Generator
from app.settings import Settings


class FakeUsage:
    def __init__(self):
        self.prompt_tokens = 10
        self.completion_tokens = 20


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage()


class ScriptedChat:
    """Returns a scripted sequence of decisions, recording every prompt."""

    def __init__(self, script):
        self.responses = iter(script)
        self.calls: list[str] = []

    def create(self, **kwargs):
        messages = kwargs.get("messages") or []
        if messages:
            self.calls.append(messages[-1].get("content", ""))
        return FakeCompletion(next(self.responses))


class FakeCompletions:
    def __init__(self, script):
        self._chat = ScriptedChat(script)
        self.create = self._chat.create


class FakeChat:
    def __init__(self, script):
        self.completions = FakeCompletions(script)


class FakeOpenAI:
    def __init__(self, script):
        self.chat = FakeChat(script)


def build_fake_generator(script, settings: Settings | None = None) -> Generator:
    """Generator whose client answers ``script`` items in order."""
    return Generator(settings or Settings(), client=FakeOpenAI(script))
