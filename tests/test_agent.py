from app.agent.loop import build_system_prompt
from app.memory.manager import ConversationTurn, LongTermMemory, MemoryManager, ShortTermMemory
from app.tools.registry import BaseTool, ToolRegistry, ToolResult


class DummyTool(BaseTool):
    name = "dummy"
    description = "A dummy tool"
    parameters = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    required = ["x"]

    async def execute(self, x: int) -> ToolResult:
        return ToolResult(success=True, data={"result": x * 2})


def test_system_prompt_building():
    reg = ToolRegistry()
    reg.register(DummyTool())
    prompt = build_system_prompt(reg, max_iterations=5)
    assert "dummy" in prompt
    assert "5" in prompt
    assert "Think before acting" in prompt


def test_short_term_memory():
    mem = ShortTermMemory(max_turns=3)
    for i in range(5):
        mem.add(
            ConversationTurn(
                turn_id=f"t{i}",
                timestamp=0,
                user_message=f"Q{i}",
                assistant_message=f"A{i}",
            )
        )
    assert len(mem.turns) == 3
    assert mem.turns[0].user_message == "Q2"  # oldest dropped
    ctx = mem.get_context()
    assert "Q2" in ctx and "Q4" in ctx


def test_long_term_memory(tmp_path):
    mem = LongTermMemory(tmp_path)
    turn = ConversationTurn(
        turn_id="t1",
        timestamp=0,
        user_message="test",
        assistant_message="ans",
        tools_used=["search_knowledge"],
        success=False,
    )
    mem.update_from_turn(turn)
    assert "search_knowledge" in mem.topics
    assert mem.topics["search_knowledge"].weakness_score > 0
    assert mem.get_weakest_topic() == "search_knowledge"


def test_memory_manager(tmp_path):
    mm = MemoryManager(
        long_term_path=tmp_path / "memory",
        episodic_path=tmp_path / "episodic",
    )
    conv_id = mm.start_conversation()
    assert conv_id is not None
    turn = mm.add_turn("hi", "hello", ["search_knowledge"], success=True)
    assert turn.tools_used == ["search_knowledge"]
    assert mm.short_term.turns[-1].turn_id == turn.turn_id
    mm.end_conversation()
    assert mm.current_conv_id is None
    assert (tmp_path / "episodic" / f"{conv_id}.json").exists()  # persisted


def test_end_conversation_persist_false_writes_nothing(tmp_path):
    """Eval/e2e runs must never write training data into data/episodic."""
    episodic = tmp_path / "episodic"
    mm = MemoryManager(long_term_path=tmp_path / "m", episodic_path=episodic)
    mm.start_conversation()
    mm.add_turn("q", "hello", ["search_knowledge"])
    mm.end_conversation({"test_id": "conv1"}, persist=False)
    assert mm.current_conv_id is None
    assert not episodic.exists() or list(episodic.iterdir()) == []


def test_tool_registry():
    reg = ToolRegistry()
    reg.register(DummyTool())
    assert reg.get("dummy") is not None
    assert reg.get("nonexistent") is None
    schemas = reg.all_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "dummy"


def test_dummy_tool_execute():
    import asyncio

    tool = DummyTool()
    result = asyncio.run(tool.execute(x=5))
    assert result.success
    assert result.data["result"] == 10


def test_agent_loop_imports():
    # Just verify imports work

    assert True
