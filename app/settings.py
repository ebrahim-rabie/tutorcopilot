from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TC_",
        env_file=str(ROOT_DIR / ".env"),
        extra="ignore",
    )

    # --- LLM (OpenAI-compatible) ---
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_judge_model: str = ""
    temperature: float = 0.2
    max_tokens: int = 4000

    # --- Embeddings ---
    embedding_mode: str = "local"  # "local" | "api"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_api_model: str = "text-embedding-3-small"

    # --- Rerank ---
    use_rerank: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Chunking ---
    chunk_strategy: str = "markdown"  # "markdown" | "recursive" | "semantic"
    chunk_size: int = 500
    chunk_overlap: int = 80

    # --- Retrieval ---
    hybrid_top_k: int = 12
    final_top_k: int = 4
    rrf_k: int = 60

    # --- Paths ---
    sources_dir: Path = ROOT_DIR / "data" / "sources"
    index_dir: Path = ROOT_DIR / "config" / "saved_index"
    trace_dir: Path = ROOT_DIR / "traces"
    eval_data_path: Path = ROOT_DIR / "evals" / "golden_queries.jsonl"
    eval_report_dir: Path = ROOT_DIR / "evals" / "reports"
    episodic_dir: Path = ROOT_DIR / "data" / "episodic"
    finetune_dir: Path = ROOT_DIR / "config" / "finetune"

    # --- Phase 2: Agent ---
    agent_max_iterations: int = 6
    agent_max_input_chars: int = 5000
    faithfulness_threshold: int = 3
    rate_limit_requests: int = 100
    rate_limit_refill: float = 10.0

    # --- Voice (STT / TTS) ---
    stt_model: str = "whisper-large-v3-turbo"
    stt_max_upload_mb: int = 10
    tts_provider: str = "edge"  # "edge" | "none"
    tts_voice: str = "en-US-AriaNeural"
