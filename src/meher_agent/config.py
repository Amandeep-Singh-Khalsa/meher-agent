"""Configuration loading.

Two sources, in increasing priority:
  1. config.toml  - every tunable (temperature, step limit, INR/USD rate, ...)
  2. environment  - the three model-access vars, plus a few overrides

Nothing in the codebase may hardcode a tunable. See .env.example.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.toml"

# Rupee constants that appear in data/policies.md. They are part of the shop's
# data, not tunables, so they live here as named constants and are asserted
# against policies.md by tests/test_corpus.py.
POLICY_FREE_DELIVERY_ABOVE_INR = 999
POLICY_DELIVERY_FEE_INR = 60
POLICY_COD_LIMIT_INR = 5000
POLICY_DISCOUNT_PCT = 5
POLICY_DISCOUNT_MIN_BOXES = 50
POLICY_BULK_ADVANCE_PCT = 30
POLICY_BULK_KG = 10
POLICY_BULK_BOXES = 25
POLICY_DELIVERY_RADIUS_KM = 8
POLICY_SAME_DAY_CUTOFF = "4:00 pm"
POLICY_GST_PCT = 5
POLICY_DAMAGED_WINDOW_HOURS = 2


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_steps: int
    num_ctx: int
    request_timeout_s: float
    max_retries: int
    retry_backoff_s: float


@dataclass(frozen=True)
class AgentConfig:
    max_reply_chars: int
    min_reply_chars: int
    allow_reply_repair: bool
    allow_intent_repair: bool


@dataclass(frozen=True)
class RetrievalConfig:
    section_top_k: int
    sku_top_k: int
    bm25_k1: float
    bm25_b: float
    min_section_score: float


@dataclass(frozen=True)
class CostConfig:
    inr_per_usd: float
    input_cost_per_mtok_usd: float
    output_cost_per_mtok_usd: float


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int


@dataclass(frozen=True)
class RuntimeConfig:
    log_level: str
    max_conversations: int
    max_history_turns: int
    data_dir: Path


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    agent: AgentConfig
    retrieval: RetrievalConfig
    cost: CostConfig
    server: ServerConfig
    runtime: RuntimeConfig
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def chat_completions_url(self) -> str:
        """Resolve {base}/chat/completions, tolerating a missing /v1 suffix."""
        base = self.llm.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/chat/completions"


def load_config(path: Path | str | None = None) -> Config:
    cfg_path = Path(path) if path else Path(os.getenv("MEHER_CONFIG", DEFAULT_CONFIG_PATH))
    raw = _load_toml(cfg_path)
    llm_t = raw.get("llm", {})
    agent_t = raw.get("agent", {})
    retr_t = raw.get("retrieval", {})
    cost_t = raw.get("cost", {})
    srv_t = raw.get("server", {})
    rt_t = raw.get("runtime", {})

    base_url = os.getenv("LLM_BASE_URL", "").strip() or llm_t.get("base_url", "http://localhost:11434/v1")
    api_key = os.getenv("LLM_API_KEY", "").strip() or llm_t.get("api_key", "ollama")
    model = os.getenv("LLM_MODEL", "").strip() or llm_t.get("model", "qwen2.5:7b-instruct")

    return Config(
        llm=LLMConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=_env_float("MEHER_TEMPERATURE", float(llm_t.get("temperature", 0.0))),
            max_steps=_env_int("MEHER_MAX_STEPS", int(llm_t.get("max_steps", 4))),
            num_ctx=_env_int("MEHER_NUM_CTX", int(llm_t.get("num_ctx", 8192))),
            request_timeout_s=_env_float(
                "MEHER_REQUEST_TIMEOUT_S", float(llm_t.get("request_timeout_s", 120.0))
            ),
            max_retries=_env_int("MEHER_MAX_RETRIES", int(llm_t.get("max_retries", 2))),
            retry_backoff_s=_env_float(
                "MEHER_RETRY_BACKOFF_S", float(llm_t.get("retry_backoff_s", 1.5))
            ),
        ),
        agent=AgentConfig(
            max_reply_chars=_env_int(
                "MEHER_MAX_REPLY_CHARS", int(agent_t.get("max_reply_chars", 1200))
            ),
            min_reply_chars=int(agent_t.get("min_reply_chars", 1)),
            allow_reply_repair=_env_bool(
                "MEHER_ALLOW_REPLY_REPAIR", bool(agent_t.get("allow_reply_repair", True))
            ),
            allow_intent_repair=_env_bool(
                "MEHER_ALLOW_INTENT_REPAIR", bool(agent_t.get("allow_intent_repair", True))
            ),
        ),
        retrieval=RetrievalConfig(
            section_top_k=int(retr_t.get("section_top_k", 4)),
            sku_top_k=int(retr_t.get("sku_top_k", 5)),
            bm25_k1=float(retr_t.get("bm25_k1", 1.5)),
            bm25_b=float(retr_t.get("bm25_b", 0.75)),
            min_section_score=float(retr_t.get("min_section_score", 0.35)),
        ),
        cost=CostConfig(
            inr_per_usd=_env_float("MEHER_INR_PER_USD", float(cost_t.get("inr_per_usd", 87.0))),
            input_cost_per_mtok_usd=float(cost_t.get("input_cost_per_mtok_usd", 0.0)),
            output_cost_per_mtok_usd=float(cost_t.get("output_cost_per_mtok_usd", 0.0)),
        ),
        server=ServerConfig(
            host=os.getenv("MEHER_HOST", srv_t.get("host", "127.0.0.1")),
            port=_env_int("MEHER_PORT", int(srv_t.get("port", 8000))),
        ),
        runtime=RuntimeConfig(
            log_level=os.getenv("MEHER_LOG_LEVEL", rt_t.get("log_level", "INFO")).upper(),
            max_conversations=int(rt_t.get("max_conversations", 5000)),
            max_history_turns=int(rt_t.get("max_history_turns", 12)),
            data_dir=Path(os.getenv("MEHER_DATA_DIR", REPO_ROOT / "data")),
        ),
        raw=raw,
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached process-wide config. Tests should call load_config() directly."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except Exception:  # pragma: no cover - dotenv is optional at runtime
        pass
    return load_config()
