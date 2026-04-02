"""OmniNPC 全局配置 — 基于 Pydantic Settings，自动读取 .env 文件。"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 先将 .env 加载到进程环境变量中，
# 这样所有嵌套的 BaseSettings 子类（如 LLMSettings）也能正确读取
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _default_chroma_persist_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "OmniNPC" / "chroma_db"
    return PROJECT_ROOT / "data" / "chroma_db"


class LLMSettings(BaseSettings):
    """大模型调用配置。"""

    model_config = SettingsConfigDict(env_prefix="ARK_")

    # 火山方舟
    api_key: str = ""
    model_endpoint: str = ""  # 推理接入点 Endpoint ID
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3"

    # 生成参数
    temperature: float = 0.8
    max_tokens: int = 1024
    top_p: float = 0.9


class MemorySettings(BaseSettings):
    """记忆系统配置。"""

    model_config = SettingsConfigDict(env_prefix="OMNI_NPC_")

    # 工作记忆：保留最近 N 轮对话
    working_memory_window: int = 10

    # 情景记忆：向量检索返回 top-K
    episodic_top_k: int = 5

    # ChromaDB 存储路径
    chroma_persist_dir: str = Field(
        default_factory=lambda: str(_default_chroma_persist_dir()),
    )

    # 向量化模型（ChromaDB 默认使用 all-MiniLM-L6-v2）
    embedding_model: str = "all-MiniLM-L6-v2"


class GameSettings(BaseSettings):
    """游戏状态配置。"""

    # 状态持久化路径
    state_dir: str = str(PROJECT_ROOT / "data" / "state")


class SandboxSettings(BaseSettings):
    """离线 Tick 引擎配置。"""

    tick_interval_seconds: int = 300  # 5 分钟一次 tick
    enable_rumor_spread: bool = True
    rumor_spread_probability: float = 0.3


class Settings(BaseSettings):
    """顶层配置聚合。"""

    model_config = SettingsConfigDict(
        env_prefix="OMNI_NPC_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 应用
    app_name: str = "OmniNPC"
    debug: bool = False
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("OMNI_NPC_LOG_LEVEL", "LOG_LEVEL"),
    )

    # 子配置
    llm: LLMSettings = LLMSettings()
    memory: MemorySettings = MemorySettings()
    game: GameSettings = GameSettings()
    sandbox: SandboxSettings = SandboxSettings()

    # 角色配置目录
    characters_dir: str = str(PROJECT_ROOT / "data" / "characters")


# 全局单例
settings = Settings()
