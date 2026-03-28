"""
游戏状态持久化 — JSON 文件存储。

Phase 1 使用简单的 JSON 文件作为状态后端，
Phase 3 将升级为 PostgreSQL。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings
from src.models.game_state import GameState


class GameStateStore:
    """
    基于 JSON 文件的游戏状态持久化。

    每个 session_id 对应一个 JSON 文件。
    """

    def __init__(self):
        self._state_dir = Path(settings.game.state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"[GameStateStore] 存储目录: {self._state_dir}")

    def _get_path(self, session_id: str) -> Path:
        return self._state_dir / f"{session_id}.json"

    def save(self, state: GameState) -> None:
        """保存游戏状态到 JSON 文件。"""
        path = self._get_path(state.session_id)
        path.write_text(
            state.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.debug(f"[GameStateStore] 已保存: {path.name}")

    def load(self, session_id: str) -> Optional[GameState]:
        """从 JSON 文件加载游戏状态。"""
        path = self._get_path(session_id)
        if not path.exists():
            logger.debug(f"[GameStateStore] 存档不存在: {path.name}")
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            state = GameState(**data)
            logger.debug(f"[GameStateStore] 已加载: {path.name}")
            return state
        except Exception as e:
            logger.error(f"[GameStateStore] 加载失败: {e}")
            return None

    def load_or_create(self, session_id: str) -> GameState:
        """加载已有存档或创建新的游戏状态。"""
        state = self.load(session_id)
        if state is None:
            state = GameState(session_id=session_id)
            self.save(state)
            logger.info(f"[GameStateStore] 创建新存档: {session_id}")
        return state

    def delete(self, session_id: str) -> bool:
        """删除存档。"""
        path = self._get_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False
