"""Tool adapter package.

Import legacy tool modules for their registration side effects so the
historical registry remains populated for adapter consumers.
"""

from src.tools import item_manager as _item_manager  # noqa: F401
from src.tools import state_updater as _state_updater  # noqa: F401

__all__: list[str] = []
