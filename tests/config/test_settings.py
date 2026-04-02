import importlib
from pathlib import Path
import sys


def _reload_settings_module():
    sys.modules.pop("config.settings", None)
    return importlib.import_module("config.settings")


def test_settings_ignore_generic_debug_environment_variable(monkeypatch):
    monkeypatch.setenv("DEBUG", "release")
    monkeypatch.delenv("OMNI_NPC_DEBUG", raising=False)

    settings_module = _reload_settings_module()

    assert settings_module.settings.debug is False


def test_settings_use_namespaced_debug_environment_variable(monkeypatch):
    monkeypatch.setenv("DEBUG", "release")
    monkeypatch.setenv("OMNI_NPC_DEBUG", "true")

    settings_module = _reload_settings_module()

    assert settings_module.settings.debug is True


def test_memory_settings_default_to_local_app_data_on_windows(monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\lenovo\AppData\Local")
    monkeypatch.delenv("OMNI_NPC_CHROMA_PERSIST_DIR", raising=False)

    settings_module = _reload_settings_module()

    assert settings_module.settings.memory.chroma_persist_dir == str(
        Path(r"C:\Users\lenovo\AppData\Local") / "OmniNPC" / "chroma_db"
    )
