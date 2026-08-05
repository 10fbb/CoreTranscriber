from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from .models import AppSettings


def app_data_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".techtranscriber"))
    path = base / "TechTranscriber"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_root() -> Path:
    path = Path.home() / "Documents" / "ТехСтенограмма"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_cache_dir() -> Path:
    path = app_data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_path() -> Path:
    return app_data_dir() / "settings.json"


def load_settings() -> AppSettings:
    path = settings_path()
    if not path.exists():
        return AppSettings(output_root=default_output_root())
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        output = payload.get("output_root")
        payload["output_root"] = Path(output) if output else default_output_root()
        return AppSettings(**payload)
    except (OSError, ValueError, TypeError):
        return AppSettings(output_root=default_output_root())


def save_settings(settings: AppSettings) -> None:
    payload = asdict(settings)
    payload["output_root"] = str(settings.output_root or default_output_root())
    settings_path().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

