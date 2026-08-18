from __future__ import annotations

import os
import sys
from typing import TextIO


_NULL_STREAMS: list[TextIO] = []


def configure_windowed_runtime() -> None:
    """Make console-oriented ML libraries safe inside a windowed PyInstaller EXE."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    for attribute in ("stdout", "stderr"):
        if getattr(sys, attribute, None) is not None:
            continue
        stream = open(os.devnull, "w", encoding="utf-8")
        _NULL_STREAMS.append(stream)
        setattr(sys, attribute, stream)
