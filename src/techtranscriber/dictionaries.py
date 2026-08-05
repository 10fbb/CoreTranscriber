from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .config import app_data_dir, default_output_root


@dataclass(frozen=True, slots=True)
class DictionaryInfo:
    filename: str
    title: str
    path: Path
    term_count: int


class DictionaryManager:
    def __init__(
        self, directory: Path | None = None, install_defaults: bool = True
    ) -> None:
        self.directory = directory or (default_output_root() / "Словари")
        self.directory.mkdir(parents=True, exist_ok=True)
        if install_defaults:
            self._install_defaults_once()

    def list(self) -> list[DictionaryInfo]:
        result: list[DictionaryInfo] = []
        for path in sorted(self.directory.glob("*.txt"), key=lambda item: item.name.casefold()):
            terms = self.load(path.name)
            result.append(
                DictionaryInfo(path.name, self.display_name(path.name), path, len(terms))
            )
        return result

    def load(self, filename: str) -> list[str]:
        path = self._resolve(filename)
        if not path.exists():
            return []
        return _clean_terms(path.read_text(encoding="utf-8").splitlines())

    def read_raw(self, filename: str) -> str:
        path = self._resolve(filename)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def save(self, filename: str, content: str) -> None:
        path = self._resolve(filename)
        terms = _clean_terms(content.splitlines())
        text = "\n".join(terms)
        path.write_text(text + ("\n" if text else ""), encoding="utf-8")

    def create(self, title: str) -> str:
        stem = re.sub(r"[<>:\"/\\|?*]+", "_", title.strip())
        stem = re.sub(r"\s+", "_", stem).strip("._") or "Мой_словарь"
        candidate = f"{stem}.txt"
        index = 2
        while (self.directory / candidate).exists():
            candidate = f"{stem}_{index}.txt"
            index += 1
        (self.directory / candidate).write_text("", encoding="utf-8")
        return candidate

    def delete(self, filename: str) -> None:
        path = self._resolve(filename)
        if path.exists():
            path.unlink()

    def combine(
        self,
        filenames: list[str],
        extra_terms: list[str] | None = None,
        max_terms: int = 1500,
    ) -> list[str]:
        combined: list[str] = []
        seen: set[str] = set()
        sources = [extra_terms or []] + [self.load(name) for name in filenames]
        for terms in sources:
            for term in terms:
                key = term.casefold()
                if key not in seen:
                    seen.add(key)
                    combined.append(term)
                if len(combined) >= max_terms:
                    return combined
        return combined

    @staticmethod
    def display_name(filename: str) -> str:
        stem = Path(filename).stem
        stem = re.sub(r"^\d+_", "", stem)
        return stem.replace("_", " ")

    def _resolve(self, filename: str) -> Path:
        name = Path(filename).name
        if not name.lower().endswith(".txt"):
            name += ".txt"
        return self.directory / name

    def _install_defaults_once(self) -> None:
        marker = app_data_dir() / "dictionaries_v1_initialized"
        if marker.exists():
            return
        source = resources.files("techtranscriber").joinpath("default_dictionaries")
        for item in source.iterdir():
            if item.name.lower().endswith(".txt"):
                target = self.directory / item.name
                if not target.exists():
                    target.write_text(item.read_text(encoding="utf-8"), encoding="utf-8")
        marker.write_text("1", encoding="ascii")


def _clean_terms(lines) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in lines:
        term = str(value).strip()
        if not term or term.startswith("#"):
            continue
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            result.append(term)
    return result
