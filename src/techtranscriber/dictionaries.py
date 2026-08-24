from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib import resources
from itertools import zip_longest
from pathlib import Path

from .config import app_data_dir, default_output_root

MAX_PROMPT_TERMS = 300

LEGACY_DEFAULT_HASHES = {
    "01_Общие_IT_и_платформы.txt": "8ca6201a5ee5b6d1fbfffebf506338e34a6469b2793104283baaba96ccef619f",
    "02_API_и_архитектура.txt": "3bf88f701d09091dbdebd703f2f56640f4f4678069b0d0459c50a3293f88c2c6",
    "03_Данные_и_очереди.txt": "5c979624b8479ba2417fbe0910afe9c7fe5f95d35c7deeb70ff0e32286cc227c",
    "04_Cloud_DevOps_и_сети.txt": "6b1bb5629ce7f383f3d85ebbb50eccef39b822b2255393f8c09791f5c0bc1966",
    "05_Безопасность.txt": "d3b61775a7f5373bf007975eccf4742a56aa97491e5c4445ef1ef750822922fb",
    "06_Разработка_и_тестирование.txt": "c6a5864109bcf941af483a87b1636af6d07da2abcfb05695daf4f2971a519076",
    "07_Наблюдаемость_и_системы.txt": "113281c479df942fd4ae79f43b4f378140c485653b012ebd99096ccd02a5c6f5",
    "08_Аудио_и_AI.txt": "40bf4548c5f5b246d1005cc0e5b0a9511a646cd8ac3f34fe1a6daaf5f303d347",
    "09_UI_UX_и_Web.txt": "52daf755a838b589adb650f9ea3cbc5f209da143f8cbfe4e1605cecb74509fce",
    "10_Управление_проектами.txt": "3224abc2ef8769c9cdaaf6844245da72994108ac3128f6a33f39b56b1844a457",
    "11_Эквайринг_и_банковский_процессинг.txt": "47e5f4be1eb0b82c7a2cc2e006ee1e7f527393c6dad424da8524188c11563b42",
}


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
        max_terms: int = MAX_PROMPT_TERMS,
    ) -> list[str]:
        combined: list[str] = []
        seen: set[str] = set()
        def append(term: str) -> bool:
            key = term.casefold()
            if key not in seen:
                seen.add(key)
                combined.append(term)
            return len(combined) >= max_terms

        for term in _clean_terms(extra_terms or []):
            if append(term):
                return combined

        dictionary_terms = [self.load(name) for name in filenames]
        for group in zip_longest(*dictionary_terms):
            for term in group:
                if term is not None and append(term):
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
        marker = app_data_dir() / "dictionaries_v2_initialized"
        if marker.exists():
            return
        source = resources.files("techtranscriber").joinpath("default_dictionaries")
        for item in source.iterdir():
            if item.name.lower().endswith(".txt"):
                target = self.directory / item.name
                if not target.exists():
                    target.write_text(item.read_text(encoding="utf-8"), encoding="utf-8")
                    continue
                existing = target.read_text(encoding="utf-8-sig")
                updated = item.read_text(encoding="utf-8")
                if _normalized_hash(existing) == LEGACY_DEFAULT_HASHES.get(item.name):
                    target.write_text(updated, encoding="utf-8")
                    continue
                merged = _clean_terms(
                    existing.splitlines() + updated.splitlines()
                )
                target.write_text("\n".join(merged) + "\n", encoding="utf-8")
        marker.write_text("2", encoding="ascii")


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


def _normalized_hash(content: str) -> str:
    lines = []
    for raw in content.lstrip("\ufeff").splitlines():
        term = raw.strip()
        if term and not term.startswith("#"):
            lines.append(term)
    normalized = "\n".join(lines) + ("\n" if lines else "")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
