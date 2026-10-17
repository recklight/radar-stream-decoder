"""Wording and privacy checks on every text file.

README.md is English; README.zh-TW.md and the other docs are Traditional
Chinese with Taiwan usage, and so is the viewer's interface text. The two
READMEs are kept in step, so both get the same layout checks. No file may
contain a local path or an e-mail address that slipped in by accident. The
mainland terms below are written as escapes, so searching the repository for
one of them only turns up a real slip, not this list.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
README_ZH = (PROJECT_ROOT / "README.zh-TW.md").read_text(encoding="utf-8")
THIS_FILE = Path(__file__).resolve()

#: Every level-2 heading of the English README, in order.
EXPECTED_H2 = [
    "Contents",
    "Overview",
    "Supported firmware formats and verification",
    "Where TI's documents and firmware disagree",
    "Method",
    "Installation",
    "Quick start",
    "Usage examples",
    "CLI reference",
    "Graphical interface",
    "Project structure",
    "Data preparation",
    "Configuration files",
    "Results and metrics",
    "Known limitations",
    "Design notes",
    "Development and testing",
    "Vendor technical documents",
    "References",
    "License",
]

#: The same headings in the Traditional Chinese README.
EXPECTED_H2_ZH = [
    "目錄",
    "專案簡介",
    "支援的韌體格式與驗證程度",
    "TI 文件與韌體不一致之處",
    "方法說明",
    "安裝步驟",
    "快速開始",
    "使用範例",
    "CLI 指令對照表",
    "圖形介面",
    "專案結構",
    "資料準備",
    "設定檔",
    "結果與評估指標",
    "已知限制",
    "設計重點",
    "開發與測試",
    "廠商公開技術文件",
    "References",
    "License",
]

#: Mainland wording and the Taiwan wording to use instead.
MAINLAND_TERMS: dict[str, str] = {
    "\u6587\u4ef6\u593e": "資料夾",
    "\u5167\u5b58": "記憶體",
    "\u6578\u7d44": "陣列",
    "\u512a\u5316": "最佳化",
    "\u9ed8\u8a8d": "預設",
    "\u7f3a\u7701": "預設",
    "\u5b57\u7b26\u4e32": "字串",
    "\u5206\u8fa8\u7387": "解析度",
    "\u7db2\u7d61": "網路",
    "\u4fe1\u606f": "資訊",
    "\u6a19\u7c3d": "標籤",
    "\u8edf\u4ef6": "軟體",
    "\u786c\u4ef6": "硬體",
    "\u904b\u884c": "執行",
    "\u8abf\u7528": "呼叫",
    "\u5c4f\u5e55": "螢幕",
    "\u754c\u9762": "介面",
    "\u7dda\u7a0b": "執行緒",
    "\u6253\u5370": "列印",
    "\u6a21\u584a": "模組",
    "\u6587\u6a94": "文件",
    "\u8b8a\u91cf": "變數",
    "\u5e38\u91cf": "常數",
    "\u5be6\u73fe": "實作",
    "\u63a1\u6a23": "取樣",
    "\u9b6f\u68d2": "穩健",
    "\u7a0b\u5e8f": "程式",
    "\u4e32\u53e3": "序列埠",
}

#: Patterns that must not appear anywhere in the repository.
FORBIDDEN_PATTERNS: dict[str, str] = {
    "drive-letter path": r"\b[A-Za-z]:[\\/]{1,2}\w[\w .-]*[\\/]",
    "home folder path": r"/(?:home|Users)/[^/\s]+/",
    "e-mail address": r"[\w.+-]+@[\w-]+\.[a-z]{2,}\b",
}

_CJK = re.compile(r"[\u4e00-\u9fff]")
#: Captures, run outputs and environments are not source.
_SKIP_TOP = {".git", ".venv", "venv", "build", "dist", "captures", "outputs"}
_SKIP_ANY = {"__pycache__", ".pytest_cache", ".ruff_cache"}
_TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".cff", ".txt", ".json", ".cfg"}


def text_files() -> list[Path]:
    files = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or path.resolve() == THIS_FILE:
            continue
        parts = path.relative_to(PROJECT_ROOT).parts
        if parts[0] in _SKIP_TOP or _SKIP_ANY & set(parts) or parts[:2] == ("examples", "output"):
            continue
        # pip install -e leaves metadata (with a copy of the README) under src/.
        if any(part.endswith(".egg-info") for part in parts):
            continue
        if path.suffix in _TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore"}:
            files.append(path)
    return files


def test_there_is_something_to_check():
    assert len(text_files()) > 10


@pytest.mark.parametrize("path", text_files(), ids=lambda p: p.name)
def test_no_mainland_wording(path):
    text = path.read_text(encoding="utf-8")
    found = [f"{term} -> {fix}" for term, fix in MAINLAND_TERMS.items() if term in text]
    assert not found, f"{path.name}: " + "; ".join(found)


@pytest.mark.parametrize("path", text_files(), ids=lambda p: p.name)
def test_every_chinese_character_is_traditional(path):
    """Simplified-only characters have no Big5 (cp950) encoding."""
    text = path.read_text(encoding="utf-8")
    bad = []
    for char in sorted(set(_CJK.findall(text))):
        try:
            char.encode("cp950")
        except UnicodeEncodeError:
            bad.append(char)
    assert not bad, f"{path.name}: {''.join(bad)}"


@pytest.mark.parametrize("path", text_files(), ids=lambda p: p.name)
def test_no_local_path_or_e_mail_address(path):
    text = path.read_text(encoding="utf-8")
    found = [label for label, pattern in FORBIDDEN_PATTERNS.items() if re.search(pattern, text)]
    assert not found, f"{path.name}: {found}"


def test_the_patterns_catch_what_they_are_meant_to():
    samples = {
        "drive-letter path": "C:\\Users\\someone\\data",
        "home folder path": "/home/someone/data",
        "e-mail address": "someone@example.org",
    }
    for label, sample in samples.items():
        assert re.search(FORBIDDEN_PATTERNS[label], sample), label
    assert not re.search(FORBIDDEN_PATTERNS["drive-letter path"], "--data-port COM4")


def h2_headings(readme: str) -> list[str]:
    return re.findall(r"^## (.+?)\s*$", readme, re.MULTILINE)


def check_title_badges_and_language_switch(readme: str, language_switch: str) -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    name = re.search(r'^name = "([^"]+)"$', pyproject, re.MULTILINE).group(1)
    lines = readme.splitlines()
    assert lines[0] == f"# {name}"
    assert lines[2].startswith("[![CI](") and lines[3].startswith("[![Python](")
    assert lines[4].startswith("[![License: MIT](")
    assert lines[6] == language_switch


def test_readme_headings_are_in_order():
    assert h2_headings(README) == EXPECTED_H2


def test_chinese_readme_headings_are_in_order():
    assert h2_headings(README_ZH) == EXPECTED_H2_ZH


def test_readme_opens_with_badges_and_author_and_closes_with_the_license():
    check_title_badges_and_language_switch(README, "English | [繁體中文](README.zh-TW.md)")
    assert "\nAuthor: RL\n" in README
    assert README.rstrip().endswith("MIT License, Copyright (c) 2026 RL. See [LICENSE](LICENSE).")


def test_chinese_readme_opens_with_badges_and_author_and_closes_with_the_license():
    check_title_badges_and_language_switch(README_ZH, "[English](README.md) | 繁體中文")
    assert "\n作者：RL\n" in README_ZH
    closing = "MIT License，Copyright (c) 2026 RL。詳見 [LICENSE](LICENSE)。"
    assert README_ZH.rstrip().endswith(closing)


def test_licence_year_and_holder_agree():
    licence = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Copyright (c) 2026 RL" in licence


def test_readme_is_mostly_english_prose():
    # The only Chinese in it is the link to the Chinese README. The Chinese
    # README has about 1,700 lowercase English words of three or more letters
    # (code, field names, references), so a floor of 4,000 tells the two apart.
    assert len(_CJK.findall(README)) < 20
    assert len(re.findall(r"\b[a-z]{3,}\b", README)) > 4000


def test_chinese_readme_is_mostly_chinese_prose():
    assert len(_CJK.findall(README_ZH)) > 5000


def test_contents_links_match_the_headings():
    # GitHub anchors: lower case, punctuation dropped, spaces turned into hyphens.
    def anchor(heading: str) -> str:
        return re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")

    for readme, headings in ((README, EXPECTED_H2), (README_ZH, EXPECTED_H2_ZH)):
        links = re.findall(r"^- \[[^\]]+\]\(#([^)]+)\)$", readme, re.MULTILINE)
        assert links == [anchor(h) for h in headings[1:]]


@pytest.mark.parametrize(
    ("readme", "suffix"),
    [(README, ""), (README_ZH, ".zh-TW")],
    ids=["README.md", "README.zh-TW.md"],
)
def test_each_readme_shows_the_diagram_in_its_own_language(readme, suffix):
    # The figure, the online page and the local page all have to follow the
    # README's language; the English files carry no suffix.
    assert f"](docs/images/workflow{suffix}.png)" in readme
    assert f"/master/docs/workflow{suffix}.html?theme=dark)" in readme
    assert f"](docs/workflow{suffix}.html)" in readme


@pytest.mark.parametrize("readme", [README, README_ZH], ids=["README.md", "README.zh-TW.md"])
def test_relative_links_point_to_files_that_exist(readme):
    targets = re.findall(r"\]\(([^)#\s]+)\)", readme)
    missing = [t for t in targets if "://" not in t and not (PROJECT_ROOT / t).is_file()]
    assert not missing, missing


@pytest.mark.parametrize("readme", [README, README_ZH], ids=["README.md", "README.zh-TW.md"])
def test_readme_test_command_needs_no_pythonpath(readme):
    # pytest takes pythonpath = ["src"] from pyproject.toml, so the documented
    # command is plain "python -m pytest".
    assert "python -m pytest -q" in readme
    assert "PYTHONPATH=src python -m pytest" not in readme
