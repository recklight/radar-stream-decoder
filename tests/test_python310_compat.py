"""The Python 3.10 gate.

The package declares ``requires-python = ">=3.10"``. Running the suite on 3.10
only covers the lines the tests reach, so every module is also parsed with the
3.10 grammar, and executable code (not comments or strings) is scanned for
standard-library names that only exist in 3.11 or later and for the NumPy
aliases removed in 2.0.
"""

from __future__ import annotations

import ast
import io
import os
import re
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROSE_TOKENS = frozenset({"COMMENT", "STRING", "FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"})

FORBIDDEN_STDLIB: dict[str, str] = {
    "tomllib": "a JSON or YAML reader (3.11+)",
    "typing.Self": "the class name as a string (3.11+)",
    "LiteralString": "str (3.11+)",
    "assert_never": "raise AssertionError (3.11+)",
    "typing.Unpack": "explicit parameters (3.11+)",
    "TypeVarTuple": "a plain TypeVar (3.11+)",
    "StrEnum": "class Foo(str, Enum) (3.11+)",
    "ReprEnum": "Enum (3.11+)",
    "datetime.UTC": "datetime.timezone.utc (3.11+)",
    "TaskGroup": "asyncio.gather (3.11+)",
    "ExceptionGroup": "a single exception (3.11+)",
    "contextlib.chdir": "os.chdir in try/finally (3.11+)",
    "file_digest": "hash the file in chunks (3.11+)",
    "math.cbrt": "x ** (1/3) (3.11+)",
    "getmembers_static": "inspect.getmembers (3.11+)",
    "typing.override": "a plain method (3.12+)",
    "itertools.batched": "an explicit slice loop (3.12+)",
    "Path.walk": "os.walk (3.12+)",
    "ReadOnly": "a plain annotation (3.13+)",
    "copy.replace": "dataclasses.replace (3.13+)",
}

FORBIDDEN_NUMPY: dict[str, str] = {
    "np.float_": "np.float64",
    "np.int0": "np.intp",
    "np.NaN": "np.nan",
    "np.Inf": "np.inf",
    "np.product": "np.prod",
    "np.alltrue": "np.all",
    "np.round_": "np.round",
    "np.msort": "np.sort",
    "np.trapz": "an explicit sum (renamed in NumPy 2.0)",
}


def code_only(source: str) -> str:
    """Strip comments and string literals so documentation is not mistaken for code."""
    pieces = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if tokenize.tok_name.get(token.type, "") not in _PROSE_TOKENS:
            pieces.append(token.string)
    return "\n".join(pieces)


def python_files() -> list[Path]:
    files = sorted(PROJECT_ROOT.glob("*.py"))
    for folder in ("src", "tests", "examples"):
        files.extend(sorted((PROJECT_ROOT / folder).rglob("*.py")))
    return [path for path in files if "__pycache__" not in path.parts]


def files_to_scan() -> list[Path]:
    """Every module except this one, which spells out the forbidden names on purpose."""
    this_file = Path(__file__).resolve()
    return [path for path in python_files() if path.resolve() != this_file]


def test_there_is_something_to_check():
    assert len(python_files()) > 40
    assert len(files_to_scan()) == len(python_files()) - 1


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_every_module_parses_under_the_python_310_grammar(path: Path):
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
    except SyntaxError as exc:
        pytest.fail(f"{path.name} does not parse as Python 3.10: {exc.msg} (line {exc.lineno})")


@pytest.mark.parametrize("path", files_to_scan(), ids=lambda p: p.name)
def test_no_standard_library_name_newer_than_310(path: Path):
    source = code_only(path.read_text(encoding="utf-8"))
    offences = [
        f"{name} -> {fix}"
        for name, fix in FORBIDDEN_STDLIB.items()
        if re.search(rf"(?<![\w]){re.escape(name)}\b", source)
    ]
    assert not offences, f"{path.name}: " + "; ".join(offences)


@pytest.mark.parametrize("path", files_to_scan(), ids=lambda p: p.name)
def test_no_numpy_alias_removed_in_2_0(path: Path):
    source = code_only(path.read_text(encoding="utf-8"))
    offences = [
        f"{name} -> {fix}"
        for name, fix in FORBIDDEN_NUMPY.items()
        if re.search(rf"(?<![\w.]){re.escape(name)}\b", source)
    ]
    assert not offences, f"{path.name}: " + "; ".join(offences)


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_no_pep695_generics_or_type_aliases(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        assert getattr(node, "type_params", []) == [], f"{path.name}: PEP 695 type parameters"
        assert type(node).__name__ != "TypeAlias", f"{path.name}: 'type X = ...' statement"


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_every_module_defers_annotation_evaluation(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    ), f"{path.name} lacks 'from __future__ import annotations'"


@pytest.mark.parametrize("path", sorted((PROJECT_ROOT / "src").rglob("*.py")), ids=lambda p: p.name)
def test_no_frozen_slotted_dataclasses(path: Path):
    """Frozen dataclasses with slots do not unpickle on early 3.10 releases."""
    source = path.read_text(encoding="utf-8")
    assert not re.search(r"@dataclass\([^)]*frozen=True[^)]*slots=True", source)
    assert not re.search(r"@dataclass\([^)]*slots=True[^)]*frozen=True", source)


def test_pyproject_targets_python_310_with_reachable_floors():
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10"' in text
    assert 'target-version = "py310"' in text
    for requirement in ("numpy>=1.24", "matplotlib>=3.6", "pandas>=1.5", "pyserial>=3.5"):
        assert requirement in text, requirement
    assert "numpy>=2" not in text
    assert 'authors = [{ name = "RL" }]' in text


def test_package_imports_without_optional_dependencies():
    """Importing the package must not pull in the GUI, pandas or pyserial."""
    probe = (
        "import sys, radar_stream_decoder, radar_stream_decoder.cli;"
        "heavy = [m for m in ('PySide6', 'matplotlib', 'pandas', 'serial', 'torch', 'scipy')"
        " if m in sys.modules];"
        "print(','.join(heavy))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src")),
        check=True,
    )
    assert completed.stdout.strip() == ""
    import radar_stream_decoder

    assert radar_stream_decoder.__version__ == "0.1.0"
