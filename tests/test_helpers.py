"""Tests for utils/helpers.py"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from utils.helpers import load_json, save_json, truncate, strip_whitespace, now_iso


class TestLoadJson:
    def test_returns_empty_list_when_file_missing(self, tmp_path: Path) -> None:
        result = load_json(tmp_path / "nonexistent.json")
        assert result == []

    def test_loads_list(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        p.write_text('[{"a": 1}]', encoding="utf-8")
        assert load_json(p) == [{"a": 1}]

    def test_loads_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        p.write_text('{"key": "value"}', encoding="utf-8")
        assert load_json(p) == {"key": "value"}


class TestSaveJson:
    def test_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        save_json(p, [1, 2, 3])
        assert p.exists()
        assert json.loads(p.read_text()) == [1, 2, 3]

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "deep" / "nested" / "out.json"
        save_json(p, {"x": 1})
        assert p.exists()

    def test_serialises_datetime(self, tmp_path: Path) -> None:
        p = tmp_path / "dt.json"
        dt = datetime(2026, 1, 1, 12, 0, 0)
        save_json(p, {"ts": dt})
        data = json.loads(p.read_text())
        assert "2026-01-01T12:00:00" in data["ts"]

    def test_raises_for_unserializable(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            save_json(tmp_path / "bad.json", {"obj": object()})


class TestTruncate:
    def test_short_string_unchanged(self) -> None:
        assert truncate("hello", 80) == "hello"

    def test_exact_length_unchanged(self) -> None:
        s = "a" * 80
        assert truncate(s, 80) == s

    def test_long_string_truncated(self) -> None:
        s = "a" * 100
        result = truncate(s, 80)
        assert len(result) == 80
        assert result.endswith("…")

    def test_custom_max_chars(self) -> None:
        result = truncate("hello world", 5)
        assert len(result) == 5
        assert result[-1] == "…"


class TestStripWhitespace:
    def test_collapses_spaces(self) -> None:
        assert strip_whitespace("a   b") == "a b"

    def test_collapses_newlines(self) -> None:
        assert strip_whitespace("a\n\nb") == "a b"

    def test_strips_ends(self) -> None:
        assert strip_whitespace("  hello  ") == "hello"


class TestNowIso:
    def test_returns_string(self) -> None:
        result = now_iso()
        assert isinstance(result, str)
        # Should parse back without error
        datetime.fromisoformat(result)
