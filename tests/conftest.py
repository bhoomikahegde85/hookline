"""Shared test setup."""

from __future__ import annotations

import pytest

from hookline import provenance


@pytest.fixture(autouse=True)
def neutral_interpreter_prefix(monkeypatch):
    """The suite runs in both .venv and .venv-models on temporary fixture roots. The guard that
    refuses Task 2 features in .venv-models is tested explicitly (``prefix=``); elsewhere the
    running interpreter is treated as a neutral environment."""
    monkeypatch.setattr(provenance, "current_prefix", lambda: "/hookline-test-neutral-prefix")
