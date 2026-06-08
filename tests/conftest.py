from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path

import pytest


_TMP_ROOT = Path(__file__).resolve().parent.parent / ".tmp" / "tests"


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return name.strip("._")[:80] or "test"


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    path = _TMP_ROOT / f"{_safe_name(request.node.name)}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
