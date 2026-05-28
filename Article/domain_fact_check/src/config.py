from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def load_project_dotenv() -> None:
    if load_dotenv is None:
        return

    base_dir = Path(__file__).resolve().parents[1]
    candidates = [
        base_dir / ".env",
        base_dir.parent / ".env",
        Path.cwd() / ".env",
    ]

    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            break
