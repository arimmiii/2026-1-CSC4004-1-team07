from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def setup_factcheck_logger(name: str = "domain_fact_check", log_path: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def log_json_event(logger: logging.Logger | None, event: str, payload: dict[str, Any]) -> None:
    if logger is None:
        return
    logger.info("%s | %s", event, json.dumps(payload, ensure_ascii=False))
