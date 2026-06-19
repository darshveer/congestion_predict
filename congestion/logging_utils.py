"""Central logging: console + timestamped file in logs/, plus a metrics log.

Usage:
    from congestion.logging_utils import get_logger, log_metrics
    log = get_logger("train")
    log.info("starting ...")
    log_metrics("road_closure", {"auc": 0.81, "pr_auc": 0.285})

All metric rows are also appended as JSON lines to logs/metrics.jsonl so runs are
auditable over time, and mirrored to results.json for the latest snapshot.
"""
from __future__ import annotations
import json
import logging
import os
from datetime import datetime

LOG_DIR = "logs"
METRICS_JSONL = os.path.join(LOG_DIR, "metrics.jsonl")
RESULTS_JSON = "results.json"

_CONFIGURED = set()


def get_logger(name: str = "congestion") -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
                            datefmt="%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    day = datetime.now().strftime("%Y%m%d")
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"run_{day}.log"))
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s %(name)s  %(message)s"))
    logger.addHandler(fh)

    _CONFIGURED.add(name)
    return logger


def log_metrics(section: str, metrics: dict, logger: logging.Logger | None = None):
    """Append one metrics row to logs/metrics.jsonl and echo it to the logger."""
    os.makedirs(LOG_DIR, exist_ok=True)
    row = {"ts": datetime.now().isoformat(timespec="seconds"),
           "section": section, **{k: _round(v) for k, v in metrics.items()}}
    with open(METRICS_JSONL, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    if logger:
        pretty = "  ".join(f"{k}={v}" for k, v in metrics.items())
        logger.info(f"[metrics:{section}] {pretty}")
    return row


def write_results(results: dict):
    """Write the latest full metrics snapshot to results.json."""
    with open(RESULTS_JSON, "w") as fh:
        json.dump(results, fh, indent=2)


def _round(v):
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return v
