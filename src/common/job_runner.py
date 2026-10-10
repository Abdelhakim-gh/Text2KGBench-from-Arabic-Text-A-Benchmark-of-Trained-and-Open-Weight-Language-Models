"""Resumable JSONL job runner.

Writes one result per line to disk as it goes. Stopping (crash, kill, laptop
sleep) and restarting skips whatever already has a line in the output file,
identified by `item_id`. This is the mechanism that makes multi-hour LLM runs
safe to interrupt.
"""

import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed = set()
    with open(output_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A crash mid-write can leave a truncated final line. Skip it;
                # that item_id is treated as not-yet-completed and redone.
                logger.warning("%s: skipping unparseable line %d (likely a truncated write)", output_path, line_num)
                continue
            completed.add(row["item_id"])
    return completed


def run_job(
    items: Iterable[dict],
    worker: Callable[[dict], dict[str, Any]],
    output_path: Path,
    item_id_key: str = "item_id",
    on_error: str = "record",  # "record": write an error row and continue. "raise": stop the run.
) -> None:
    """Runs `worker(item)` for every item in `items` not already present in
    `output_path`, appending one JSON line per result as it completes.

    Each item dict must contain `item_id_key`. `worker` should return a dict;
    `item_id` is added to it automatically before writing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(output_path)
    logger.info("%s: %d already completed, resuming", output_path, len(completed))

    pending = [item for item in items if item[item_id_key] not in completed]
    logger.info("%s: %d items to run", output_path, len(pending))

    with open(output_path, "a", encoding="utf-8", buffering=1) as f:  # line-buffered
        for i, item in enumerate(pending, start=1):
            item_id = item[item_id_key]
            try:
                result = worker(item)
            except Exception as exc:
                if on_error == "raise":
                    raise
                logger.error("%s: item %s failed: %s", output_path, item_id, exc)
                result = {"error": str(exc)}

            result["item_id"] = item_id
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            if i % 50 == 0 or i == len(pending):
                logger.info("%s: %d/%d done this run", output_path, i, len(pending))
