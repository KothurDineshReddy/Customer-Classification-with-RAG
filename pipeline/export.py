"""CSV output writer for classification results."""

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "customer_id",
    "customer_name",
    "address",
    "city",
    "state",
    "zip",
    "national_regional",
    "unit_type",
    "premise",
    "channel",
    "establishment_type",
    "confidence",
    "needs_review",
    "reason",
    "error",
]


def write_output_chunk(
    records: list[dict[str, Any]],
    output_path: Path,
    is_first_chunk: bool,
) -> None:
    """Append *records* to the CSV at *output_path*.

    Writes the header row only when *is_first_chunk* is True so that
    successive chunks merge into a single well-formed CSV.

    Each record dict should contain all OUTPUT_COLUMNS keys; missing keys
    are written as empty strings.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if is_first_chunk else "a"
    with output_path.open(mode, newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=OUTPUT_COLUMNS,
            extrasaction="ignore",
            restval="",
        )
        if is_first_chunk:
            writer.writeheader()
        writer.writerows(records)

    logger.debug("Wrote %d rows to %s (first_chunk=%s)", len(records), output_path, is_first_chunk)
