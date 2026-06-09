"""Input CSV validation."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"customer_id", "customer_name", "address", "city", "state", "zip"}


class ValidationError(Exception):
    """Raised when the input CSV fails schema checks."""


def validate_csv_schema(path: Path) -> int:
    """Validate that the CSV at *path* has the required columns and at least one row.

    Returns the number of data rows on success.
    Raises ValidationError with a descriptive message on failure.
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"Input file not found: {path}")
    if path.stat().st_size == 0:
        raise ValidationError(f"Input file is empty: {path}")

    try:
        df = pd.read_csv(path, nrows=0)
    except Exception as exc:
        raise ValidationError(f"Cannot parse CSV: {exc}") from exc

    actual = set(df.columns.str.strip().str.lower())
    missing = REQUIRED_COLUMNS - actual
    if missing:
        raise ValidationError(
            f"CSV is missing required columns: {sorted(missing)}. Found: {sorted(actual)}"
        )

    # Count data rows without loading entire file
    with path.open() as fh:
        n_rows = sum(1 for _ in fh) - 1  # subtract header

    if n_rows < 1:
        raise ValidationError(f"Input file has no data rows: {path}")

    logger.info("Validated %s — %d rows, required columns present", path.name, n_rows)
    return n_rows
