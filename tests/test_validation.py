"""Tests for pipeline/validation.py."""

import pytest

from pipeline.validation import ValidationError, validate_csv_schema


def _write_csv(path, content: str):
    path.write_text(content)
    return path


class TestValidateCsvSchema:
    def test_valid_csv_returns_row_count(self, tmp_path):
        p = _write_csv(
            tmp_path / "good.csv",
            "customer_id,customer_name,address,city,state,zip\n"
            "CUST000001,Joe's Grill,123 Main St,Austin,TX,78701\n"
            "CUST000002,HEB,456 Oak Ave,Houston,TX,77001\n",
        )
        assert validate_csv_schema(p) == 2

    def test_missing_column_raises(self, tmp_path):
        p = _write_csv(
            tmp_path / "bad.csv",
            "customer_id,customer_name,city,state,zip\n"  # address missing
            "C1,Joe,Austin,TX,78701\n",
        )
        with pytest.raises(ValidationError, match="address"):
            validate_csv_schema(p)

    def test_multiple_missing_columns_listed(self, tmp_path):
        p = _write_csv(tmp_path / "bad2.csv", "customer_id\nC1\n")
        with pytest.raises(ValidationError, match="missing required columns"):
            validate_csv_schema(p)

    def test_extra_columns_are_allowed(self, tmp_path):
        p = _write_csv(
            tmp_path / "extra.csv",
            "customer_id,customer_name,address,city,state,zip,extra_col\n"
            "C1,Foo,1 A St,NY,NY,10001,whatever\n",
        )
        assert validate_csv_schema(p) == 1

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="not found"):
            validate_csv_schema(tmp_path / "nonexistent.csv")

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_bytes(b"")
        with pytest.raises(ValidationError, match="empty"):
            validate_csv_schema(p)

    def test_header_only_no_data_raises(self, tmp_path):
        p = _write_csv(
            tmp_path / "header_only.csv",
            "customer_id,customer_name,address,city,state,zip\n",
        )
        with pytest.raises(ValidationError, match="no data rows"):
            validate_csv_schema(p)
