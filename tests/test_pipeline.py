"""E2E pipeline test using mock providers."""

import csv
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from pipeline.classification import MockLLMProvider
from pipeline.retrieval import MockEmbeddingProvider
from pipeline.runner import run_pipeline

TAXONOMY_YAML = Path("data/taxonomy/v1.yaml")

VALID_CHANNELS = {
    "Restaurant",
    "Bar",
    "Hotel",
    "Nightclub",
    "Cafe",
    "Catering",
    "Stadium",
    "Liquor Store",
    "Grocery",
    "Convenience",
    "Warehouse Club",
    "Drug Store",
    "Online Retail",
}
VALID_NATIONAL_REGIONAL = {"National", "Regional"}
VALID_UNIT_TYPES = {"Multi-unit", "Single-unit"}
VALID_PREMISES = {"On-Premise", "Off-Premise"}


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory) -> Path:
    """Generate a 50-row sample CSV covering diverse customer types."""
    rows = [
        ("CUST000001", "Joe's Bar & Grill", "100 Main St", "Austin", "TX", "78701"),
        ("CUST000002", "Total Wine & More", "200 Oak Ave", "Dallas", "TX", "75201"),
        ("CUST000003", "Hilton Garden Inn Dallas", "300 Park Blvd", "Dallas", "TX", "75202"),
        ("CUST000004", "Kroger #4521", "400 Elm Dr", "Houston", "TX", "77001"),
        ("CUST000005", "7-Eleven Store #1234", "500 Pine St", "San Antonio", "TX", "78201"),
        ("CUST000006", "Mama Rosa's Pizzeria", "601 Cedar Ln", "Chicago", "IL", "60601"),
        ("CUST000007", "Austin Brewing Co. Taproom", "702 Walnut Way", "Austin", "TX", "78702"),
        ("CUST000008", "Walgreens", "803 Maple Ave", "Denver", "CO", "80201"),
        ("CUST000009", "Outback Steakhouse", "904 Chestnut Pkwy", "Phoenix", "AZ", "85001"),
        ("CUST000010", "Sam's Club #842", "1005 Birch Rd", "Las Vegas", "NV", "89101"),
        ("CUST000011", "Sunset Lounge", "1106 Spruce Ct", "Miami", "FL", "33101"),
        ("CUST000012", "HEB #307", "1207 Hickory Pl", "San Antonio", "TX", "78202"),
        ("CUST000013", "Marriott Houston Downtown", "1308 Willow Pkwy", "Houston", "TX", "77002"),
        ("CUST000014", "Spec's Liquor", "1409 Magnolia Blvd", "Houston", "TX", "77003"),
        ("CUST000015", "Texas Smokehouse BBQ", "1510 Peach Dr", "Nashville", "TN", "37201"),
        ("CUST000016", "Corner Bistro", "1611 Cherry Ave", "Seattle", "WA", "98101"),
        ("CUST000017", "CVS Pharmacy", "1712 Valley Rd", "Portland", "OR", "97201"),
        ("CUST000018", "Chili's", "1813 Summit St", "Atlanta", "GA", "30301"),
        ("CUST000019", "Downtown Food Market", "1914 Harbor Ln", "Baltimore", "MD", "21201"),
        ("CUST000020", "The Draft House", "2015 Bay Ct", "Boston", "MA", "02101"),
        ("CUST000021", "Papa Garcia's Kitchen", "2116 Lake Blvd", "Los Angeles", "CA", "90001"),
        ("CUST000022", "Costco Wholesale #789", "2217 River Pkwy", "San Diego", "CA", "92101"),
        ("CUST000023", "BevMo", "2318 Forest Ave", "Sacramento", "CA", "95801"),
        ("CUST000024", "Courtyard by Marriott Chicago", "2419 Meadow Dr", "Chicago", "IL", "60602"),
        ("CUST000025", "Applebee's", "2520 Garden Way", "Minneapolis", "MN", "55401"),
        ("CUST000026", "Smith's Liquor Store", "2621 Green St", "Las Vegas", "NV", "89102"),
        ("CUST000027", "Sushi Roku", "2722 Willow Ave", "Los Angeles", "CA", "90002"),
        ("CUST000028", "Speedway #5512", "2823 Maple Ln", "Columbus", "OH", "43201"),
        ("CUST000029", "Café Lumière", "2924 Oak Ct", "New York", "NY", "10001"),
        ("CUST000030", "Wingstop", "3025 Pine Rd", "Charlotte", "NC", "28201"),
        ("CUST000031", "La Quinta Inn & Suites", "3126 Elm Dr", "Orlando", "FL", "32801"),
        ("CUST000032", "Rite Aid", "3227 Cedar Blvd", "Philadelphia", "PA", "19101"),
        ("CUST000033", "Old Town Tavern", "3328 Birch St", "Denver", "CO", "80202"),
        ("CUST000034", "Whole Foods Market", "3429 Chestnut Ave", "San Francisco", "CA", "94101"),
        ("CUST000035", "Hampton Inn Nashville", "3530 Walnut Pkwy", "Nashville", "TN", "37202"),
        ("CUST000036", "Carlo's Seafood Grille", "3631 Spruce Way", "Seattle", "WA", "98102"),
        ("CUST000037", "Kwik Trip #4401", "3732 Magnolia Rd", "Milwaukee", "WI", "53201"),
        ("CUST000038", "Panera Bread", "3833 Peach Ln", "Indianapolis", "IN", "46201"),
        ("CUST000039", "Miller's Taproom", "3934 Cherry St", "Portland", "OR", "97202"),
        ("CUST000040", "Safeway #2201", "4035 Valley Blvd", "Phoenix", "AZ", "85002"),
        (
            "CUST000041",
            "The Rusty Nail Cocktail Bar",
            "4136 Harbor Dr",
            "New Orleans",
            "LA",
            "70101",
        ),
        ("CUST000042", "Total Wine & More #88", "4237 Bay Ave", "Tampa", "FL", "33601"),
        ("CUST000043", "Doubletree by Hilton Seattle", "4338 Lake Ct", "Seattle", "WA", "98103"),
        ("CUST000044", "Casa Martinez", "4439 River Way", "San Antonio", "TX", "78203"),
        ("CUST000045", "Circle K #9901", "4540 Forest Pkwy", "Las Vegas", "NV", "89103"),
        ("CUST000046", "Trader Joe's", "4641 Meadow St", "Boston", "MA", "02102"),
        ("CUST000047", "Down the Hatch", "4742 Garden Rd", "Austin", "TX", "78703"),
        ("CUST000048", "Raising Cane's", "4843 Green Ave", "Houston", "TX", "77004"),
        ("CUST000049", "Neighborhood Catering Co.", "4944 Willow Ln", "Chicago", "IL", "60603"),
        ("CUST000050", "Joe's Place", "5045 Oak Blvd", "Dallas", "TX", "75203"),
    ]

    p = tmp_path_factory.mktemp("data") / "sample_50.csv"
    with p.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["customer_id", "customer_name", "address", "city", "state", "zip"])
        writer.writerows(rows)
    return p


@pytest.fixture(scope="module")
def pipeline_output(sample_csv, tmp_path_factory) -> Path:
    """Run the full pipeline once and return the output CSV path."""
    out = tmp_path_factory.mktemp("out") / "results.csv"
    mock_tracker = MagicMock()

    run_pipeline(
        job_id="test-job-001",
        input_path=sample_csv,
        output_path=out,
        taxonomy_yaml=TAXONOMY_YAML,
        confidence_threshold=0.75,
        job_tracker=mock_tracker,
        embedding_provider=MockEmbeddingProvider(),
        llm_provider=MockLLMProvider(),
    )
    return out


class TestE2EPipeline:
    def test_output_file_created(self, pipeline_output):
        assert pipeline_output.exists()
        assert pipeline_output.stat().st_size > 0

    def test_output_row_count(self, pipeline_output):
        df = pd.read_csv(pipeline_output)
        assert len(df) == 50

    def test_all_expected_columns_present(self, pipeline_output):
        df = pd.read_csv(pipeline_output)
        required = {
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
        }
        assert required.issubset(set(df.columns))

    def test_confidence_between_0_and_1(self, pipeline_output):
        df = pd.read_csv(pipeline_output)
        # Exclude failed rows (confidence == 0.0 with an error)
        successful = df[df["error"].fillna("") == ""]
        assert (successful["confidence"] >= 0.0).all()
        assert (successful["confidence"] <= 1.0).all()

    def test_all_successful_rows_have_valid_channel(self, pipeline_output):
        df = pd.read_csv(pipeline_output)
        successful = df[df["error"].fillna("") == ""]
        invalid = successful[~successful["channel"].isin(VALID_CHANNELS)]
        assert len(invalid) == 0, (
            f"Invalid channels: {invalid[['customer_name', 'channel']].to_dict()}"
        )

    def test_national_regional_values(self, pipeline_output):
        df = pd.read_csv(pipeline_output)
        successful = df[df["error"].fillna("") == ""]
        assert successful["national_regional"].isin(VALID_NATIONAL_REGIONAL).all()

    def test_unit_type_values(self, pipeline_output):
        df = pd.read_csv(pipeline_output)
        successful = df[df["error"].fillna("") == ""]
        assert successful["unit_type"].isin(VALID_UNIT_TYPES).all()

    def test_premise_values(self, pipeline_output):
        df = pd.read_csv(pipeline_output)
        successful = df[df["error"].fillna("") == ""]
        assert successful["premise"].isin(VALID_PREMISES).all()

    def test_needs_review_is_boolean_compatible(self, pipeline_output):
        df = pd.read_csv(pipeline_output)
        assert df["needs_review"].isin([True, False]).all()

    def test_needs_review_matches_threshold(self, pipeline_output):
        df = pd.read_csv(pipeline_output)
        successful = df[df["error"].fillna("") == ""]
        low_conf = successful[successful["confidence"] < 0.75]
        high_conf = successful[successful["confidence"] >= 0.75]
        assert low_conf["needs_review"].all(), "Low-confidence rows should need review"
        assert not high_conf["needs_review"].any(), "High-confidence rows should not need review"

    def test_reason_field_populated(self, pipeline_output):
        df = pd.read_csv(pipeline_output)
        successful = df[df["error"].fillna("") == ""]
        assert (successful["reason"].str.len() > 0).all()

    def test_job_tracker_called(self, sample_csv, tmp_path):
        out = tmp_path / "tracker_test.csv"
        tracker = MagicMock()
        run_pipeline(
            job_id="tracker-test",
            input_path=sample_csv,
            output_path=out,
            taxonomy_yaml=TAXONOMY_YAML,
            confidence_threshold=0.75,
            job_tracker=tracker,
            embedding_provider=MockEmbeddingProvider(),
            llm_provider=MockLLMProvider(),
        )
        tracker.set_progress.assert_called()
        tracker.set_completed.assert_called_once_with("tracker-test")
        tracker.set_failed.assert_not_called()

    def test_channel_diversity(self, pipeline_output):
        """The 50-row sample should produce at least 5 distinct channels."""
        df = pd.read_csv(pipeline_output)
        successful = df[df["error"].fillna("") == ""]
        n_channels = successful["channel"].nunique()
        assert n_channels >= 5, (
            f"Expected ≥5 channels, got {n_channels}: {successful['channel'].unique()}"
        )

    def test_off_premise_accounts_present(self, pipeline_output):
        df = pd.read_csv(pipeline_output)
        successful = df[df["error"].fillna("") == ""]
        off_premise = successful[successful["premise"] == "Off-Premise"]
        assert len(off_premise) > 0

    def test_on_premise_accounts_present(self, pipeline_output):
        df = pd.read_csv(pipeline_output)
        successful = df[df["error"].fillna("") == ""]
        on_premise = successful[successful["premise"] == "On-Premise"]
        assert len(on_premise) > 0
