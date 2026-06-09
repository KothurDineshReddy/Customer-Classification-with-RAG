"""Tests for pipeline/preprocessing.py."""

from pipeline.preprocessing import clean_record


class TestCleanRecord:
    def _clean(self, **kwargs) -> dict:
        base = {
            "customer_id": "C1",
            "customer_name": kwargs.pop("customer_name", "Test Co."),
            "address": kwargs.pop("address", "123 Main St"),
            "city": kwargs.pop("city", "austin"),
            "state": kwargs.pop("state", "tx"),
            "zip": kwargs.pop("zip", "78701"),
        }
        base.update(kwargs)
        return clean_record(base)

    # --- customer_name ---

    def test_strips_llc(self):
        r = self._clean(customer_name="Joe's Bar LLC")
        assert r["customer_name"] == "Joe's Bar"

    def test_strips_inc(self):
        r = self._clean(customer_name="Big Drinks Inc.")
        assert r["customer_name"] == "Big Drinks"

    def test_strips_corp(self):
        r = self._clean(customer_name="MegaCorp Corp.")
        assert r["customer_name"] == "MegaCorp"

    def test_strips_ltd(self):
        r = self._clean(customer_name="Fine Wines Ltd")
        assert r["customer_name"] == "Fine Wines"

    def test_strips_llp(self):
        r = self._clean(customer_name="Partners LLP")
        assert r["customer_name"] == "Partners"

    def test_preserves_name_without_suffix(self):
        r = self._clean(customer_name="Mama Rosa's Pizzeria")
        assert r["customer_name"] == "Mama Rosa's Pizzeria"

    def test_collapses_extra_whitespace_in_name(self):
        r = self._clean(customer_name="  Joe's   Grill  ")
        assert r["customer_name"] == "Joe's Grill"

    def test_strips_trailing_comma_after_suffix_removal(self):
        r = self._clean(customer_name="Drinks Co.,")
        # "Co." is a suffix; trailing comma cleaned up
        assert not r["customer_name"].endswith(",")

    # --- address ---

    def test_normalises_ste_to_suite(self):
        r = self._clean(address="100 Oak Ave, Ste 200")
        assert "Suite 200" in r["address"]

    def test_normalises_hash_to_suite(self):
        r = self._clean(address="500 Elm St #101")
        assert "Suite 101" in r["address"]

    def test_normalises_unit(self):
        r = self._clean(address="200 Pine Rd Unit B")
        assert "Suite B" in r["address"]

    def test_strips_address_whitespace(self):
        r = self._clean(address="  123 Main St  ")
        assert r["address"] == "123 Main St"

    # --- city / state ---

    def test_city_title_cased(self):
        r = self._clean(city="new york")
        assert r["city"] == "New York"

    def test_state_uppercased(self):
        r = self._clean(state="tx")
        assert r["state"] == "TX"

    def test_zip_stripped(self):
        r = self._clean(zip=" 78701 ")
        assert r["zip"] == "78701"

    # --- passthrough ---

    def test_non_string_passthrough(self):
        record = {
            "customer_id": "C1",
            "customer_name": "Test",
            "address": "1 A St",
            "city": "LA",
            "state": "CA",
            "zip": "90001",
            "extra_int": 42,
        }
        result = clean_record(record)
        assert result["extra_int"] == 42
