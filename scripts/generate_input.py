"""Generate synthetic customer CSVs for pipeline testing."""

import argparse
import csv
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Seeded RNG wrapper so callers get deterministic output for a given n_rows
# ---------------------------------------------------------------------------

RNG = random.Random(0)


# ---------------------------------------------------------------------------
# US cities with realistic zip code prefixes and state abbreviations
# ---------------------------------------------------------------------------

CITIES: list[tuple[str, str, str]] = [
    # (city, state, zip_prefix)
    ("New York", "NY", "100"),
    ("Brooklyn", "NY", "112"),
    ("Queens", "NY", "113"),
    ("Buffalo", "NY", "142"),
    ("Los Angeles", "CA", "900"),
    ("San Francisco", "CA", "941"),
    ("San Diego", "CA", "921"),
    ("Sacramento", "CA", "958"),
    ("Chicago", "IL", "606"),
    ("Springfield", "IL", "627"),
    ("Houston", "TX", "770"),
    ("Dallas", "TX", "752"),
    ("Austin", "TX", "787"),
    ("San Antonio", "TX", "782"),
    ("Fort Worth", "TX", "761"),
    ("Phoenix", "AZ", "850"),
    ("Tucson", "AZ", "857"),
    ("Philadelphia", "PA", "191"),
    ("Pittsburgh", "PA", "152"),
    ("Miami", "FL", "331"),
    ("Orlando", "FL", "328"),
    ("Tampa", "FL", "336"),
    ("Jacksonville", "FL", "322"),
    ("Atlanta", "GA", "303"),
    ("Savannah", "GA", "314"),
    ("Seattle", "WA", "981"),
    ("Spokane", "WA", "992"),
    ("Denver", "CO", "802"),
    ("Colorado Springs", "CO", "809"),
    ("Boston", "MA", "021"),
    ("Worcester", "MA", "016"),
    ("Detroit", "MI", "482"),
    ("Grand Rapids", "MI", "495"),
    ("Minneapolis", "MN", "554"),
    ("St. Paul", "MN", "551"),
    ("Nashville", "TN", "372"),
    ("Memphis", "TN", "381"),
    ("Portland", "OR", "972"),
    ("Eugene", "OR", "974"),
    ("Las Vegas", "NV", "891"),
    ("Reno", "NV", "895"),
    ("Charlotte", "NC", "282"),
    ("Raleigh", "NC", "276"),
    ("Columbus", "OH", "432"),
    ("Cleveland", "OH", "441"),
    ("Cincinnati", "OH", "452"),
    ("Indianapolis", "IN", "462"),
    ("Louisville", "KY", "402"),
    ("Lexington", "KY", "405"),
    ("New Orleans", "LA", "701"),
    ("Baton Rouge", "LA", "708"),
    ("Baltimore", "MD", "212"),
    ("Annapolis", "MD", "214"),
    ("Richmond", "VA", "232"),
    ("Virginia Beach", "VA", "234"),
    ("Kansas City", "MO", "641"),
    ("St. Louis", "MO", "631"),
    ("Oklahoma City", "OK", "731"),
    ("Tulsa", "OK", "741"),
    ("Albuquerque", "NM", "871"),
    ("Salt Lake City", "UT", "841"),
    ("Milwaukee", "WI", "532"),
    ("Madison", "WI", "537"),
    ("Birmingham", "AL", "352"),
    ("Anchorage", "AK", "995"),
    ("Honolulu", "HI", "968"),
    ("Newark", "NJ", "071"),
    ("Jersey City", "NJ", "073"),
    ("Omaha", "NE", "681"),
    ("Sioux Falls", "SD", "571"),
    ("Fargo", "ND", "581"),
    ("Burlington", "VT", "054"),
    ("Providence", "RI", "029"),
    ("Hartford", "CT", "061"),
    ("Wilmington", "DE", "198"),
    ("Charleston", "SC", "294"),
    ("Columbia", "SC", "292"),
    ("Little Rock", "AR", "722"),
    ("Jackson", "MS", "392"),
]

STREET_TYPES = ["St", "Ave", "Blvd", "Dr", "Rd", "Way", "Ln", "Ct", "Pl", "Pkwy"]
STREET_NAMES = [
    "Main",
    "Oak",
    "Maple",
    "Cedar",
    "Pine",
    "Elm",
    "Washington",
    "Park",
    "Lake",
    "Hill",
    "River",
    "Sunset",
    "Highland",
    "Broad",
    "Market",
    "Spring",
    "Union",
    "Church",
    "Center",
    "Division",
    "Commerce",
    "Industrial",
    "Harbor",
    "Bay",
    "Valley",
    "Green",
    "Garden",
    "Forest",
    "Meadow",
    "Willow",
    "Magnolia",
    "Peach",
    "Cherry",
    "Walnut",
    "Chestnut",
]


def _zip(prefix: str) -> str:
    return prefix + str(RNG.randint(0, 99)).zfill(2)


def _address() -> str:
    num = RNG.randint(100, 9999)
    name = RNG.choice(STREET_NAMES)
    stype = RNG.choice(STREET_TYPES)
    suffixes = ["", "", "", "", "Suite 100", "Suite 200", "Ste A", "Unit B", "#101"]
    suffix = RNG.choice(suffixes)
    return f"{num} {name} {stype}" + (f", {suffix}" if suffix else "")


# ---------------------------------------------------------------------------
# Name generators by segment
# ---------------------------------------------------------------------------

OWNER_FIRST = [
    "Joe",
    "Mike",
    "Tony",
    "Marco",
    "Luigi",
    "Carlos",
    "Maria",
    "Rosa",
    "Anna",
    "Pete",
    "Sam",
    "Dave",
    "Tom",
    "Bob",
    "Jim",
    "Bill",
    "Frank",
    "Sal",
    "Gino",
    "Vic",
    "Nick",
    "Leo",
    "Ray",
    "Phil",
    "Lou",
    "Al",
    "Eddie",
    "Richie",
    "Danny",
    "Chris",
    "Pat",
    "Jack",
    "Jimmy",
]

OWNER_LAST = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Miller",
    "Davis",
    "Garcia",
    "Martinez",
    "Rossi",
    "Rizzo",
    "Romano",
    "DeLuca",
    "Caruso",
    "Esposito",
    "Ferreira",
    "Chen",
    "Kim",
    "Park",
    "Nguyen",
    "Patel",
]

LOCAL_RESTAURANT_FORMATS = [
    "{first}'s Bar & Grill",
    "{first}'s Grill",
    "{last}'s Diner",
    "Mama {last}'s",
    "Mama {last}'s Pizzeria",
    "Mama {last}'s Kitchen",
    "Papa {last}'s",
    "Casa {last}",
    "{last} Family Restaurant",
    "{last} Kitchen",
    "The {last} House",
    "{first} & {first2}'s Steakhouse",
    "{first}'s Seafood",
    "{last}'s BBQ",
    "{city} Grill",
    "{city} Diner",
    "{city} Bistro",
    "{city} Kitchen",
    "Old {city} Tavern",
    "The {city} Grille",
]

CHAIN_RESTAURANTS = [
    "Applebee's",
    "Chili's",
    "Olive Garden",
    "TGI Fridays",
    "Red Lobster",
    "Outback Steakhouse",
    "Cracker Barrel",
    "IHOP",
    "Denny's",
    "Waffle House",
    "Buffalo Wild Wings",
    "Hooters",
    "Red Robin",
    "The Cheesecake Factory",
    "P.F. Chang's",
    "Panera Bread",
    "Chipotle Mexican Grill",
    "Five Guys",
    "Shake Shack",
    "Wingstop",
    "Raising Cane's",
    "Texas Roadhouse",
    "LongHorn Steakhouse",
    "Golden Corral",
    "Perkins Restaurant",
    "Bob Evans",
    "Steak 'n Shake",
    "Sonic Drive-In",
    "Whataburger",
    "In-N-Out Burger",
]

LOCAL_BAR_ADJECTIVES = [
    "Rusty",
    "Golden",
    "Silver",
    "Blue",
    "Red",
    "Black",
    "Lucky",
    "Wild",
    "Old",
    "New",
    "Royal",
    "Iron",
    "Copper",
    "Amber",
    "Midnight",
]

LOCAL_BAR_NOUNS = [
    "Anchor",
    "Barrel",
    "Tap",
    "Keg",
    "Rail",
    "Draft",
    "Pint",
    "Mug",
    "Stein",
    "Stool",
    "Saddle",
    "Spur",
    "Crow",
    "Eagle",
    "Fox",
    "Hound",
    "Wolf",
    "Bull",
    "Stag",
    "Bear",
]

LOCAL_BAR_FORMATS = [
    "The {adj} {noun}",
    "The {noun} & {noun2}",
    "{city} Tap Room",
    "{city} Brewing Co.",
    "{city} Taproom",
    "{city} Ale House",
    "{last}'s Pub",
    "{last}'s Tavern",
    "Sunset Lounge",
    "The Tap Room",
    "Down the Hatch",
    "The Rusty Nail",
    "The Broken Spoke",
    "Cheers Bar & Grill",
    "Sports Page Bar",
    "The Draft House",
    "Neighborhood Bar & Grill",
    "Corner Pocket Billiards & Bar",
]

CHAIN_BARS = [
    "Dave & Buster's",
    "Yard House",
    "Bar Louie",
    "Kona Grill",
    "World of Beer",
    "Punch Bowl Social",
    "Twin Peaks",
]

HOTEL_BRANDS = [
    ("Hilton", ["Garden Inn", "Homewood Suites", "Hampton Inn", "DoubleTree"]),
    ("Marriott", ["Courtyard", "Residence Inn", "Fairfield Inn", "Springhill Suites"]),
    ("Hyatt", ["Place", "House", "Regency", "Grand"]),
    ("IHG", ["Holiday Inn", "Holiday Inn Express", "Crowne Plaza", "Staybridge Suites"]),
    ("Wyndham", ["Days Inn", "Super 8", "La Quinta", "Ramada"]),
    ("Best Western", ["Plus", "Premier", ""]),
    ("Choice Hotels", ["Comfort Inn", "Quality Inn", "Clarion", "Econo Lodge"]),
    ("Radisson", ["Blu", "Red", ""]),
]

LIQUOR_STORES_LOCAL = [
    "{city} Liquors",
    "{city} Wine & Spirits",
    "{city} Bottle Shop",
    "{last}'s Liquor Store",
    "Total Spirits {city}",
    "Fine Wine & Spirits",
    "The Bottle Shop",
    "Premier Wine & Spirits",
    "Main Street Liquors",
    "Discount Liquor Mart",
]

CHAIN_LIQUOR = [
    "Total Wine & More",
    "Spec's Liquor",
    "BevMo",
    "ABC Fine Wine & Spirits",
    "Binny's Beverage Depot",
    "Liquor Barn",
    "Keg Liquors",
    "Central Liquors",
    "Marketview Liquor",
]

CHAIN_GROCERY = [
    "Kroger",
    "HEB",
    "Safeway",
    "Albertsons",
    "Publix",
    "Meijer",
    "Wegmans",
    "Trader Joe's",
    "Whole Foods Market",
    "Aldi",
    "Lidl",
    "WinCo Foods",
    "Food Lion",
    "Giant Eagle",
    "Stop & Shop",
    "Winn-Dixie",
    "Smart & Final",
    "Stater Bros.",
    "Hy-Vee",
    "Raley's",
    "Piggly Wiggly",
    "Ingles Markets",
    "Price Chopper",
]

CHAIN_CONVENIENCE = [
    "7-Eleven",
    "Circle K",
    "Speedway",
    "Wawa",
    "Sheetz",
    "Casey's General Store",
    "ampm",
    "Kwik Trip",
    "GetGo",
    "Pilot Flying J",
    "Love's Travel Stop",
]

CHAIN_WAREHOUSE = [
    "Costco Wholesale",
    "Sam's Club",
    "BJ's Wholesale Club",
]

CHAIN_DRUG = [
    "Walgreens",
    "CVS Pharmacy",
    "Rite Aid",
]

CHAIN_HOTEL_ONLY = [
    "Marriott Houston Downtown",
    "Hilton Garden Inn Dallas",
    "Courtyard by Marriott Chicago",
    "Holiday Inn Express Boston",
    "Hampton Inn Nashville",
    "Doubletree by Hilton Seattle",
    "Hyatt Place Denver",
    "Sheraton New York Times Square",
    "Westin Atlanta Peachtree",
    "Omni Dallas Hotel",
    "Loews Hotel Philadelphia",
    "JW Marriott Los Angeles",
    "Ritz-Carlton New Orleans",
    "Four Seasons Miami",
]


# ---------------------------------------------------------------------------
# Segment weights (how often each category appears)
# ---------------------------------------------------------------------------

SEGMENTS = [
    ("local_restaurant", 0.22),
    ("chain_restaurant", 0.18),
    ("local_bar", 0.12),
    ("chain_bar", 0.04),
    ("hotel", 0.10),
    ("local_liquor", 0.06),
    ("chain_liquor", 0.06),
    ("chain_grocery", 0.09),
    ("chain_convenience", 0.05),
    ("chain_warehouse", 0.02),
    ("chain_drug", 0.03),
    ("chain_hotel_only", 0.03),
]

_SEG_NAMES = [s for s, _ in SEGMENTS]
_SEG_WEIGHTS = [w for _, w in SEGMENTS]


def _pick_city() -> tuple[str, str, str]:
    return RNG.choice(CITIES)


def _name_local_restaurant(city: str) -> str:
    fmt = RNG.choice(LOCAL_RESTAURANT_FORMATS)
    first = RNG.choice(OWNER_FIRST)
    first2 = RNG.choice(OWNER_FIRST)
    last = RNG.choice(OWNER_LAST)
    return fmt.format(first=first, first2=first2, last=last, city=city)


def _name_local_bar(city: str) -> str:
    fmt = RNG.choice(LOCAL_BAR_FORMATS)
    adj = RNG.choice(LOCAL_BAR_ADJECTIVES)
    noun = RNG.choice(LOCAL_BAR_NOUNS)
    noun2 = RNG.choice(LOCAL_BAR_NOUNS)
    last = RNG.choice(OWNER_LAST)
    return fmt.format(adj=adj, noun=noun, noun2=noun2, city=city, last=last)


def _name_hotel(city: str) -> str:
    brand, sub_brands = RNG.choice(HOTEL_BRANDS)
    sub = RNG.choice(sub_brands)
    if sub:
        return (
            f"{sub} by {brand} {city}"
            if brand not in ("Best Western", "Choice Hotels")
            else f"{brand} {sub} {city}"
        )
    return f"{brand} {city}"


def _name_local_liquor(city: str) -> str:
    fmt = RNG.choice(LIQUOR_STORES_LOCAL)
    last = RNG.choice(OWNER_LAST)
    return fmt.format(city=city, last=last)


def _name_chain_convenience() -> str:
    chain = RNG.choice(CHAIN_CONVENIENCE)
    if chain == "7-Eleven":
        return f"7-Eleven Store #{RNG.randint(1000, 9999)}"
    store_num = RNG.randint(100, 9999)
    return f"{chain} #{store_num}"


def _generate_name(segment: str, city: str) -> str:
    if segment == "local_restaurant":
        return _name_local_restaurant(city)
    if segment == "chain_restaurant":
        return RNG.choice(CHAIN_RESTAURANTS)
    if segment == "local_bar":
        return _name_local_bar(city)
    if segment == "chain_bar":
        return RNG.choice(CHAIN_BARS)
    if segment == "hotel":
        return _name_hotel(city)
    if segment == "local_liquor":
        return _name_local_liquor(city)
    if segment == "chain_liquor":
        return RNG.choice(CHAIN_LIQUOR)
    if segment == "chain_grocery":
        return RNG.choice(CHAIN_GROCERY)
    if segment == "chain_convenience":
        return _name_chain_convenience()
    if segment == "chain_warehouse":
        return RNG.choice(CHAIN_WAREHOUSE)
    if segment == "chain_drug":
        return RNG.choice(CHAIN_DRUG)
    if segment == "chain_hotel_only":
        return RNG.choice(CHAIN_HOTEL_ONLY)
    return "Unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_input_csv(n_rows: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segments = RNG.choices(_SEG_NAMES, weights=_SEG_WEIGHTS, k=n_rows)

    rows: list[dict] = []
    for i, segment in enumerate(segments, start=1):
        city_name, state, zip_prefix = _pick_city()
        name = _generate_name(segment, city_name)
        rows.append(
            {
                "customer_id": f"CUST{i:06d}",
                "customer_name": name,
                "address": _address(),
                "city": city_name,
                "state": state,
                "zip": _zip(zip_prefix),
            }
        )

    fieldnames = ["customer_id", "customer_name", "address", "city", "state", "zip"]
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {n_rows} rows to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic customer CSVs for testing.")
    parser.add_argument(
        "--rows", type=int, default=1000, help="Number of rows to generate (default: 1000)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/samples/sample.csv"),
        help="Output CSV path (default: data/samples/sample.csv)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    RNG.seed(args.seed)
    generate_input_csv(args.rows, args.output)
