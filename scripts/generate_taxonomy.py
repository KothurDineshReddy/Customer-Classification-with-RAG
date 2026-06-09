"""Generate data/taxonomy/v1.yaml with ~4000 classification entries."""

import itertools
import random
from pathlib import Path

import yaml

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "taxonomy" / "v1.yaml"

# ---------------------------------------------------------------------------
# Taxonomy dimensions
# ---------------------------------------------------------------------------

NATIONAL_REGIONAL = ["National", "Regional"]
UNIT_TYPES = ["Multi-unit", "Single-unit"]
PREMISES = ["On-Premise", "Off-Premise"]

ON_PREMISE_CHANNELS = [
    "Restaurant",
    "Bar",
    "Hotel",
    "Nightclub",
    "Cafe",
    "Catering",
    "Stadium",
]

OFF_PREMISE_CHANNELS = [
    "Liquor Store",
    "Grocery",
    "Convenience",
    "Warehouse Club",
    "Drug Store",
    "Online Retail",
]

# ---------------------------------------------------------------------------
# Establishment types per channel
# Each tuple: (name, definition, inclusion_examples, exclusion_examples)
# ---------------------------------------------------------------------------

ON_PREMISE_ESTABLISHMENTS: dict[str, list[tuple]] = {
    "Restaurant": [
        (
            "Fine Dining",
            "Upscale restaurants with formal table service, premium menus, and elevated ambiance.",
            ["white-tablecloth restaurants", "prix-fixe tasting menus", "Michelin-starred venues"],
            ["casual dining", "fast casual", "cafeterias"],
        ),
        (
            "Casual Dining",
            "Full-service restaurants with table service and a relaxed, informal atmosphere.",
            ["family restaurants", "diners", "neighborhood grills", "chain sit-down restaurants"],
            ["fast food", "fine dining", "bars primarily serving alcohol"],
        ),
        (
            "Fast Casual",
            "Counter-service restaurants with higher-quality ingredients than fast food.",
            ["build-your-own bowl concepts", "elevated burger chains", "artisan sandwich shops"],
            ["full table service", "drive-through-only chains", "food courts"],
        ),
        (
            "Quick Service / Fast Food",
            "Limited-service restaurants focused on speed, standardized menus, and low price points.",  # noqa: E501
            ["burger chains", "fried chicken chains", "pizza delivery", "taco chains"],
            ["casual dining", "food trucks with table seating", "cafeterias"],
        ),
        (
            "Ethnic / Specialty Cuisine",
            "Restaurants centered on a specific regional or cultural cuisine.",
            ["Italian trattorias", "Japanese izakayas", "Indian curry houses", "Mexican taquerias"],
            ["fusion concepts without dominant single cuisine", "generic American diners"],
        ),
        (
            "Steakhouse",
            "Restaurants specializing in beef cuts with full bar programs.",
            ["Texas-style steakhouses", "chophouses", "dry-aged beef specialists"],
            ["burger-only casual concepts", "BBQ joints without steak focus"],
        ),
        (
            "Seafood Restaurant",
            "Restaurants with menus primarily composed of fish and shellfish.",
            ["fish houses", "raw bars", "coastal seafood shacks", "crab houses"],
            ["sushi bars", "fish-and-chip takeaways without seating"],
        ),
        (
            "Pizza Restaurant",
            "Establishments with pizza as the primary menu item, often with beer/wine service.",
            ["Neapolitan pizza parlors", "New York-style pizza by the slice", "deep-dish chains"],
            ["fast-food pizza delivery without dine-in", "Italian full-menu restaurants"],
        ),
        (
            "BBQ / Smokehouse",
            "Restaurants specializing in slow-smoked meats with regional sauce styles.",
            ["Texas BBQ joints", "Kansas City smokehouses", "Carolina pulled-pork shacks"],
            ["steakhouses", "casual dining chains with a BBQ section"],
        ),
        (
            "Sushi / Japanese",
            "Japanese restaurants focused on sushi, sashimi, and sake programs.",
            ["omakase counters", "conveyor-belt sushi", "traditional izakayas"],
            ["pan-Asian fusion without sushi focus", "ramen-only shops"],
        ),
        (
            "Ramen / Noodle Bar",
            "Casual restaurants specializing in ramen or other noodle-centric dishes.",
            ["tonkotsu ramen shops", "udon bars", "Vietnamese pho restaurants"],
            ["full Japanese restaurant menus", "Chinese-American takeout"],
        ),
        (
            "Tapas / Small Plates",
            "Restaurants structured around sharing small plates with cocktail or wine emphasis.",
            ["Spanish tapas bars", "mezze restaurants", "modern small-plate concepts"],
            ["full-service restaurants with an appetizer section", "bar food"],
        ),
        (
            "Brunch / Breakfast Specialist",
            "Daypart-focused establishments serving brunch or breakfast with Bloody Marys and mimosas.",  # noqa: E501
            ["weekend brunch spots", "all-day breakfast diners", "French crêperies"],
            ["hotels with complimentary breakfast buffets", "coffee-only cafes"],
        ),
        (
            "Ghost Kitchen / Virtual Brand",
            "Delivery-only restaurant concepts operating from shared or licensed kitchen space.",
            ["delivery-only burger brands", "virtual chicken wing concepts"],
            ["brick-and-mortar with delivery as a side channel", "catering commissaries"],
        ),
        (
            "Food Hall Vendor",
            "Restaurant stalls or counters operating inside a curated food hall.",
            ["market-hall vendors", "artisan food stalls", "food court upscale concepts"],
            ["standalone restaurants", "food trucks parked outside"],
        ),
    ],
    "Bar": [
        (
            "Sports Bar",
            "Bars with large-screen TVs focused on game-day viewing and domestic beer.",
            ["game-day bars", "arena-adjacent sports bars", "fantasy sports lounges"],
            ["nightclubs", "wine bars", "cocktail lounges"],
        ),
        (
            "Cocktail Lounge",
            "Bars emphasizing craft cocktails, premium spirits, and skilled bartending.",
            ["speakeasies", "craft cocktail bars", "bitters-focused bars"],
            ["dive bars", "sports bars", "nightclubs"],
        ),
        (
            "Dive Bar",
            "Unpretentious neighborhood bars with low prices and loyal regulars.",
            ["corner taverns", "neighborhood pubs", "cash-only bars"],
            ["cocktail lounges", "wine bars", "nightclubs"],
        ),
        (
            "Wine Bar",
            "Bars with extensive curated wine lists and light food pairings.",
            ["natural wine bars", "champagne bars", "wine-and-cheese concepts"],
            ["cocktail bars", "beer halls", "full-service restaurants"],
        ),
        (
            "Beer Bar / Taproom",
            "Bars centered on draft beer selection, often craft-focused or brewery-owned.",
            ["craft beer taprooms", "brewery taprooms", "Belgian beer bars"],
            ["sports bars with limited taps", "nightclubs", "wine bars"],
        ),
        (
            "Whiskey / Spirits Bar",
            "Bars specializing in curated selections of whiskey, bourbon, or other spirits.",
            ["bourbon bars", "scotch whisky bars", "mezcal bars", "rum bars"],
            ["generic cocktail bars", "nightclubs", "dive bars"],
        ),
        (
            "Piano Bar / Live Music Venue",
            "Bars with regular live musical performances as a primary draw.",
            ["piano bars", "jazz bars", "blues clubs", "acoustic music venues"],
            ["nightclubs with DJs", "karaoke bars", "concert halls"],
        ),
        (
            "Karaoke Bar",
            "Bars offering private or open karaoke rooms with beverage service.",
            ["private-room karaoke venues", "open-mic karaoke nights"],
            ["nightclubs", "live music venues", "piano bars"],
        ),
        (
            "Rooftop Bar",
            "Open-air bars on building rooftops, often attached to hotels or mixed-use buildings.",
            ["hotel rooftop bars", "skyline view bars", "seasonal outdoor bars"],
            ["hotel lobby bars", "nightclub rooftop parties"],
        ),
        (
            "Gastropub",
            "Pubs that balance elevated food menus with a curated craft beer or spirits program.",
            ["craft beer pubs with chef-driven menus", "modern British pubs"],
            ["casual dining with a bar section", "dive bars with pub food"],
        ),
    ],
    "Hotel": [
        (
            "Full-Service Hotel Bar",
            "Lobby or lounge bar within a full-service hotel catering to guests and locals.",
            ["hotel lobby bars", "resort bars", "business hotel lounges"],
            ["airport hotel vending areas", "motel common rooms"],
        ),
        (
            "Resort Pool Bar",
            "Outdoor bars adjacent to hotel pools, serving frozen drinks and tropical cocktails.",
            ["swim-up bars", "tiki pool bars", "beach resort bars"],
            ["indoor hotel bars", "standalone beach bars"],
        ),
        (
            "Hotel Restaurant",
            "Full-service restaurant located inside or attached to a hotel property.",
            ["hotel dining rooms", "celebrity-chef hotel restaurants", "resort buffet restaurants"],
            ["hotel room service", "complimentary continental breakfast"],
        ),
        (
            "Boutique Hotel Bar",
            "Intimate bar within an independent or lifestyle hotel with a curated drink program.",
            ["design hotel bars", "art-hotel lounges", "literary hotel bars"],
            ["chain hotel lobby bars", "resort bars"],
        ),
        (
            "Airport Hotel Bar",
            "Hotel bars primarily serving travelers at or near major airports.",
            ["transit hotel bars", "airline crew layover hotel bars"],
            ["airport terminal bars", "downtown full-service hotel bars"],
        ),
        (
            "Conference / Convention Hotel",
            "Hotels with significant meeting space and banquet beverage service for events.",
            ["convention center hotels", "large conference resort properties"],
            ["boutique hotels", "extended-stay hotels"],
        ),
    ],
    "Nightclub": [
        (
            "Dance Club",
            "Large venues with DJ entertainment, dance floors, and bottle-service programs.",
            ["EDM clubs", "hip-hop clubs", "Latin dance clubs"],
            ["live music venues", "piano bars", "sports bars"],
        ),
        (
            "Rooftop Nightclub",
            "Outdoor nightclub experiences on elevated terraces with premium pricing.",
            ["open-air DJ terraces", "rooftop pool parties", "sky lounges"],
            ["rooftop bars without late-night entertainment", "hotel rooftop bars"],
        ),
        (
            "Adult Entertainment Venue",
            "Adult-entertainment venues with licensed beverage service and on-premise consumption.",
            ["gentlemen's clubs", "adult cabarets"],
            ["nightclubs", "comedy clubs", "karaoke bars"],
        ),
        (
            "Comedy Club",
            "Venues combining live stand-up or improv comedy with two-drink minimums.",
            ["improv theaters with bar service", "stand-up comedy venues"],
            ["nightclubs", "live music bars", "karaoke bars"],
        ),
        (
            "Ultra-Lounge",
            "Upscale nightlife venues with low lighting, table service, and premium spirits.",
            ["VIP lounges", "bottle-service-only venues", "exclusive membership clubs"],
            ["dive bars", "dance clubs", "sports bars"],
        ),
    ],
    "Cafe": [
        (
            "Coffee & Wine Cafe",
            "Daytime cafe transitioning to a wine and beer program in the afternoon/evening.",
            ["wine-and-coffee bars", "espresso-by-day wine-by-night concepts"],
            ["coffee-only shops", "full-service restaurants"],
        ),
        (
            "Bakery Cafe",
            "Cafe anchored by baked goods with limited beer or wine for afternoon service.",
            ["French patisseries with wine", "artisan bread cafes with beer"],
            ["coffee shops without alcohol", "full-service restaurants"],
        ),
        (
            "Tea House",
            "Specialty tea cafes that may offer sake, plum wine, or kombucha on tap.",
            ["Japanese tea houses", "Taiwanese bubble-tea shops with alcohol"],
            ["coffee shops", "standard cafes"],
        ),
        (
            "Co-Working Cafe / Bar",
            "Cafe-bar hybrids with daytime work-friendly seating and evening bar service.",
            ["laptop-friendly bars", "work-café combos with beer on tap"],
            ["standard coffee shops", "coworking spaces without beverage licenses"],
        ),
    ],
    "Catering": [
        (
            "Full-Service Catering Company",
            "Off-site catering providing staffed bar and food service at events.",
            ["wedding caterers", "corporate event caterers", "gala caterers"],
            ["food trucks", "restaurant catering departments"],
        ),
        (
            "Corporate Catering",
            "Catering focused on business meals, office deliveries, and company events.",
            ["office lunch caterers", "boardroom meeting caterers", "company picnic operators"],
            ["wedding caterers", "personal chef services"],
        ),
        (
            "Wedding / Special Events Caterer",
            "Catering companies specializing in weddings, mitzvahs, and milestone events.",
            [
                "wedding reception caterers",
                "rehearsal-dinner caterers",
                "anniversary party caterers",
            ],
            ["corporate caterers", "stadium concession operators"],
        ),
        (
            "Concession Caterer",
            "Caterers managing food and beverage concessions at venues under a contracted agreement.",  # noqa: E501
            ["stadium concession operators", "arena concessionaires", "fairground food vendors"],
            ["standalone food trucks", "restaurant pop-ups"],
        ),
    ],
    "Stadium": [
        (
            "Professional Sports Arena",
            "Major league stadiums and arenas with premium bar and concession programs.",
            ["NFL stadiums", "NBA arenas", "MLB ballparks", "NHL arenas"],
            ["minor-league parks", "high school stadiums"],
        ),
        (
            "College / University Stadium",
            "Campus athletic facilities with licensed alcohol service at sporting events.",
            ["Power-5 football stadiums", "university basketball arenas"],
            ["high school stadiums", "club sports fields"],
        ),
        (
            "Concert Venue / Amphitheater",
            "Large outdoor or indoor music venues with extensive bar and concession service.",
            ["shed amphitheaters", "indoor arenas hosting concerts", "outdoor festival grounds"],
            ["small clubs", "bars with live music", "comedy clubs"],
        ),
        (
            "Minor League / Independent Sports Venue",
            "Smaller stadiums for minor-league teams with craft beer and local spirits focus.",
            ["minor-league baseball parks", "USL soccer stadiums", "indoor football arenas"],
            ["professional sports arenas", "high school gyms"],
        ),
        (
            "Racetrack / Motorsports Venue",
            "Racing facilities with trackside bars, clubs, and concession operations.",
            ["NASCAR tracks", "horse-racing tracks with infield bars", "go-kart complexes"],
            ["professional sports arenas", "concert venues"],
        ),
    ],
}

OFF_PREMISE_ESTABLISHMENTS: dict[str, list[tuple]] = {
    "Liquor Store": [
        (
            "Specialty / Independent Liquor Store",
            "Independent retailers with curated spirits, wine, and beer selections.",
            ["boutique bottle shops", "single-malt specialists", "natural wine shops"],
            ["chain liquor superstores", "grocery store wine aisles"],
        ),
        (
            "Chain Liquor Superstore",
            "Large-format chain liquor retailers with high SKU counts and competitive pricing.",
            ["Total Wine & More", "BevMo", "ABC Fine Wine & Spirits"],
            ["independent bottle shops", "state-controlled liquor stores"],
        ),
        (
            "State-Controlled Liquor Store",
            "Government-operated retail stores in control states with fixed pricing.",
            ["PLCB stores (Pennsylvania)", "OLCC stores (Oregon)", "NH state liquor stores"],
            ["private liquor retailers", "grocery store wine departments"],
        ),
        (
            "Duty-Free Liquor Retail",
            "Airport or border retailers selling alcohol free of local excise taxes.",
            ["airport duty-free shops", "border-crossing duty-free stores"],
            ["standard airport convenience stores", "hotel gift shops"],
        ),
    ],
    "Grocery": [
        (
            "Conventional Supermarket",
            "Full-service grocery chains with a dedicated beer, wine, and spirits aisle.",
            ["Kroger", "Safeway", "Albertsons", "Publix", "regional chains"],
            ["warehouse clubs", "natural food stores", "convenience stores"],
        ),
        (
            "Natural / Organic Grocery",
            "Health-focused grocers with curated organic wines, craft beers, and local spirits.",
            ["Whole Foods Market", "Natural Grocers", "co-op grocers"],
            ["conventional supermarkets", "discount grocery chains"],
        ),
        (
            "Ethnic / International Grocery",
            "Specialty grocers catering to specific cultural communities with imported wines and spirits.",  # noqa: E501
            ["Asian supermarkets with sake and soju", "Latin markets with agave spirits"],
            ["conventional supermarkets", "specialty bottle shops"],
        ),
        (
            "Discount Grocery",
            "Value-oriented grocery retailers with a limited but price-competitive alcohol selection.",  # noqa: E501
            ["ALDI", "Lidl", "WinCo Foods", "Grocery Outlet"],
            ["conventional supermarkets", "natural food stores"],
        ),
        (
            "Upscale / Gourmet Grocery",
            "Premium grocery retailers with extensive wine programs and curated craft beer sections.",  # noqa: E501
            ["Dean & DeLuca", "Bristol Farms", "The Fresh Market"],
            ["conventional supermarkets", "discount grocers"],
        ),
    ],
    "Convenience": [
        (
            "Gas Station Convenience Store",
            "Fuel-adjacent convenience retailers with cold beer coolers and wine.",
            ["Circle K", "Speedway", "Shell with convenience store"],
            ["standalone convenience stores", "drug stores"],
        ),
        (
            "Urban Convenience Store",
            "Non-fuel convenience retailers in dense urban areas with cold beer and wine.",
            ["7-Eleven", "ampm", "Wawa", "Casey's General Store", "Sheetz"],
            ["gas station c-stores", "grocery stores"],
        ),
        (
            "College-Area Convenience",
            "Convenience stores in high-density college neighborhoods with extended hours.",
            ["campus-edge c-stores", "late-night convenience retailers near universities"],
            ["standard urban c-stores", "gas station c-stores"],
        ),
    ],
    "Warehouse Club": [
        (
            "Membership Warehouse Club",
            "Bulk-format retailers selling large-format wine, spirits, and case beer at warehouse pricing.",  # noqa: E501
            ["Costco Wholesale", "Sam's Club", "BJ's Wholesale Club"],
            ["grocery chains", "liquor superstores", "discount grocery"],
        ),
    ],
    "Drug Store": [
        (
            "National Drug Store Chain",
            "Pharmacy-led retailers with wine and beer sections.",
            ["Walgreens", "CVS Pharmacy", "Rite Aid"],
            ["independent pharmacies", "grocery store pharmacy counters"],
        ),
        (
            "Independent Pharmacy / Drug Store",
            "Locally owned pharmacies that carry a small selection of wine and beer.",
            ["corner drug stores", "compounding pharmacies with a beverage section"],
            ["national chain drug stores", "grocery store pharmacies"],
        ),
    ],
    "Online Retail": [
        (
            "Direct-to-Consumer Wine Club",
            "Subscription-based wine retailers shipping directly from wineries to consumers.",
            ["Wine.com", "Winc", "Naked Wines", "winery DTC clubs"],
            ["marketplace platforms", "alcohol delivery apps"],
        ),
        (
            "Online Alcohol Marketplace",
            "E-commerce platforms aggregating multiple retailers' inventory for home delivery.",
            ["Drizly", "Minibar", "ReserveBar", "Total Wine online"],
            ["DTC wine clubs", "grocery delivery apps"],
        ),
        (
            "Grocery Delivery with Alcohol",
            "Grocery delivery services that include alcohol in their product catalog.",
            ["Instacart with alcohol", "Gopuff", "DoorDash Alcohol", "Amazon Fresh with alcohol"],
            ["standalone alcohol delivery apps", "DTC wine clubs"],
        ),
        (
            "Spirits Subscription / Discovery Service",
            "Subscription boxes or curated sample services delivering spirits to consumers.",
            ["Flaviar", "Taster's Club", "Single Cask Nation subscription"],
            ["DTC wine clubs", "online marketplaces"],
        ),
    ],
}

# ---------------------------------------------------------------------------
# Variation axes
# ---------------------------------------------------------------------------

CUISINE_VARIANTS = [
    "American",
    "Italian",
    "French",
    "Japanese",
    "Chinese",
    "Mexican",
    "Thai",
    "Indian",
    "Mediterranean",
    "Greek",
    "Korean",
    "Vietnamese",
    "Spanish",
    "Middle Eastern",
    "Latin Fusion",
    "Peruvian",
    "Brazilian",
    "Ethiopian",
    "Caribbean",
    "Southern / Soul Food",
    "New American",
]

HOTEL_CATEGORIES = [
    "Budget",
    "Select-Service",
    "Full-Service",
    "Upscale",
    "Upper-Upscale",
    "Luxury",
    "Boutique",
    "Extended-Stay",
    "Resort",
    "Airport",
    "Conference",
]

STORE_FORMATS = [
    "Urban",
    "Suburban",
    "Rural",
    "Strip-Mall",
    "Downtown",
    "Highway",
    "Tourist District",
    "College Town",
    "Resort Area",
]


def make_entries() -> list[dict]:
    entries: list[dict] = []
    idx = 1

    def add(
        nat_reg: str,
        unit: str,
        premise: str,
        channel: str,
        est_name: str,
        definition: str,
        inclusions: list[str],
        exclusions: list[str],
        suffix: str = "",
    ) -> None:
        nonlocal idx
        full_name = f"{est_name} ({suffix})" if suffix else est_name
        entries.append(
            {
                "id": f"TAX{idx:04d}",
                "national_regional": nat_reg,
                "unit_type": unit,
                "premise": premise,
                "channel": channel,
                "establishment_type": full_name,
                "definition": definition,
                "inclusion_examples": inclusions,
                "exclusion_examples": exclusions,
            }
        )
        idx += 1

    nr_ut = list(itertools.product(NATIONAL_REGIONAL, UNIT_TYPES))

    # Base On-Premise pass: all channels × all ests × NR × UT
    for channel, est_list in ON_PREMISE_ESTABLISHMENTS.items():
        for est_name, defn, incl, excl in est_list:
            for nat_reg, unit in nr_ut:
                add(nat_reg, unit, "On-Premise", channel, est_name, defn, incl, excl)

    # Cuisine variants for Restaurant and Bar
    for cuisine in CUISINE_VARIANTS:
        for est_name, defn, incl, excl in ON_PREMISE_ESTABLISHMENTS["Restaurant"]:
            for nat_reg, unit in nr_ut:
                add(nat_reg, unit, "On-Premise", "Restaurant", est_name, defn, incl, excl, cuisine)
        for est_name, defn, incl, excl in ON_PREMISE_ESTABLISHMENTS["Bar"]:
            for nat_reg, unit in nr_ut:
                add(nat_reg, unit, "On-Premise", "Bar", est_name, defn, incl, excl, cuisine)

    # Hotel category variants
    for category in HOTEL_CATEGORIES:
        for est_name, defn, incl, excl in ON_PREMISE_ESTABLISHMENTS["Hotel"]:
            for nat_reg, unit in nr_ut:
                add(nat_reg, unit, "On-Premise", "Hotel", est_name, defn, incl, excl, category)

    # Store-format variants for Nightclub, Cafe, Catering, Stadium
    for channel_key in ("Nightclub", "Cafe", "Catering", "Stadium"):
        for fmt in STORE_FORMATS:
            for est_name, defn, incl, excl in ON_PREMISE_ESTABLISHMENTS[channel_key]:
                for nat_reg, unit in nr_ut:
                    add(nat_reg, unit, "On-Premise", channel_key, est_name, defn, incl, excl, fmt)

    # Base Off-Premise pass
    for channel, est_list in OFF_PREMISE_ESTABLISHMENTS.items():
        for est_name, defn, incl, excl in est_list:
            for nat_reg, unit in nr_ut:
                add(nat_reg, unit, "Off-Premise", channel, est_name, defn, incl, excl)

    # Store-format variants for all off-premise channels  # noqa: E501
    for channel_key in (
        "Grocery",
        "Convenience",
        "Liquor Store",
        "Drug Store",
        "Warehouse Club",
        "Online Retail",
    ):
        for fmt in STORE_FORMATS:
            for est_name, defn, incl, excl in OFF_PREMISE_ESTABLISHMENTS[channel_key]:
                for nat_reg, unit in nr_ut:
                    add(nat_reg, unit, "Off-Premise", channel_key, est_name, defn, incl, excl, fmt)

    # Cuisine variants for Ethnic / International Grocery
    ethnic_est = next(e for e in OFF_PREMISE_ESTABLISHMENTS["Grocery"] if "Ethnic" in e[0])
    for cuisine in CUISINE_VARIANTS:
        for nat_reg, unit in nr_ut:
            add(
                nat_reg,
                unit,
                "Off-Premise",
                "Grocery",
                ethnic_est[0],
                ethnic_est[1],
                ethnic_est[2],
                ethnic_est[3],
                cuisine,
            )

    return entries


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    entries = make_entries()

    taxonomy = {
        "taxonomy_version": "v1",
        "generated_by": "scripts/generate_taxonomy.py",
        "total_entries": len(entries),
        "entries": entries,
    }

    with OUTPUT_PATH.open("w") as fh:
        yaml.dump(taxonomy, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"Wrote {len(entries)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
