"""
00_download_data.py - Download and Process Raw Data
====================================================

This script downloads and processes the raw data needed for the analysis:
1. VMT data from FHWA via data.gov
2. FARS crash-level data from BTS geodata portal

Usage:
------
    python code/00_download_data.py

Requirements:
-------------
    pip install pandas requests tqdm

Outputs:
--------
    data/raw/vmt_by_state.csv - State-year VMT totals
    data/raw/fars_fatalities.csv - State-year fatality counts
"""

import os
import sys
from pathlib import Path
import requests
import io

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Years for analysis
START_YEAR = 2010
END_YEAR = 2022

# State FIPS to abbreviation mapping
FIPS_TO_ABBREV = {
    1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT',
    10: 'DE', 11: 'DC', 12: 'FL', 13: 'GA', 15: 'HI', 16: 'ID', 17: 'IL',
    18: 'IN', 19: 'IA', 20: 'KS', 21: 'KY', 22: 'LA', 23: 'ME', 24: 'MD',
    25: 'MA', 26: 'MI', 27: 'MN', 28: 'MS', 29: 'MO', 30: 'MT', 31: 'NE',
    32: 'NV', 33: 'NH', 34: 'NJ', 35: 'NM', 36: 'NY', 37: 'NC', 38: 'ND',
    39: 'OH', 40: 'OK', 41: 'OR', 42: 'PA', 44: 'RI', 45: 'SC', 46: 'SD',
    47: 'TN', 48: 'TX', 49: 'UT', 50: 'VT', 51: 'VA', 53: 'WA', 54: 'WV',
    55: 'WI', 56: 'WY'
}

STATE_NAMES = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
    'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
    'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE',
    'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI',
    'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
    'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}


def download_vmt_data():
    """
    Download and process VMT data from FHWA via data.gov.

    Source: https://catalog.data.gov/dataset/vehicle-miles-of-travel-by-functional-system-and-state-1980-2023-vm-2
    Direct URL: https://data.transportation.gov/api/views/nps9-3pm2/rows.csv?accessType=DOWNLOAD
    """
    print("\n" + "="*60)
    print("Downloading VMT Data from FHWA/data.gov")
    print("="*60)

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas required. Run: pip install pandas")
        return False

    vmt_url = "https://data.transportation.gov/api/views/nps9-3pm2/rows.csv?accessType=DOWNLOAD"
    raw_vmt_path = RAW_DATA_DIR / "vmt_raw.csv"
    output_path = RAW_DATA_DIR / "vmt_by_state.csv"

    # Check if raw file already exists
    if not raw_vmt_path.exists():
        print(f"Downloading from {vmt_url}...")
        try:
            response = requests.get(vmt_url, timeout=120)
            response.raise_for_status()
            with open(raw_vmt_path, 'wb') as f:
                f.write(response.content)
            print(f"  Saved raw data to {raw_vmt_path}")
        except requests.RequestException as e:
            print(f"  ERROR downloading: {e}")
            return False
    else:
        print(f"  Using existing raw file: {raw_vmt_path}")

    # Process the VMT data
    print("\nProcessing VMT data...")
    vmt = pd.read_csv(raw_vmt_path)
    print(f"  Raw data shape: {vmt.shape}")
    print(f"  Columns: {list(vmt.columns)}")

    # Filter to study years
    vmt = vmt[(vmt['Year'] >= START_YEAR) & (vmt['Year'] <= END_YEAR)]
    print(f"  After year filter ({START_YEAR}-{END_YEAR}): {len(vmt)} rows")

    # Convert state names to abbreviations
    vmt['state_abbrev'] = vmt['State'].map(STATE_NAMES)

    # Check for unmapped states
    unmapped = vmt[vmt['state_abbrev'].isna()]['State'].unique()
    if len(unmapped) > 0:
        print(f"  Warning: Unmapped states: {unmapped}")
        # Filter to valid states only
        vmt = vmt[vmt['state_abbrev'].notna()]

    # Aggregate VMT by state and year (sum across functional classes and areas)
    vmt_agg = vmt.groupby(['state_abbrev', 'Year'])['VMT'].sum().reset_index()
    vmt_agg.columns = ['state', 'year', 'vmt']

    # VMT is in actual miles, convert to millions
    vmt_agg['vmt_millions'] = vmt_agg['vmt'] / 1_000_000

    # Keep just the columns we need
    vmt_final = vmt_agg[['state', 'year', 'vmt_millions']].copy()
    vmt_final = vmt_final.sort_values(['state', 'year'])

    # Save
    vmt_final.to_csv(output_path, index=False)
    print(f"\nVMT data saved to {output_path}")
    print(f"  Shape: {vmt_final.shape}")
    print(f"  States: {vmt_final['state'].nunique()}")
    print(f"  Years: {sorted(vmt_final['year'].unique())}")

    # Show sample
    print("\n  Sample data:")
    print(vmt_final.head(10).to_string(index=False))

    return True


def download_fars_data():
    """
    Download FARS crash-level data from BTS and aggregate to state-year fatalities.

    This downloads each year's accident file and counts fatalities by state.

    Source: https://geodata.bts.gov/datasets/usdot::fatality-analysis-reporting-system-fars-{YEAR}-accidents
    """
    print("\n" + "="*60)
    print("Downloading FARS Fatality Data from BTS")
    print("="*60)

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas required. Run: pip install pandas")
        return False

    output_path = RAW_DATA_DIR / "fars_fatalities.csv"
    all_fatalities = []

    # BTS GeoJSON API endpoints for each year
    # Format varies slightly by year
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n  Processing {year}...")

        # Try different URL patterns
        urls_to_try = [
            f"https://geodata.bts.gov/datasets/usdot::fatality-analysis-reporting-system-fars-{year}-accidents.csv",
            f"https://opendata.arcgis.com/api/v3/datasets/usdot::fatality-analysis-reporting-system-fars-{year}-accidents/downloads/data?format=csv&spatialRefId=4326",
        ]

        df = None
        for url in urls_to_try:
            try:
                print(f"    Trying: {url[:80]}...")
                response = requests.get(url, timeout=60, allow_redirects=True)
                if response.status_code == 200 and len(response.content) > 1000:
                    df = pd.read_csv(io.StringIO(response.text))
                    print(f"    Downloaded {len(df)} crash records")
                    break
            except Exception as e:
                continue

        if df is None:
            print(f"    WARNING: Could not download {year} data")
            continue

        # Look for state and fatality columns
        # Column names vary by year; common patterns: STATE, STATENAME, FATALS
        state_col = None
        fatals_col = None

        for col in df.columns:
            if col.upper() == 'STATE':
                state_col = col
            elif col.upper() == 'FATALS':
                fatals_col = col

        if state_col is None or fatals_col is None:
            print(f"    ERROR: Could not find STATE ({state_col}) or FATALS ({fatals_col}) columns")
            print(f"    Available columns: {list(df.columns)[:10]}...")
            continue

        # Aggregate by state
        # Convert FIPS to abbreviation and sum fatalities
        df['state_abbrev'] = df[state_col].map(FIPS_TO_ABBREV)

        state_fatals = df.groupby('state_abbrev')[fatals_col].sum().reset_index()
        state_fatals.columns = ['state', 'fatalities']
        state_fatals['year'] = year

        all_fatalities.append(state_fatals)
        print(f"    Aggregated to {len(state_fatals)} states, {state_fatals['fatalities'].sum()} total fatalities")

    if not all_fatalities:
        print("\nERROR: No FARS data could be downloaded.")
        print("Alternative: Manually download from FARS Encyclopedia")
        print("  https://www-fars.nhtsa.dot.gov/States/StatesFatalitiesFatalityRates.aspx")
        return False

    # Combine all years
    fars_df = pd.concat(all_fatalities, ignore_index=True)
    fars_df = fars_df[['state', 'year', 'fatalities']].sort_values(['state', 'year'])

    # Save
    fars_df.to_csv(output_path, index=False)
    print(f"\nFARS data saved to {output_path}")
    print(f"  Shape: {fars_df.shape}")
    print(f"  States: {fars_df['state'].nunique()}")
    print(f"  Years: {sorted(fars_df['year'].unique())}")
    print(f"  Total fatalities: {fars_df['fatalities'].sum():,}")

    # Show sample
    print("\n  Sample data:")
    print(fars_df.head(10).to_string(index=False))

    return True


def main():
    print("="*60)
    print("Data Download Script for Texting Ban DiD Analysis")
    print("="*60)
    print(f"\nStudy period: {START_YEAR}-{END_YEAR}")
    print(f"Output directory: {RAW_DATA_DIR}")

    # Download VMT data
    vmt_success = download_vmt_data()

    # Download FARS data
    fars_success = download_fars_data()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  VMT data:  {'SUCCESS' if vmt_success else 'FAILED'}")
    print(f"  FARS data: {'SUCCESS' if fars_success else 'FAILED'}")

    if vmt_success and fars_success:
        print("\nAll data downloaded successfully!")
        print("Run: python code/01_clean.py")
    else:
        print("\nSome data failed to download. See errors above.")

    return vmt_success and fars_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
