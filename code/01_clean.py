"""
01_clean.py - Data Cleaning and Preparation
============================================

Purpose:
--------
This script cleans and merges raw data sources to create an analysis-ready
panel dataset of state-year observations for the staggered DiD analysis
of primary enforcement handheld device bans on traffic fatalities.

Data Sources:
-------------

1. FARS (Fatality Analysis Reporting System) - Traffic Fatalities
   - Source: NHTSA (National Highway Traffic Safety Administration)
   - URL: https://www-fars.nhtsa.dot.gov/States/StatesFatalitiesFatalityRates.aspx
   - Alternative: FARS Encyclopedia at https://www-fars.nhtsa.dot.gov/
   - Contains: Annual state-level traffic fatality counts, 1994-2023
   - Format: Can be downloaded as CSV/Excel from FARS Encyclopedia

   API Alternative (may have access restrictions):
   - Endpoint: https://crashviewer.nhtsa.dot.gov/CrashAPI/
   - Example: /FARSData/GetFARSData?dataset=Accident&FromYear=2010&ToYear=2022&State=0&format=csv

   NBER Mirror (easier access):
   - URL: https://www.nber.org/research/data/fatality-analysis-reporting-system-fars
   - Contains processed FARS files in SAS/Stata formats

2. VMT (Vehicle Miles Traveled) by State
   - Source: FHWA Highway Statistics, Table VM-2
   - URL: https://www.fhwa.dot.gov/policyinformation/statistics/2022/vm2.cfm
   - Data.gov direct download:
     https://data.transportation.gov/api/views/nps9-3pm2/rows.csv?accessType=DOWNLOAD
   - Contains: Annual VMT by state and functional road class, 1980-2024

3. State Law Effective Dates - Primary Enforcement Handheld Bans
   - Compiled from: GHSA (ghsa.org), IIHS (iihs.org), state legislatures
   - File: data/raw/law_dates.csv (included in repository)
   - Key sources consulted:
     * GHSA: https://www.ghsa.org/state-laws/issues/Distracted%20Driving
     * IIHS: https://www.iihs.org/topics/distracted-driving/cellphone-use-laws
     * Zhu et al. (2021) Epidemiology paper supplementary materials

Inputs:
-------
- data/raw/fars_fatalities.csv: State-year fatality counts from FARS
  Columns: state, state_name, year, fatalities, fatality_rate

- data/raw/vmt_by_state.csv: VMT data from FHWA
  Columns: state, year, vmt_millions (vehicle miles traveled in millions)

- data/raw/law_dates.csv: Treatment timing data (included)
  Columns: state, state_abbrev, state_fips, primary_ban_date,
           primary_ban_year, ever_treated, notes

Outputs:
--------
- data/processed/panel.csv: Clean balanced panel with columns:
    - state: State name
    - state_abbrev: Two-letter abbreviation
    - state_fips: FIPS code (string, zero-padded)
    - year: Calendar year (2010-2022)
    - fatalities: Total traffic fatalities
    - vmt: Vehicle miles traveled (millions)
    - fatalities_per_100m_vmt: Fatalities per 100 million VMT
    - population: State population (if available)
    - treated: Binary indicator (1 if primary enforcement in effect)
    - first_treat_year: Year of adoption (0 if never treated)
    - rel_time: Years since treatment (event time)
    - cohort: Treatment cohort label for CS estimator

Notes:
------
- Treatment = PRIMARY enforcement handheld device ban in effect
- Secondary enforcement states (AL, MO) are never-treated controls
- States adopting after 2022 (MI, OH, CO, IA, PA, LA) are never-treated
- Panel covers 2010-2022 (13 years × 51 units = 663 observations)
- DC is included as a treated unit (ban effective 2004)

References:
-----------
- Zhu, M., et al. (2021). Bans on Cellphone Use While Driving and
  Traffic Fatalities in the United States. Epidemiology, 32(5), 731-739.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Study period
START_YEAR = 2010
END_YEAR = 2022

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# State FIPS codes for reference
STATE_FIPS = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
    'CO': '08', 'CT': '09', 'DE': '10', 'DC': '11', 'FL': '12',
    'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18',
    'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23',
    'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
    'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33',
    'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38',
    'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44',
    'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49',
    'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55',
    'WY': '56'
}

STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut',
    'DE': 'Delaware', 'DC': 'District of Columbia', 'FL': 'Florida',
    'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois',
    'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky',
    'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
    'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana',
    'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire',
    'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania',
    'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota',
    'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming'
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_law_dates():
    """
    Load treatment timing data from law_dates.csv.

    Returns DataFrame with columns:
    - state_abbrev: Two-letter state abbreviation
    - state_fips: FIPS code (zero-padded string)
    - primary_ban_year: Year of primary enforcement adoption (NaN if never)
    - ever_treated: 1 if treated within study period, 0 otherwise
    """
    law_path = RAW_DATA_DIR / "law_dates.csv"

    if not law_path.exists():
        raise FileNotFoundError(
            f"Law dates file not found at {law_path}. "
            "This file should be included in the repository."
        )

    laws = pd.read_csv(law_path)

    # For states that adopted outside our study period, mark as never-treated
    # Study period is 2010-2022, so states adopting in 2023+ are controls
    laws['ever_treated_in_study'] = (
        (laws['ever_treated'] == 1) &
        (laws['primary_ban_year'] <= END_YEAR) &
        (laws['primary_ban_year'].notna())
    ).astype(int)

    # For treatment year, use 0 for never-treated (CS convention)
    laws['first_treat_year'] = laws['primary_ban_year'].fillna(0).astype(int)
    laws.loc[laws['first_treat_year'] > END_YEAR, 'first_treat_year'] = 0
    laws.loc[laws['first_treat_year'] > END_YEAR, 'ever_treated_in_study'] = 0

    print(f"Loaded law dates for {len(laws)} states/territories")
    print(f"  Treated within study period: {laws['ever_treated_in_study'].sum()}")
    print(f"  Never treated (controls): {(laws['ever_treated_in_study'] == 0).sum()}")

    return laws


def load_fars_data():
    """
    Load FARS fatality data from CSV.

    Expected input format (data/raw/fars_fatalities.csv):
    - state: State name or abbreviation
    - year: Calendar year
    - fatalities: Total fatalities

    If file doesn't exist, provides instructions for obtaining data.
    """
    fars_path = RAW_DATA_DIR / "fars_fatalities.csv"

    if not fars_path.exists():
        print("\n" + "="*70)
        print("FARS DATA NOT FOUND")
        print("="*70)
        print("""
To obtain FARS fatality data:

Option 1: FARS Encyclopedia (Recommended for state-level totals)
  1. Go to: https://www-fars.nhtsa.dot.gov/States/StatesFatalitiesFatalityRates.aspx
  2. Select years 2010-2022
  3. Download the table as CSV
  4. Save as: data/raw/fars_fatalities.csv

Option 2: NHTSA CrashStats Publications
  1. Go to: https://crashstats.nhtsa.dot.gov/
  2. Search for "State Traffic Data" or annual overview reports
  3. Extract state-level fatality counts

Option 3: NBER FARS Mirror
  1. Go to: https://www.nber.org/research/data/fatality-analysis-reporting-system-fars
  2. Download accident-level files for each year
  3. Aggregate to state-year level (count records by STATE variable)

Expected CSV format:
  state,year,fatalities
  Alabama,2010,862
  Alabama,2011,894
  ...
""")
        print("="*70)
        return None

    fars = pd.read_csv(fars_path)
    print(f"Loaded FARS data: {len(fars)} rows")

    # Standardize column names
    fars.columns = fars.columns.str.lower().str.strip()

    # Standardize state names/abbreviations
    fars = standardize_state_column(fars)

    return fars


def load_vmt_data():
    """
    Load Vehicle Miles Traveled data from FHWA.

    Primary source: data/raw/vmt_by_state.csv

    Alternative: Download directly from data.gov:
    https://data.transportation.gov/api/views/nps9-3pm2/rows.csv?accessType=DOWNLOAD

    The raw FHWA data has VMT by functional road class; we need total VMT per state.
    """
    vmt_path = RAW_DATA_DIR / "vmt_by_state.csv"

    if not vmt_path.exists():
        print("\n" + "="*70)
        print("VMT DATA NOT FOUND")
        print("="*70)
        print("""
To obtain VMT data:

Option 1: Data.gov Direct Download
  1. Download from:
     https://data.transportation.gov/api/views/nps9-3pm2/rows.csv?accessType=DOWNLOAD
  2. This is the FHWA VM-2 table with VMT by functional system
  3. Aggregate by state and year to get total VMT
  4. Save as: data/raw/vmt_by_state.csv

Option 2: FHWA Highway Statistics Tables
  1. Go to: https://www.fhwa.dot.gov/policyinformation/statistics.cfm
  2. Select each year (2010-2022) and download VM-2 table
  3. Combine and aggregate to state-year totals

Expected CSV format:
  state,year,vmt_millions
  Alabama,2010,64523
  Alabama,2011,65012
  ...

Note: VMT should be in MILLIONS of vehicle miles traveled.
""")
        print("="*70)
        return None

    vmt = pd.read_csv(vmt_path)
    print(f"Loaded VMT data: {len(vmt)} rows")

    # Standardize column names
    vmt.columns = vmt.columns.str.lower().str.strip()

    # Standardize state names/abbreviations
    vmt = standardize_state_column(vmt)

    return vmt


def standardize_state_column(df):
    """
    Standardize state identifiers to two-letter abbreviations.
    Handles common formats: full names, abbreviations, FIPS codes.
    """
    # Look for state column
    state_cols = [c for c in df.columns if 'state' in c.lower()]
    if not state_cols:
        print("Warning: No state column found")
        return df

    state_col = state_cols[0]

    # Create reverse lookup from names to abbreviations
    name_to_abbrev = {v.lower(): k for k, v in STATE_NAMES.items()}
    fips_to_abbrev = {v: k for k, v in STATE_FIPS.items()}

    def convert_state(val):
        if pd.isna(val):
            return None
        val_str = str(val).strip()

        # Already an abbreviation
        if val_str.upper() in STATE_FIPS:
            return val_str.upper()

        # Full state name
        if val_str.lower() in name_to_abbrev:
            return name_to_abbrev[val_str.lower()]

        # FIPS code (as string)
        val_fips = val_str.zfill(2)
        if val_fips in fips_to_abbrev:
            return fips_to_abbrev[val_fips]

        # Numeric FIPS
        try:
            numeric_fips = str(int(float(val_str))).zfill(2)
            if numeric_fips in fips_to_abbrev:
                return fips_to_abbrev[numeric_fips]
        except (ValueError, TypeError):
            pass

        return None

    df['state_abbrev'] = df[state_col].apply(convert_state)

    # Report any failed conversions
    failed = df[df['state_abbrev'].isna()][state_col].unique()
    if len(failed) > 0:
        print(f"Warning: Could not convert states: {failed}")

    return df


# ============================================================================
# PANEL CONSTRUCTION
# ============================================================================

def create_state_year_skeleton():
    """
    Create a balanced panel skeleton with all state-year combinations.
    Returns DataFrame with state_abbrev × year for study period.
    """
    states = list(STATE_FIPS.keys())  # 51 states + DC
    years = list(range(START_YEAR, END_YEAR + 1))

    # Create all combinations
    skeleton = pd.DataFrame([
        {'state_abbrev': state, 'year': year}
        for state in states
        for year in years
    ])

    # Add state identifiers
    skeleton['state_fips'] = skeleton['state_abbrev'].map(STATE_FIPS)
    skeleton['state_name'] = skeleton['state_abbrev'].map(STATE_NAMES)

    print(f"Created panel skeleton: {len(skeleton)} observations "
          f"({len(states)} states × {len(years)} years)")

    return skeleton


def merge_treatment_indicators(panel, laws):
    """
    Merge treatment timing information and create treatment indicators.

    Creates:
    - first_treat_year: Year of first treatment (0 if never treated)
    - treated: 1 if treated by that year, 0 otherwise
    - rel_time: Years since treatment (event time)
    - cohort: String label for treatment cohort
    """
    # Merge law dates
    laws_subset = laws[['state_abbrev', 'first_treat_year', 'ever_treated_in_study']].copy()
    panel = panel.merge(laws_subset, on='state_abbrev', how='left')

    # Fill missing (should not happen if law_dates.csv is complete)
    panel['first_treat_year'] = panel['first_treat_year'].fillna(0).astype(int)
    panel['ever_treated_in_study'] = panel['ever_treated_in_study'].fillna(0).astype(int)

    # Create time-varying treatment indicator
    # treated = 1 if year >= first_treat_year AND first_treat_year > 0
    panel['treated'] = (
        (panel['first_treat_year'] > 0) &
        (panel['year'] >= panel['first_treat_year'])
    ).astype(int)

    # Create relative time (event time)
    # rel_time = year - first_treat_year for treated units
    # For never-treated, set to a large negative value or NA
    panel['rel_time'] = np.where(
        panel['first_treat_year'] > 0,
        panel['year'] - panel['first_treat_year'],
        np.nan
    )

    # Create cohort labels for CS estimator
    # Format: "g2017" for treated in 2017, "Never Treated" for controls
    panel['cohort'] = np.where(
        panel['first_treat_year'] > 0,
        'g' + panel['first_treat_year'].astype(str),
        'Never Treated'
    )

    # Summary statistics
    n_treated_units = (panel.groupby('state_abbrev')['ever_treated_in_study'].max() == 1).sum()
    n_control_units = (panel.groupby('state_abbrev')['ever_treated_in_study'].max() == 0).sum()

    print(f"\nTreatment summary:")
    print(f"  Treated units: {n_treated_units}")
    print(f"  Control units: {n_control_units}")
    print(f"  Treatment cohorts: {panel[panel['first_treat_year'] > 0]['first_treat_year'].nunique()}")

    # Show cohort distribution
    print("\n  Cohort distribution:")
    cohort_counts = panel[panel['first_treat_year'] > 0].groupby('first_treat_year')['state_abbrev'].nunique()
    for year, count in cohort_counts.items():
        print(f"    {year}: {count} states")

    return panel


def merge_fars_data(panel, fars):
    """
    Merge FARS fatality data into panel.
    """
    if fars is None:
        print("\nSkipping FARS merge (data not loaded)")
        panel['fatalities'] = np.nan
        return panel

    # Prepare FARS data for merge
    fars_clean = fars[['state_abbrev', 'year', 'fatalities']].copy()
    fars_clean = fars_clean.dropna(subset=['state_abbrev'])

    # Merge
    panel = panel.merge(fars_clean, on=['state_abbrev', 'year'], how='left')

    # Report coverage
    n_missing = panel['fatalities'].isna().sum()
    if n_missing > 0:
        print(f"\nWarning: {n_missing} state-year observations missing fatality data")
    else:
        print(f"\nFARS data merged: {panel['fatalities'].notna().sum()} observations")

    return panel


def merge_vmt_data(panel, vmt):
    """
    Merge VMT data into panel and calculate fatality rates.
    """
    if vmt is None:
        print("\nSkipping VMT merge (data not loaded)")
        panel['vmt'] = np.nan
        panel['fatalities_per_100m_vmt'] = np.nan
        return panel

    # Identify VMT column
    vmt_cols = [c for c in vmt.columns if 'vmt' in c.lower()]
    if not vmt_cols:
        print("Warning: No VMT column found in data")
        panel['vmt'] = np.nan
        panel['fatalities_per_100m_vmt'] = np.nan
        return panel

    vmt_col = vmt_cols[0]

    # Prepare VMT data for merge
    vmt_clean = vmt[['state_abbrev', 'year', vmt_col]].copy()
    vmt_clean = vmt_clean.rename(columns={vmt_col: 'vmt'})
    vmt_clean = vmt_clean.dropna(subset=['state_abbrev'])

    # Merge
    panel = panel.merge(vmt_clean, on=['state_abbrev', 'year'], how='left')

    # Calculate fatality rate per 100 million VMT
    # VMT is in millions, so divide by 100 to get per 100M
    panel['fatalities_per_100m_vmt'] = (
        panel['fatalities'] / (panel['vmt'] / 100)
    )

    # Report coverage
    n_missing = panel['vmt'].isna().sum()
    if n_missing > 0:
        print(f"\nWarning: {n_missing} state-year observations missing VMT data")
    else:
        print(f"\nVMT data merged: {panel['vmt'].notna().sum()} observations")

    return panel


# ============================================================================
# VALIDATION
# ============================================================================

def validate_panel(panel):
    """
    Validate the final panel dataset.
    """
    print("\n" + "="*70)
    print("PANEL VALIDATION")
    print("="*70)

    # Check dimensions
    n_states = panel['state_abbrev'].nunique()
    n_years = panel['year'].nunique()
    expected_rows = n_states * n_years

    print(f"\nDimensions:")
    print(f"  States: {n_states}")
    print(f"  Years: {panel['year'].min()} - {panel['year'].max()} ({n_years} years)")
    print(f"  Observations: {len(panel)} (expected: {expected_rows})")

    if len(panel) != expected_rows:
        print("  WARNING: Panel is unbalanced!")

    # Check for duplicates
    dups = panel.duplicated(subset=['state_abbrev', 'year'])
    if dups.any():
        print(f"\n  WARNING: {dups.sum()} duplicate state-year observations")

    # Check treatment timing
    print(f"\nTreatment status:")
    print(f"  Ever-treated units: {panel.groupby('state_abbrev')['ever_treated_in_study'].max().sum()}")
    print(f"  Never-treated units: {(panel.groupby('state_abbrev')['ever_treated_in_study'].max() == 0).sum()}")
    print(f"  Treated observations: {panel['treated'].sum()}")
    print(f"  Control observations: {(panel['treated'] == 0).sum()}")

    # Check outcome data
    print(f"\nOutcome data coverage:")
    print(f"  Fatalities: {panel['fatalities'].notna().sum()} / {len(panel)} "
          f"({100*panel['fatalities'].notna().mean():.1f}%)")
    print(f"  VMT: {panel['vmt'].notna().sum()} / {len(panel)} "
          f"({100*panel['vmt'].notna().mean():.1f}%)")
    print(f"  Fatality rate: {panel['fatalities_per_100m_vmt'].notna().sum()} / {len(panel)} "
          f"({100*panel['fatalities_per_100m_vmt'].notna().mean():.1f}%)")

    if panel['fatalities'].notna().any():
        print(f"\nOutcome summary (fatalities per 100M VMT):")
        rate = panel['fatalities_per_100m_vmt'].dropna()
        print(f"  Mean: {rate.mean():.2f}")
        print(f"  Std:  {rate.std():.2f}")
        print(f"  Min:  {rate.min():.2f}")
        print(f"  Max:  {rate.max():.2f}")

    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main data cleaning pipeline.
    """
    print("="*70)
    print("01_clean.py - Data Cleaning and Preparation")
    print("="*70)
    print(f"\nStudy period: {START_YEAR}-{END_YEAR}")

    # Ensure output directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load treatment timing data
    print("\n[1/6] Loading treatment timing data...")
    laws = load_law_dates()

    # Step 2: Create panel skeleton
    print("\n[2/6] Creating panel skeleton...")
    panel = create_state_year_skeleton()

    # Step 3: Merge treatment indicators
    print("\n[3/6] Creating treatment indicators...")
    panel = merge_treatment_indicators(panel, laws)

    # Step 4: Load and merge FARS data
    print("\n[4/6] Loading FARS fatality data...")
    fars = load_fars_data()
    panel = merge_fars_data(panel, fars)

    # Step 5: Load and merge VMT data
    print("\n[5/6] Loading VMT data...")
    vmt = load_vmt_data()
    panel = merge_vmt_data(panel, vmt)

    # Step 6: Validate and export
    print("\n[6/6] Validating panel...")
    validate_panel(panel)

    # Reorder columns for clarity
    col_order = [
        'state_name', 'state_abbrev', 'state_fips', 'year',
        'fatalities', 'vmt', 'fatalities_per_100m_vmt',
        'first_treat_year', 'treated', 'rel_time', 'cohort',
        'ever_treated_in_study'
    ]
    # Only include columns that exist
    col_order = [c for c in col_order if c in panel.columns]
    panel = panel[col_order]

    # Export
    output_path = PROCESSED_DATA_DIR / "panel.csv"
    panel.to_csv(output_path, index=False)

    print("\n" + "="*70)
    print(f"SUCCESS: Panel saved to {output_path}")
    print("="*70)

    # Print summary of what's needed
    if fars is None or vmt is None:
        print("\nNEXT STEPS:")
        print("-----------")
        if fars is None:
            print("1. Obtain FARS fatality data (see instructions above)")
            print("   Save to: data/raw/fars_fatalities.csv")
        if vmt is None:
            print("2. Obtain VMT data (see instructions above)")
            print("   Save to: data/raw/vmt_by_state.csv")
        print("\nThen re-run this script to complete the panel.")

    return panel


if __name__ == "__main__":
    panel = main()
