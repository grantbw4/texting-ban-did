"""
02_twfe.py - Two-Way Fixed Effects Estimation
==============================================

Purpose:
--------
This script estimates the effect of primary enforcement handheld device bans
on traffic fatalities using traditional two-way fixed effects (TWFE) regression.
TWFE serves as a baseline estimator for comparison with modern methods.

Inputs:
-------
- data/processed/panel.csv: Clean panel dataset from 01_clean.py

Outputs:
--------
- output/tables/twfe_results.csv: CSV of coefficient estimates and standard errors
- output/figures/twfe_coef_plot.png: Coefficient comparison plot
- Console output with formatted regression tables

Model Specification:
--------------------
Y_it = α_i + λ_t + β * Treated_it + ε_it

Where:
- Y_it: Outcome (fatalities per 100M VMT) for state i in year t
- α_i: State fixed effects
- λ_t: Year fixed effects
- Treated_it: Binary indicator = 1 if state i has primary enforcement in year t
- β: Treatment effect of interest (ATT under parallel trends)

Caveats:
--------
- TWFE can be biased with staggered adoption and heterogeneous treatment effects
- Early-treated units may serve as controls for late-treated units
- Results should be compared with Callaway-Sant'Anna estimates in 04_cs.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Create output directories
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_panel():
    """Load and prepare panel data for estimation."""
    panel = pd.read_csv(DATA_DIR / "panel.csv")

    # Create log outcome (add small constant to handle zeros)
    panel['log_fatalities'] = np.log(panel['fatalities'] + 1)
    panel['log_fatality_rate'] = np.log(panel['fatalities_per_100m_vmt'] + 0.01)

    # Create numeric state ID for fixed effects
    panel['state_id'] = pd.Categorical(panel['state_abbrev']).codes

    print(f"Loaded panel: {len(panel)} observations")
    print(f"  States: {panel['state_abbrev'].nunique()}")
    print(f"  Years: {panel['year'].min()}-{panel['year'].max()}")
    print(f"  Treated obs: {panel['treated'].sum()}")
    print(f"  Control obs: {(panel['treated'] == 0).sum()}")

    return panel


def twfe_manual(panel, outcome_col, weight_col=None):
    """
    Estimate TWFE using manual demeaning approach with statsmodels.
    This allows for clustered standard errors at the state level.

    Y_it - Y_i_bar - Y_t_bar + Y_bar = β(D_it - D_i_bar - D_t_bar + D_bar) + ε_it
    """
    import statsmodels.api as sm

    df = panel.copy()

    # Demean outcome and treatment by entity and time
    for col in [outcome_col, 'treated']:
        # Entity mean
        entity_mean = df.groupby('state_abbrev')[col].transform('mean')
        # Time mean
        time_mean = df.groupby('year')[col].transform('mean')
        # Grand mean
        grand_mean = df[col].mean()
        # Demeaned variable
        df[f'{col}_demean'] = df[col] - entity_mean - time_mean + grand_mean

    # Weights
    if weight_col:
        weights = df[weight_col]
    else:
        weights = None

    # OLS on demeaned data
    y = df[f'{outcome_col}_demean']
    X = sm.add_constant(df['treated_demean'])

    model = sm.OLS(y, X)

    # Fit with clustered standard errors
    # Create cluster variable
    clusters = pd.Categorical(df['state_abbrev']).codes

    results = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})

    return results


def twfe_linearmodels(panel, outcome_col, weight_col=None):
    """
    Estimate TWFE using linearmodels PanelOLS.
    Provides entity and time fixed effects with clustered SEs.
    """
    from linearmodels.panel import PanelOLS

    df = panel.copy()
    df = df.set_index(['state_abbrev', 'year'])

    y = df[outcome_col]
    X = df[['treated']]

    if weight_col:
        weights = df[weight_col]
        model = PanelOLS(y, X, entity_effects=True, time_effects=True, weights=weights)
    else:
        model = PanelOLS(y, X, entity_effects=True, time_effects=True)

    results = model.fit(cov_type='clustered', cluster_entity=True)

    return results


def format_results(results, model_name, outcome_name, method='linearmodels'):
    """Format regression results for display."""
    if method == 'linearmodels':
        coef = results.params['treated']
        se = results.std_errors['treated']
        tstat = results.tstats['treated']
        pval = results.pvalues['treated']
        n = results.nobs
        r2 = results.rsquared_overall
    else:  # statsmodels
        coef = results.params['treated_demean']
        se = results.bse['treated_demean']
        tstat = results.tvalues['treated_demean']
        pval = results.pvalues['treated_demean']
        n = results.nobs
        r2 = results.rsquared

    # Significance stars
    if pval < 0.01:
        stars = '***'
    elif pval < 0.05:
        stars = '**'
    elif pval < 0.1:
        stars = '*'
    else:
        stars = ''

    return {
        'model': model_name,
        'outcome': outcome_name,
        'coefficient': coef,
        'std_error': se,
        't_statistic': tstat,
        'p_value': pval,
        'significance': stars,
        'n_obs': int(n),
        'r_squared': r2
    }


def print_regression_table(results_list):
    """Print formatted regression table."""
    print("\n" + "="*80)
    print("TWO-WAY FIXED EFFECTS REGRESSION RESULTS")
    print("="*80)
    print("\nDependent Variable: Traffic Fatality Rate (per 100 million VMT)")
    print("Fixed Effects: State + Year")
    print("Standard Errors: Clustered by State")
    print("-"*80)

    # Header
    print(f"{'Model':<30} {'Coef':>10} {'SE':>10} {'t-stat':>10} {'p-val':>10}")
    print("-"*80)

    for r in results_list:
        coef_str = f"{r['coefficient']:.4f}{r['significance']}"
        print(f"{r['model']:<30} {coef_str:>10} {r['std_error']:>10.4f} "
              f"{r['t_statistic']:>10.2f} {r['p_value']:>10.4f}")

    print("-"*80)
    print("Significance: *** p<0.01, ** p<0.05, * p<0.1")
    print(f"N = {results_list[0]['n_obs']} state-year observations")
    print("="*80)


def create_coefficient_plot(results_list):
    """Create coefficient comparison plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    models = [r['model'] for r in results_list]
    coefs = [r['coefficient'] for r in results_list]
    ses = [r['std_error'] for r in results_list]

    # 95% CI
    ci_low = [c - 1.96*s for c, s in zip(coefs, ses)]
    ci_high = [c + 1.96*s for c, s in zip(coefs, ses)]

    y_pos = range(len(models))

    ax.errorbar(coefs, y_pos, xerr=[np.array(coefs)-np.array(ci_low),
                                      np.array(ci_high)-np.array(coefs)],
                fmt='o', capsize=5, capthick=2, markersize=8, color='steelblue')

    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Null (β=0)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Treatment Effect (Fatalities per 100M VMT)', fontsize=12)
    ax.set_title('TWFE Estimates: Effect of Primary Enforcement Handheld Bans\n'
                 '(95% Confidence Intervals, Clustered SEs)', fontsize=12)

    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'twfe_coef_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nCoefficient plot saved to {FIGURES_DIR / 'twfe_coef_plot.png'}")


def main():
    print("="*80)
    print("02_twfe.py - Two-Way Fixed Effects Estimation")
    print("="*80)

    # Load data
    print("\n[1] Loading panel data...")
    panel = load_panel()

    # Check for linearmodels
    try:
        from linearmodels.panel import PanelOLS
        use_linearmodels = True
        print("\n[2] Using linearmodels.PanelOLS for estimation")
    except ImportError:
        use_linearmodels = False
        print("\n[2] linearmodels not available, using manual demeaning approach")

    # Store results
    all_results = []

    # =========================================================================
    # Model 1: Basic TWFE - Fatality Rate
    # =========================================================================
    print("\n[3] Estimating TWFE models...")

    if use_linearmodels:
        res1 = twfe_linearmodels(panel, 'fatalities_per_100m_vmt')
        print("\n" + "="*60)
        print("Model 1: Basic TWFE (Fatality Rate per 100M VMT)")
        print("="*60)
        print(res1.summary.tables[1])
    else:
        res1 = twfe_manual(panel, 'fatalities_per_100m_vmt')
        print(res1.summary())

    r1 = format_results(res1, 'Basic TWFE', 'Fatality Rate',
                        'linearmodels' if use_linearmodels else 'statsmodels')
    all_results.append(r1)

    # =========================================================================
    # Model 2: TWFE with VMT weights (population-weighted)
    # =========================================================================
    panel['vmt_weight'] = panel['vmt'] / panel['vmt'].sum()

    if use_linearmodels:
        res2 = twfe_linearmodels(panel, 'fatalities_per_100m_vmt', 'vmt_weight')
        print("\n" + "="*60)
        print("Model 2: TWFE Weighted by VMT")
        print("="*60)
        print(res2.summary.tables[1])

        r2 = format_results(res2, 'TWFE (VMT-weighted)', 'Fatality Rate', 'linearmodels')
        all_results.append(r2)

    # =========================================================================
    # Model 3: TWFE - Log Fatality Rate
    # =========================================================================
    if use_linearmodels:
        res3 = twfe_linearmodels(panel, 'log_fatality_rate')
        print("\n" + "="*60)
        print("Model 3: TWFE (Log Fatality Rate)")
        print("="*60)
        print(res3.summary.tables[1])

        r3 = format_results(res3, 'TWFE (Log Rate)', 'Log Fatality Rate', 'linearmodels')
        all_results.append(r3)

    # =========================================================================
    # Model 4: TWFE - Total Fatalities (levels)
    # =========================================================================
    if use_linearmodels:
        res4 = twfe_linearmodels(panel, 'fatalities')
        print("\n" + "="*60)
        print("Model 4: TWFE (Total Fatalities)")
        print("="*60)
        print(res4.summary.tables[1])

        r4 = format_results(res4, 'TWFE (Total Fatalities)', 'Fatalities', 'linearmodels')
        all_results.append(r4)

    # =========================================================================
    # Summary Table
    # =========================================================================
    print_regression_table(all_results)

    # =========================================================================
    # Interpretation
    # =========================================================================
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    main_coef = all_results[0]['coefficient']
    main_se = all_results[0]['std_error']
    main_pval = all_results[0]['p_value']

    # Calculate baseline mean for treated states pre-treatment
    pre_treat_mean = panel[
        (panel['ever_treated_in_study'] == 1) &
        (panel['treated'] == 0)
    ]['fatalities_per_100m_vmt'].mean()

    pct_effect = (main_coef / pre_treat_mean) * 100

    print(f"""
Primary Enforcement Effect (TWFE Estimate):
  Coefficient: {main_coef:.4f} fatalities per 100M VMT
  Standard Error: {main_se:.4f} (clustered by state)
  p-value: {main_pval:.4f}

Interpretation:
  - Pre-treatment mean (treated states): {pre_treat_mean:.3f} fatalities per 100M VMT
  - Estimated effect: {pct_effect:.1f}% change in fatality rate
  - Direction: {'Decrease' if main_coef < 0 else 'Increase'} in fatalities

Statistical Significance:
  - {'Statistically significant at 5% level' if main_pval < 0.05 else 'Not statistically significant at 5% level'}

CAUTION:
  This TWFE estimate may be biased due to:
  1. Staggered treatment adoption (14 different cohorts)
  2. Potential heterogeneous treatment effects over time
  3. "Forbidden comparisons" using early-treated as controls for late-treated

  → Compare with Goodman-Bacon decomposition (03_goodman_bacon.py)
  → Compare with Callaway-Sant'Anna estimates (04_cs.py)
""")

    # =========================================================================
    # Save Results
    # =========================================================================
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TABLES_DIR / 'twfe_results.csv', index=False)
    print(f"\nResults saved to {TABLES_DIR / 'twfe_results.csv'}")

    # Create coefficient plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        create_coefficient_plot(all_results)
    except ImportError:
        print("matplotlib not available, skipping coefficient plot")

    print("\n" + "="*80)
    print("TWFE estimation complete!")
    print("="*80)

    return all_results


if __name__ == "__main__":
    results = main()
