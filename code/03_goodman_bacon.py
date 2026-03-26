"""
03_goodman_bacon.py - Goodman-Bacon Decomposition
==================================================

Purpose:
--------
This script implements the Goodman-Bacon (2021) decomposition to diagnose
potential bias in the TWFE estimator. The decomposition reveals which 2x2
DiD comparisons contribute to the overall TWFE estimate and their weights.

Inputs:
-------
- data/processed/panel.csv: Clean panel dataset from 01_clean.py

Outputs:
--------
- output/figures/bacon_decomposition.png: Scatter plot of 2x2 DiD estimates
- output/figures/bacon_weights_by_type.png: Weight distribution by comparison type
- output/tables/bacon_decomposition.csv: Full decomposition results
- output/tables/bacon_summary.csv: Summary by comparison type

Background:
-----------
Goodman-Bacon (2021) shows that the TWFE DiD estimator is a weighted average
of all possible 2x2 DiD comparisons:

    β_TWFE = Σ_k w_k * β_k

Comparison types:
1. Treated vs Never-Treated (valid - uses clean controls)
2. Earlier Treated vs Later Treated (uses later-treated as controls before they treat)
3. Later Treated vs Earlier Treated (PROBLEMATIC - uses already-treated as controls)

The "Later vs Earlier" comparisons can bias TWFE when treatment effects are dynamic.

References:
-----------
Goodman-Bacon, A. (2021). Difference-in-differences with variation in
treatment timing. Journal of Econometrics, 225(2), 254-277.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"


def load_panel():
    """Load panel data."""
    panel = pd.read_csv(DATA_DIR / "panel.csv")
    print(f"Loaded panel: {len(panel)} observations")
    return panel


def get_cohorts(panel):
    """Extract treatment cohorts from panel."""
    cohorts = panel.groupby('state_abbrev')['first_treat_year'].first().reset_index()
    cohorts.columns = ['state', 'cohort']

    # Separate never-treated (cohort = 0) and treated
    never_treated = cohorts[cohorts['cohort'] == 0]['state'].tolist()
    treated_cohorts = cohorts[cohorts['cohort'] > 0].copy()

    print(f"\nCohort structure:")
    print(f"  Never-treated states: {len(never_treated)}")
    print(f"  Treated states: {len(treated_cohorts)}")
    print(f"  Unique treatment years: {sorted(treated_cohorts['cohort'].unique())}")

    return cohorts, never_treated, treated_cohorts


def compute_2x2_did(panel, treat_states, control_states, treat_year,
                    outcome_col='fatalities_per_100m_vmt'):
    """
    Compute a simple 2x2 DiD estimate.

    DiD = (Y_treat_post - Y_treat_pre) - (Y_control_post - Y_control_pre)

    Parameters:
    -----------
    panel : DataFrame
    treat_states : list of state abbreviations in treatment group
    control_states : list of state abbreviations in control group
    treat_year : year treatment begins for treat_states
    outcome_col : outcome variable

    Returns:
    --------
    did_estimate : float
    n_treat : int (number of treated states)
    n_control : int (number of control states)
    """
    # Treatment group: pre and post
    treat_data = panel[panel['state_abbrev'].isin(treat_states)]
    treat_pre = treat_data[treat_data['year'] < treat_year][outcome_col].mean()
    treat_post = treat_data[treat_data['year'] >= treat_year][outcome_col].mean()

    # Control group: pre and post (relative to treat_year)
    control_data = panel[panel['state_abbrev'].isin(control_states)]
    control_pre = control_data[control_data['year'] < treat_year][outcome_col].mean()
    control_post = control_data[control_data['year'] >= treat_year][outcome_col].mean()

    # DiD estimate
    did = (treat_post - treat_pre) - (control_post - control_pre)

    return did, len(treat_states), len(control_states)


def compute_weight(n_treat, n_control, t_pre, t_post, T):
    """
    Compute Goodman-Bacon weight for a 2x2 comparison.

    Weight proportional to:
    - Group sizes (n_treat * n_control)
    - Variance of treatment (more variation = higher weight)

    Simplified weight formula from Goodman-Bacon (2021):
    w ∝ n_k * n_l * V(D_kl)

    where V(D_kl) depends on timing.
    """
    n_total = n_treat + n_control

    # Share of each group
    s_treat = n_treat / n_total
    s_control = n_control / n_total

    # Variance of treatment dummy (depends on pre/post split)
    # V(D) = p(1-p) where p = fraction of post-treatment periods
    p = t_post / T
    var_d = p * (1 - p)

    # Weight
    weight = s_treat * s_control * var_d

    return weight


def bacon_decomposition(panel, outcome_col='fatalities_per_100m_vmt'):
    """
    Perform the full Goodman-Bacon decomposition.

    Returns DataFrame with all 2x2 comparisons, their DiD estimates, and weights.
    """
    cohorts, never_treated, treated_cohorts = get_cohorts(panel)

    # Get unique treatment years
    treat_years = sorted(treated_cohorts['cohort'].unique())
    T = panel['year'].nunique()  # Total time periods
    min_year = panel['year'].min()
    max_year = panel['year'].max()

    decomposition = []

    # ==========================================================================
    # Type 1: Treated vs Never-Treated
    # ==========================================================================
    print("\nComputing 2x2 DiD estimates...")
    print("  Type 1: Treated vs Never-Treated")

    if len(never_treated) > 0:
        for g in treat_years:
            # States treated in year g
            g_states = treated_cohorts[treated_cohorts['cohort'] == g]['state'].tolist()

            # DiD: g_states (treated at g) vs never_treated
            did, n_t, n_c = compute_2x2_did(panel, g_states, never_treated, g, outcome_col)

            # Weight calculation
            t_pre = g - min_year  # periods before treatment
            t_post = max_year - g + 1  # periods after treatment (inclusive)
            weight = compute_weight(n_t, n_c, t_pre, t_post, T)

            decomposition.append({
                'comparison_type': 'Treated vs Never-Treated',
                'treat_cohort': g,
                'control_cohort': 'Never',
                'did_estimate': did,
                'weight': weight,
                'n_treat': n_t,
                'n_control': n_c,
                't_pre': t_pre,
                't_post': t_post
            })

    # ==========================================================================
    # Type 2 & 3: Timing comparisons between treated cohorts
    # ==========================================================================
    print("  Type 2: Earlier vs Later Treated")
    print("  Type 3: Later vs Earlier Treated")

    for g_early, g_late in combinations(treat_years, 2):
        # Ensure g_early < g_late
        if g_early > g_late:
            g_early, g_late = g_late, g_early

        early_states = treated_cohorts[treated_cohorts['cohort'] == g_early]['state'].tolist()
        late_states = treated_cohorts[treated_cohorts['cohort'] == g_late]['state'].tolist()

        # ----- Type 2: Earlier treated vs Later treated (before late treats) -----
        # Using late-treated as controls BEFORE they get treated
        # Valid comparison window: g_early to g_late-1

        # Subset panel to window where late_states are not yet treated
        panel_window = panel[panel['year'] < g_late].copy()

        if len(panel_window) > 0 and g_early <= panel_window['year'].max():
            did_early, n_t, n_c = compute_2x2_did(
                panel_window, early_states, late_states, g_early, outcome_col
            )

            # Time periods in this comparison
            t_pre = g_early - min_year
            t_post = g_late - g_early  # periods between early and late treatment

            weight = compute_weight(n_t, n_c, t_pre, t_post, T)

            decomposition.append({
                'comparison_type': 'Earlier vs Later Treated',
                'treat_cohort': g_early,
                'control_cohort': g_late,
                'did_estimate': did_early,
                'weight': weight,
                'n_treat': n_t,
                'n_control': n_c,
                't_pre': t_pre,
                't_post': t_post
            })

        # ----- Type 3: Later treated vs Earlier treated (PROBLEMATIC) -----
        # Using already-treated early_states as controls for late_states
        # This is the "forbidden comparison"

        # Window: from g_late onwards (both groups treated, but late just started)
        panel_window = panel[panel['year'] >= g_early].copy()

        if len(panel_window) > 0:
            did_late, n_t, n_c = compute_2x2_did(
                panel_window, late_states, early_states, g_late, outcome_col
            )

            t_pre = g_late - g_early  # periods where early is treated but late is not
            t_post = max_year - g_late + 1

            weight = compute_weight(n_t, n_c, t_pre, t_post, T)

            decomposition.append({
                'comparison_type': 'Later vs Earlier Treated',
                'treat_cohort': g_late,
                'control_cohort': g_early,
                'did_estimate': did_late,
                'weight': weight,
                'n_treat': n_t,
                'n_control': n_c,
                't_pre': t_pre,
                't_post': t_post
            })

    # Create DataFrame
    decomp_df = pd.DataFrame(decomposition)

    # Normalize weights to sum to 1
    decomp_df['weight_normalized'] = decomp_df['weight'] / decomp_df['weight'].sum()

    # Compute weighted average (should approximate TWFE)
    twfe_approx = (decomp_df['did_estimate'] * decomp_df['weight_normalized']).sum()

    print(f"\n  Total comparisons: {len(decomp_df)}")
    print(f"  Weighted average (≈ TWFE): {twfe_approx:.4f}")

    return decomp_df, twfe_approx


def summarize_by_type(decomp_df):
    """Summarize decomposition by comparison type."""
    summary = decomp_df.groupby('comparison_type').agg({
        'weight_normalized': 'sum',
        'did_estimate': ['mean', 'std', 'min', 'max'],
        'treat_cohort': 'count'
    }).round(4)

    summary.columns = ['total_weight', 'mean_estimate', 'std_estimate',
                       'min_estimate', 'max_estimate', 'n_comparisons']

    # Weighted mean estimate by type
    weighted_means = []
    for comp_type in decomp_df['comparison_type'].unique():
        subset = decomp_df[decomp_df['comparison_type'] == comp_type]
        if subset['weight_normalized'].sum() > 0:
            wm = (subset['did_estimate'] * subset['weight_normalized']).sum() / subset['weight_normalized'].sum()
        else:
            wm = np.nan
        weighted_means.append({'comparison_type': comp_type, 'weighted_mean': wm})

    wm_df = pd.DataFrame(weighted_means).set_index('comparison_type')
    summary = summary.join(wm_df)

    return summary.reset_index()


def create_decomposition_plot(decomp_df, twfe_estimate):
    """Create the classic Goodman-Bacon decomposition scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color mapping for comparison types
    colors = {
        'Treated vs Never-Treated': '#2ecc71',  # Green - valid
        'Earlier vs Later Treated': '#3498db',   # Blue - valid
        'Later vs Earlier Treated': '#e74c3c'    # Red - problematic
    }

    # Plot each comparison type
    for comp_type in decomp_df['comparison_type'].unique():
        subset = decomp_df[decomp_df['comparison_type'] == comp_type]
        ax.scatter(
            subset['weight_normalized'],
            subset['did_estimate'],
            c=colors.get(comp_type, 'gray'),
            s=subset['weight_normalized'] * 3000 + 50,  # Size proportional to weight
            alpha=0.7,
            label=f"{comp_type} (n={len(subset)})",
            edgecolors='white',
            linewidth=0.5
        )

    # Add TWFE estimate line
    ax.axhline(y=twfe_estimate, color='black', linestyle='--', linewidth=2,
               label=f'TWFE Estimate: {twfe_estimate:.4f}')

    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('Weight (Normalized)', fontsize=12)
    ax.set_ylabel('2×2 DiD Estimate\n(Fatalities per 100M VMT)', fontsize=12)
    ax.set_title('Goodman-Bacon Decomposition of TWFE Estimate\n'
                 'Effect of Primary Enforcement Handheld Bans', fontsize=14)

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bacon_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nDecomposition plot saved to {FIGURES_DIR / 'bacon_decomposition.png'}")


def create_weight_barplot(summary_df):
    """Create bar plot showing weight distribution by comparison type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Colors
    colors = {
        'Treated vs Never-Treated': '#2ecc71',
        'Earlier vs Later Treated': '#3498db',
        'Later vs Earlier Treated': '#e74c3c'
    }

    color_list = [colors.get(t, 'gray') for t in summary_df['comparison_type']]

    # Plot 1: Weight distribution
    ax1 = axes[0]
    bars = ax1.barh(summary_df['comparison_type'], summary_df['total_weight'],
                    color=color_list, edgecolor='white')
    ax1.set_xlabel('Share of Total Weight', fontsize=11)
    ax1.set_title('Weight Distribution by Comparison Type', fontsize=12)
    ax1.set_xlim(0, 1)

    # Add percentage labels
    for bar, weight in zip(bars, summary_df['total_weight']):
        ax1.text(weight + 0.02, bar.get_y() + bar.get_height()/2,
                f'{weight*100:.1f}%', va='center', fontsize=10)

    # Plot 2: Mean estimates by type
    ax2 = axes[1]
    bars = ax2.barh(summary_df['comparison_type'], summary_df['weighted_mean'],
                    color=color_list, edgecolor='white')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Weighted Mean DiD Estimate', fontsize=11)
    ax2.set_title('Contribution to TWFE by Comparison Type', fontsize=12)

    # Add value labels
    for bar, val in zip(bars, summary_df['weighted_mean']):
        offset = 0.005 if val >= 0 else -0.005
        ha = 'left' if val >= 0 else 'right'
        ax2.text(val + offset, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', ha=ha, fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bacon_weights_by_type.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Weight distribution plot saved to {FIGURES_DIR / 'bacon_weights_by_type.png'}")


def main():
    print("="*80)
    print("03_goodman_bacon.py - Goodman-Bacon Decomposition")
    print("="*80)

    # Load data
    print("\n[1] Loading panel data...")
    panel = load_panel()

    # Perform decomposition
    print("\n[2] Computing Goodman-Bacon decomposition...")
    decomp_df, twfe_approx = bacon_decomposition(panel)

    # Summarize by type
    print("\n[3] Summarizing by comparison type...")
    summary_df = summarize_by_type(decomp_df)

    # Print summary
    print("\n" + "="*80)
    print("DECOMPOSITION SUMMARY")
    print("="*80)

    print("\nWeight Distribution by Comparison Type:")
    print("-"*80)
    for _, row in summary_df.iterrows():
        print(f"  {row['comparison_type']:<30} "
              f"Weight: {row['total_weight']*100:5.1f}%  "
              f"Mean Est: {row['weighted_mean']:7.4f}  "
              f"N: {int(row['n_comparisons'])}")

    print("-"*80)
    print(f"  {'TOTAL':<30} Weight: 100.0%  TWFE ≈ {twfe_approx:.4f}")

    # Diagnostics
    print("\n" + "="*80)
    print("DIAGNOSTIC ASSESSMENT")
    print("="*80)

    # Weight on problematic comparisons
    problem_weight = summary_df[
        summary_df['comparison_type'] == 'Later vs Earlier Treated'
    ]['total_weight'].values

    if len(problem_weight) > 0:
        problem_weight = problem_weight[0]
    else:
        problem_weight = 0

    valid_weight = 1 - problem_weight

    print(f"""
1. WEIGHT ON PROBLEMATIC COMPARISONS:
   - "Later vs Earlier Treated" weight: {problem_weight*100:.1f}%
   - Valid comparisons weight: {valid_weight*100:.1f}%

   Assessment: {"⚠️  CONCERN - Substantial weight on forbidden comparisons"
                if problem_weight > 0.2 else "✓ Most weight on valid comparisons"}

2. HETEROGENEITY IN ESTIMATES:
   - Range of 2x2 estimates: [{decomp_df['did_estimate'].min():.4f}, {decomp_df['did_estimate'].max():.4f}]
   - Std. dev. of estimates: {decomp_df['did_estimate'].std():.4f}

   Assessment: {"⚠️  High heterogeneity suggests dynamic treatment effects"
                if decomp_df['did_estimate'].std() > 0.05 else "✓ Relatively homogeneous estimates"}

3. SIGN CONSISTENCY:
   - Positive estimates: {(decomp_df['did_estimate'] > 0).sum()} ({(decomp_df['did_estimate'] > 0).mean()*100:.0f}%)
   - Negative estimates: {(decomp_df['did_estimate'] < 0).sum()} ({(decomp_df['did_estimate'] < 0).mean()*100:.0f}%)

   Assessment: {"⚠️  Mixed signs - direction of effect unclear"
                if 0.3 < (decomp_df['did_estimate'] > 0).mean() < 0.7
                else "✓ Estimates mostly consistent in sign"}

4. TWFE RELIABILITY:
   - If problematic comparisons have different estimates than valid ones,
     TWFE may be biased.
""")

    # Compare valid vs problematic estimates
    valid_types = ['Treated vs Never-Treated', 'Earlier vs Later Treated']
    valid_est = decomp_df[decomp_df['comparison_type'].isin(valid_types)]['did_estimate'].mean()
    problem_est = decomp_df[decomp_df['comparison_type'] == 'Later vs Earlier Treated']['did_estimate'].mean()

    if not np.isnan(problem_est):
        print(f"   - Mean estimate (valid comparisons): {valid_est:.4f}")
        print(f"   - Mean estimate (problematic comparisons): {problem_est:.4f}")
        print(f"   - Difference: {abs(valid_est - problem_est):.4f}")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("""
Given the staggered adoption across 14 cohorts (2001-2021), the TWFE estimate
may be biased by forbidden comparisons.

NEXT STEPS:
→ Use Callaway & Sant'Anna (2021) estimator in 04_cs.py for unbiased estimates
→ Examine event study in 05_event_study.py to check for dynamic effects
""")

    # Save results
    print("\n[4] Saving results...")
    decomp_df.to_csv(TABLES_DIR / 'bacon_decomposition.csv', index=False)
    summary_df.to_csv(TABLES_DIR / 'bacon_summary.csv', index=False)
    print(f"  Decomposition saved to {TABLES_DIR / 'bacon_decomposition.csv'}")
    print(f"  Summary saved to {TABLES_DIR / 'bacon_summary.csv'}")

    # Create plots
    print("\n[5] Creating visualizations...")
    create_decomposition_plot(decomp_df, twfe_approx)
    create_weight_barplot(summary_df)

    print("\n" + "="*80)
    print("Goodman-Bacon decomposition complete!")
    print("="*80)

    return decomp_df, summary_df


if __name__ == "__main__":
    decomp_df, summary_df = main()
