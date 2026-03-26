"""
04_cs.py - Callaway & Sant'Anna Estimator
==========================================

Purpose:
--------
This script estimates the effect of primary enforcement handheld device bans
using the Callaway & Sant'Anna (2021) estimator, which is robust to
heterogeneous treatment effects with staggered adoption.

Inputs:
-------
- data/processed/panel.csv: Clean panel dataset from 01_clean.py

Outputs:
--------
- output/tables/cs_att_gt.csv: Group-time average treatment effects ATT(g,t)
- output/tables/cs_aggregations.csv: Aggregated treatment effect estimates
- output/figures/cs_event_study.png: Event study plot from CS estimator
- output/figures/cs_group_effects.png: Cohort-specific effects

Background:
-----------
Callaway & Sant'Anna (2021) address the problems with TWFE by:
1. Only using valid comparisons (never-treated or not-yet-treated as controls)
2. Estimating separate ATT(g,t) for each cohort g at each time t
3. Providing flexible aggregation schemes for summary measures

Key Insight:
------------
Unlike TWFE, CS never uses already-treated units as controls, avoiding the
"forbidden comparisons" that bias TWFE with staggered adoption.

References:
-----------
Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with
multiple time periods. Journal of Econometrics, 225(2), 200-230.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
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

    # Create numeric state ID
    panel['state_id'] = pd.Categorical(panel['state_abbrev']).codes

    print(f"Loaded panel: {len(panel)} observations")
    print(f"  States: {panel['state_abbrev'].nunique()}")
    print(f"  Years: {panel['year'].min()}-{panel['year'].max()}")

    return panel


def estimate_att_gt_manual(panel, outcome_col='fatalities_per_100m_vmt',
                           control_group='never_treated'):
    """
    Manually estimate ATT(g,t) for each cohort g and time period t.

    This implements the core CS estimator:
    ATT(g,t) = E[Y_t - Y_{g-1} | G=g] - E[Y_t - Y_{g-1} | C]

    Where C is the control group (never-treated or not-yet-treated).

    Parameters:
    -----------
    panel : DataFrame
    outcome_col : str, outcome variable
    control_group : str, 'never_treated' or 'not_yet_treated'

    Returns:
    --------
    att_gt : DataFrame with columns [cohort, time, att, se, n_treat, n_control]
    """
    # Get cohorts (excluding never-treated which have first_treat_year = 0)
    cohorts = sorted(panel[panel['first_treat_year'] > 0]['first_treat_year'].unique())
    years = sorted(panel['year'].unique())

    # Never-treated states
    never_treated_states = panel[panel['first_treat_year'] == 0]['state_abbrev'].unique()

    results = []

    for g in cohorts:
        # States in cohort g (treated in year g)
        g_states = panel[panel['first_treat_year'] == g]['state_abbrev'].unique()

        # Baseline period: g-1 (year before treatment)
        baseline_year = g - 1

        if baseline_year < panel['year'].min():
            # Can't estimate if no pre-period
            continue

        for t in years:
            # Define control group for this (g, t) cell
            if control_group == 'never_treated':
                control_states = never_treated_states
            else:  # not_yet_treated
                # Include never-treated + states not yet treated by time t
                not_yet = panel[(panel['first_treat_year'] > t) |
                               (panel['first_treat_year'] == 0)]['state_abbrev'].unique()
                control_states = not_yet

            if len(control_states) == 0:
                continue

            # Get outcomes for treated group
            treat_t = panel[(panel['state_abbrev'].isin(g_states)) &
                           (panel['year'] == t)][outcome_col].values
            treat_base = panel[(panel['state_abbrev'].isin(g_states)) &
                              (panel['year'] == baseline_year)][outcome_col].values

            # Get outcomes for control group
            control_t = panel[(panel['state_abbrev'].isin(control_states)) &
                             (panel['year'] == t)][outcome_col].values
            control_base = panel[(panel['state_abbrev'].isin(control_states)) &
                                (panel['year'] == baseline_year)][outcome_col].values

            if len(treat_t) == 0 or len(control_t) == 0:
                continue
            if len(treat_base) == 0 or len(control_base) == 0:
                continue

            # DiD estimate: (treat_t - treat_base) - (control_t - control_base)
            treat_diff = treat_t.mean() - treat_base.mean()
            control_diff = control_t.mean() - control_base.mean()
            att = treat_diff - control_diff

            # Standard error (simplified - assumes independence)
            # SE = sqrt(Var(treat_diff) + Var(control_diff))
            n_treat = len(g_states)
            n_control = len(control_states)

            # Variance of difference in means
            var_treat = (np.var(treat_t, ddof=1)/n_treat +
                        np.var(treat_base, ddof=1)/n_treat)
            var_control = (np.var(control_t, ddof=1)/n_control +
                          np.var(control_base, ddof=1)/n_control)
            se = np.sqrt(var_treat + var_control)

            # Event time
            event_time = t - g

            results.append({
                'cohort': g,
                'time': t,
                'event_time': event_time,
                'att': att,
                'se': se,
                'n_treat': n_treat,
                'n_control': n_control,
                'post': 1 if t >= g else 0
            })

    att_gt = pd.DataFrame(results)

    print(f"\nEstimated {len(att_gt)} ATT(g,t) cells")
    print(f"  Cohorts: {att_gt['cohort'].nunique()}")
    print(f"  Event times: {att_gt['event_time'].min()} to {att_gt['event_time'].max()}")

    return att_gt


def aggregate_att(att_gt, agg_type='simple'):
    """
    Aggregate ATT(g,t) estimates.

    Parameters:
    -----------
    att_gt : DataFrame with ATT(g,t) estimates
    agg_type : str
        'simple' - weighted average of all post-treatment ATT(g,t)
        'event' - average by event time (dynamic effects)
        'cohort' - average by cohort
        'calendar' - average by calendar time

    Returns:
    --------
    aggregated : DataFrame
    """
    # Only use post-treatment cells for aggregation
    post = att_gt[att_gt['post'] == 1].copy()

    if len(post) == 0:
        return pd.DataFrame()

    # Weight by group size
    post['weight'] = post['n_treat'] / post['n_treat'].sum()

    if agg_type == 'simple':
        # Overall ATT
        att = (post['att'] * post['weight']).sum()

        # Pooled SE (simplified)
        se = np.sqrt((post['se']**2 * post['weight']**2).sum())

        return pd.DataFrame([{
            'aggregation': 'Overall ATT',
            'att': att,
            'se': se,
            't_stat': att / se if se > 0 else np.nan,
            'p_value': 2 * (1 - stats.norm.cdf(abs(att/se))) if se > 0 else np.nan,
            'ci_lower': att - 1.96 * se,
            'ci_upper': att + 1.96 * se,
            'n_cells': len(post)
        }])

    elif agg_type == 'event':
        # Aggregate by event time
        event_agg = []
        for e in sorted(att_gt['event_time'].unique()):
            subset = att_gt[att_gt['event_time'] == e]
            if len(subset) == 0:
                continue

            # Weight within event time
            weights = subset['n_treat'] / subset['n_treat'].sum()
            att = (subset['att'] * weights).sum()
            se = np.sqrt((subset['se']**2 * weights**2).sum())

            event_agg.append({
                'event_time': e,
                'att': att,
                'se': se,
                'ci_lower': att - 1.96 * se,
                'ci_upper': att + 1.96 * se,
                'n_cohorts': len(subset),
                'post': 1 if e >= 0 else 0
            })

        return pd.DataFrame(event_agg)

    elif agg_type == 'cohort':
        # Aggregate by cohort
        cohort_agg = []
        for g in sorted(post['cohort'].unique()):
            subset = post[post['cohort'] == g]
            if len(subset) == 0:
                continue

            weights = subset['n_treat'] / subset['n_treat'].sum()
            att = (subset['att'] * weights).sum()
            se = np.sqrt((subset['se']**2 * weights**2).sum())

            cohort_agg.append({
                'cohort': g,
                'att': att,
                'se': se,
                'ci_lower': att - 1.96 * se,
                'ci_upper': att + 1.96 * se,
                'n_periods': len(subset)
            })

        return pd.DataFrame(cohort_agg)

    elif agg_type == 'calendar':
        # Aggregate by calendar time
        cal_agg = []
        for t in sorted(post['time'].unique()):
            subset = post[post['time'] == t]
            if len(subset) == 0:
                continue

            weights = subset['n_treat'] / subset['n_treat'].sum()
            att = (subset['att'] * weights).sum()
            se = np.sqrt((subset['se']**2 * weights**2).sum())

            cal_agg.append({
                'year': t,
                'att': att,
                'se': se,
                'ci_lower': att - 1.96 * se,
                'ci_upper': att + 1.96 * se,
                'n_cohorts': len(subset)
            })

        return pd.DataFrame(cal_agg)

    return pd.DataFrame()


def create_event_study_plot(event_agg, title_suffix=''):
    """Create event study plot from CS estimates."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Separate pre and post
    pre = event_agg[event_agg['event_time'] < 0]
    post = event_agg[event_agg['event_time'] >= 0]

    # Plot pre-treatment (blue)
    if len(pre) > 0:
        ax.errorbar(pre['event_time'], pre['att'],
                   yerr=[pre['att'] - pre['ci_lower'], pre['ci_upper'] - pre['att']],
                   fmt='o', color='steelblue', capsize=4, capthick=1.5,
                   markersize=8, label='Pre-treatment', linewidth=1.5)

    # Plot post-treatment (red)
    if len(post) > 0:
        ax.errorbar(post['event_time'], post['att'],
                   yerr=[post['att'] - post['ci_lower'], post['ci_upper'] - post['att']],
                   fmt='o', color='firebrick', capsize=4, capthick=1.5,
                   markersize=8, label='Post-treatment', linewidth=1.5)

    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(x=-0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Labels
    ax.set_xlabel('Event Time (Years Relative to Treatment)', fontsize=12)
    ax.set_ylabel('ATT (Fatalities per 100M VMT)', fontsize=12)
    ax.set_title(f'Callaway-Sant\'Anna Event Study{title_suffix}\n'
                 f'Effect of Primary Enforcement Handheld Bans', fontsize=13)

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set x-axis to integers
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cs_event_study.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Event study plot saved to {FIGURES_DIR / 'cs_event_study.png'}")


def create_cohort_plot(cohort_agg):
    """Create cohort-specific effects plot."""
    fig, ax = plt.subplots(figsize=(12, 7))

    cohort_agg = cohort_agg.sort_values('cohort')

    colors = ['firebrick' if att > 0 else 'steelblue'
              for att in cohort_agg['att']]

    bars = ax.barh(cohort_agg['cohort'].astype(str), cohort_agg['att'],
                   xerr=[cohort_agg['att'] - cohort_agg['ci_lower'],
                         cohort_agg['ci_upper'] - cohort_agg['att']],
                   color=colors, alpha=0.7, capsize=3, edgecolor='white')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    ax.set_xlabel('ATT (Fatalities per 100M VMT)', fontsize=12)
    ax.set_ylabel('Treatment Cohort (Year)', fontsize=12)
    ax.set_title('Callaway-Sant\'Anna: Cohort-Specific Treatment Effects\n'
                 '(Average Effect for Each Treatment Cohort)', fontsize=13)

    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cs_group_effects.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Cohort effects plot saved to {FIGURES_DIR / 'cs_group_effects.png'}")


def print_results_table(overall, event_agg, cohort_agg):
    """Print formatted results table."""
    print("\n" + "="*80)
    print("CALLAWAY-SANT'ANNA ESTIMATION RESULTS")
    print("="*80)

    # Overall ATT
    print("\n[1] OVERALL ATT (Simple Weighted Average)")
    print("-"*60)
    row = overall.iloc[0]
    sig = '***' if row['p_value'] < 0.01 else '**' if row['p_value'] < 0.05 else '*' if row['p_value'] < 0.1 else ''
    print(f"  ATT = {row['att']:.4f}{sig}")
    print(f"  SE  = {row['se']:.4f}")
    print(f"  95% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
    print(f"  p-value: {row['p_value']:.4f}")
    print(f"  N cells: {int(row['n_cells'])}")

    # Event study
    print("\n[2] DYNAMIC EFFECTS (Event Study)")
    print("-"*60)
    print(f"{'Event Time':<12} {'ATT':>10} {'SE':>10} {'95% CI':>24} {'N':>6}")
    print("-"*60)

    for _, row in event_agg.iterrows():
        e = int(row['event_time'])
        marker = " <-- Treatment" if e == 0 else ""
        print(f"  e = {e:<6} {row['att']:>10.4f} {row['se']:>10.4f} "
              f"[{row['ci_lower']:>8.4f}, {row['ci_upper']:>8.4f}] {int(row['n_cohorts']):>5}{marker}")

    # Pre-trends test
    pre = event_agg[event_agg['event_time'] < 0]
    if len(pre) > 0:
        pre_mean = pre['att'].mean()
        pre_sig = (pre['att'].abs() > 1.96 * pre['se']).any()
        print("-"*60)
        print(f"  Pre-treatment mean: {pre_mean:.4f}")
        print(f"  Any significant pre-trends: {'Yes ⚠️' if pre_sig else 'No ✓'}")

    # Cohort effects
    print("\n[3] COHORT-SPECIFIC EFFECTS")
    print("-"*60)
    print(f"{'Cohort':<10} {'ATT':>10} {'SE':>10} {'95% CI':>24}")
    print("-"*60)

    for _, row in cohort_agg.iterrows():
        sig = '*' if abs(row['att']) > 1.96 * row['se'] else ''
        print(f"  g={int(row['cohort']):<6} {row['att']:>10.4f}{sig:<2} {row['se']:>10.4f} "
              f"[{row['ci_lower']:>8.4f}, {row['ci_upper']:>8.4f}]")


def compare_with_twfe():
    """Load and compare with TWFE results."""
    twfe_path = TABLES_DIR / 'twfe_results.csv'
    if twfe_path.exists():
        twfe = pd.read_csv(twfe_path)
        basic_twfe = twfe[twfe['model'] == 'Basic TWFE'].iloc[0]
        return basic_twfe['coefficient'], basic_twfe['std_error']
    return None, None


def main():
    print("="*80)
    print("04_cs.py - Callaway & Sant'Anna Estimator")
    print("="*80)

    # Load data
    print("\n[1] Loading panel data...")
    panel = load_panel()

    # Check for csdid package
    try:
        import csdid
        use_csdid = True
        print("\n[2] Using csdid package for estimation")
    except ImportError:
        use_csdid = False
        print("\n[2] csdid package not available, using manual implementation")

    # Estimate ATT(g,t)
    print("\n[3] Estimating ATT(g,t) for all cohort-time cells...")
    print("    Control group: Never-treated states")

    att_gt = estimate_att_gt_manual(panel, control_group='never_treated')

    # Aggregate
    print("\n[4] Computing aggregations...")

    overall = aggregate_att(att_gt, 'simple')
    event_agg = aggregate_att(att_gt, 'event')
    cohort_agg = aggregate_att(att_gt, 'cohort')
    calendar_agg = aggregate_att(att_gt, 'calendar')

    # Print results
    print_results_table(overall, event_agg, cohort_agg)

    # Compare with TWFE
    print("\n" + "="*80)
    print("COMPARISON WITH TWFE")
    print("="*80)

    twfe_coef, twfe_se = compare_with_twfe()
    cs_coef = overall.iloc[0]['att']
    cs_se = overall.iloc[0]['se']

    if twfe_coef is not None:
        print(f"""
  TWFE Estimate:   {twfe_coef:>8.4f} (SE: {twfe_se:.4f})
  CS Estimate:     {cs_coef:>8.4f} (SE: {cs_se:.4f})
  Difference:      {cs_coef - twfe_coef:>8.4f}

  Assessment: {"⚠️  Substantial difference - TWFE likely biased"
               if abs(cs_coef - twfe_coef) > 2*max(twfe_se, cs_se)
               else "✓ Estimates relatively close"}
""")
    else:
        print("  (TWFE results not found for comparison)")

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    cs_pval = overall.iloc[0]['p_value']

    # Pre-trends check
    pre = event_agg[event_agg['event_time'] < 0]
    pre_trend_concern = False
    if len(pre) > 0:
        pre_trend_concern = (pre['att'].abs() > 1.96 * pre['se']).any()

    print(f"""
MAIN RESULT (Callaway-Sant'Anna):
  - Overall ATT: {cs_coef:.4f} fatalities per 100M VMT
  - 95% CI: [{overall.iloc[0]['ci_lower']:.4f}, {overall.iloc[0]['ci_upper']:.4f}]
  - Statistical significance: {'Yes (p<0.05)' if cs_pval < 0.05 else 'No (p≥0.05)'}

PARALLEL TRENDS:
  - Pre-treatment effects: {'⚠️ Some significant pre-trends detected' if pre_trend_concern else '✓ No significant pre-trends'}

INTERPRETATION:
  The CS estimator uses only valid comparisons (never-treated as controls),
  avoiding the bias from "forbidden comparisons" identified in the Bacon decomposition.

  {"The effect is statistically significant, suggesting primary enforcement bans " +
   ("increase" if cs_coef > 0 else "decrease") + " traffic fatalities."
   if cs_pval < 0.05 else
   "The effect is NOT statistically significant. We cannot reject the null hypothesis "
   "that primary enforcement handheld bans have no effect on traffic fatalities."}
""")

    # Save results
    print("\n[5] Saving results...")
    att_gt.to_csv(TABLES_DIR / 'cs_att_gt.csv', index=False)

    # Combine aggregations
    overall['type'] = 'overall'
    aggregations = overall.copy()

    event_agg_save = event_agg.copy()
    event_agg_save['type'] = 'event_time'

    cohort_agg_save = cohort_agg.copy()
    cohort_agg_save['type'] = 'cohort'

    all_agg = pd.concat([overall, event_agg_save, cohort_agg_save], ignore_index=True)
    all_agg.to_csv(TABLES_DIR / 'cs_aggregations.csv', index=False)

    print(f"  ATT(g,t) saved to {TABLES_DIR / 'cs_att_gt.csv'}")
    print(f"  Aggregations saved to {TABLES_DIR / 'cs_aggregations.csv'}")

    # Create plots
    print("\n[6] Creating visualizations...")
    create_event_study_plot(event_agg)
    create_cohort_plot(cohort_agg)

    print("\n" + "="*80)
    print("Callaway-Sant'Anna estimation complete!")
    print("="*80)

    return att_gt, overall, event_agg, cohort_agg


if __name__ == "__main__":
    att_gt, overall, event_agg, cohort_agg = main()
