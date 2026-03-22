"""
05_event_study.py - Event Study Analysis
==========================================

Purpose:
--------
This script produces event study plots to visualize treatment effect dynamics
and assess the parallel trends assumption. It compares naive TWFE event studies
with robust event studies from the Callaway & Sant'Anna estimator.

Inputs:
-------
- data/processed/panel_data.csv: Clean panel dataset from 01_clean.py
- Optionally: CS estimates from 04_cs.py

Outputs:
--------
- output/figures/event_study_twfe.png: Traditional TWFE event study
- output/figures/event_study_cs.png: Robust CS event study
- output/figures/event_study_comparison.png: Side-by-side comparison
- output/tables/event_study_coefficients.csv: Point estimates and CIs by event time

Event Study Specification:
--------------------------
TWFE version:
Y_it = α_i + λ_t + Σ_k β_k * 1(t - G_i = k) + X_it'γ + ε_it

Where:
- G_i: Treatment year for unit i (treatment cohort)
- k: Event time (years relative to treatment)
- β_k: Effect at event time k (normalized to 0 at k = -1)

Key Elements:
-------------
1. Pre-treatment coefficients (k < 0): Test parallel trends
   - Should be statistically indistinguishable from zero
   - Trending pre-treatment effects suggest violation

2. Treatment onset (k = 0): Immediate effect
   - First period under the new law

3. Post-treatment coefficients (k > 0): Dynamic effects
   - How effects evolve over time
   - Potential for effect growth or fade-out

4. Reference period (typically k = -1): Normalized to zero
   - All coefficients relative to this baseline

Diagnostics:
------------
- Pre-trends test: Joint F-test that all pre-treatment coefficients = 0
- Effect persistence: Do post-treatment effects remain stable?
- Anticipation effects: Any effects before treatment? (k = -1, -2)

Comparison with CS:
-------------------
The Callaway & Sant'Anna event study aggregates ATT(g,t) by event time e = t - g.
This is robust to heterogeneous treatment effects, unlike TWFE event studies.

Visual Elements:
----------------
- Point estimates with 95% confidence intervals
- Reference line at zero
- Vertical line at treatment onset (k = 0)
- Pre-treatment period shading for parallel trends assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS

def main():
    # TODO: Load clean panel data
    # panel = pd.read_csv('data/processed/panel_data.csv')

    # TODO: Create event time variable
    # panel['event_time'] = panel['year'] - panel['treatment_year']
    # For never-treated: set event_time to special value or exclude

    # TODO: Create event time dummies
    # Bin endpoints (e.g., -5 or earlier, +5 or later)
    # Omit k = -1 as reference period

    # TODO: Estimate TWFE event study
    # Include state and year FEs
    # Cluster standard errors at state level

    # TODO: Extract coefficients and confidence intervals

    # TODO: Create TWFE event study plot
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.errorbar(event_times, coefficients, yerr=ci_95, fmt='o', capsize=3)
    # ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    # ax.axvline(x=-0.5, color='red', linestyle='--', alpha=0.5)
    # ax.set_xlabel('Event Time (Years Relative to Treatment)')
    # ax.set_ylabel('Effect on Fatalities per VMT')
    # ax.set_title('Event Study: Effect of Primary Enforcement Laws')
    # plt.savefig('output/figures/event_study_twfe.png', dpi=300, bbox_inches='tight')

    # TODO: Load or compute CS event study estimates

    # TODO: Create comparison plot (TWFE vs CS)
    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # plt.savefig('output/figures/event_study_comparison.png', dpi=300, bbox_inches='tight')

    # TODO: Pre-trends test
    # H0: All pre-treatment coefficients = 0

    # TODO: Export coefficients table

    print("Event study analysis complete. Results saved to output/")

if __name__ == "__main__":
    main()
