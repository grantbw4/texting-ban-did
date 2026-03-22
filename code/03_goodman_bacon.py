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
- data/processed/panel_data.csv: Clean panel dataset from 01_clean.py

Outputs:
--------
- output/figures/bacon_decomposition.png: Scatter plot of 2x2 DiD estimates
  vs. weights, colored by comparison type
- output/tables/bacon_weights.csv: Table of comparison types and their weights

Background:
-----------
Goodman-Bacon (2021) shows that the TWFE DiD estimator is a weighted average
of all possible 2x2 DiD comparisons:

    β_TWFE = Σ_k w_k * β_k

Where each β_k is a simple 2x2 DiD estimate comparing:
1. Early-treated vs. never-treated (valid comparison)
2. Late-treated vs. never-treated (valid comparison)
3. Early-treated vs. late-treated (potentially problematic)
4. Late-treated vs. early-treated (potentially problematic - "forbidden")

The "forbidden" comparisons use already-treated units as controls, which
can bias estimates when treatment effects evolve over time.

Key Diagnostics:
----------------
1. What fraction of weight comes from "forbidden" comparisons?
2. Are there negative weights? (can cause sign reversal)
3. Do estimates vary substantially across comparison types?
4. Is the TWFE estimate close to the robust estimators?

Implementation:
---------------
Using the bacon_decomp function or manual computation following the paper.

References:
-----------
Goodman-Bacon, A. (2021). Difference-in-differences with variation in
treatment timing. Journal of Econometrics, 225(2), 254-277.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # TODO: Load clean panel data
    # panel = pd.read_csv('data/processed/panel_data.csv')

    # TODO: Compute Goodman-Bacon decomposition
    # This requires:
    # 1. Identifying all unique treatment cohorts
    # 2. Computing 2x2 DiD estimates for each pair of cohorts
    # 3. Computing weights based on group sizes and timing

    # TODO: Categorize comparisons
    # - "Earlier vs Later Treated"
    # - "Later vs Earlier Treated"
    # - "Treated vs Never Treated"

    # TODO: Create decomposition plot
    # fig, ax = plt.subplots(figsize=(10, 6))
    # scatter = ax.scatter(weights, estimates, c=comparison_type, s=weight_scaled)
    # ax.axhline(y=twfe_estimate, color='red', linestyle='--', label='TWFE Estimate')
    # ax.set_xlabel('Weight')
    # ax.set_ylabel('2x2 DiD Estimate')
    # ax.set_title('Goodman-Bacon Decomposition')
    # plt.savefig('output/figures/bacon_decomposition.png', dpi=300, bbox_inches='tight')

    # TODO: Summarize weights by comparison type

    # TODO: Export decomposition results

    print("Goodman-Bacon decomposition complete. Results saved to output/")

if __name__ == "__main__":
    main()
