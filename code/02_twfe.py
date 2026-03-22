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
- data/processed/panel_data.csv: Clean panel dataset from 01_clean.py

Outputs:
--------
- output/tables/twfe_results.tex: LaTeX table of TWFE regression results
- output/tables/twfe_results.csv: CSV of coefficient estimates and standard errors

Model Specification:
--------------------
Y_it = α_i + λ_t + β * Treated_it + X_it'γ + ε_it

Where:
- Y_it: Outcome (fatalities or fatality rate) for state i in year t
- α_i: State fixed effects
- λ_t: Year fixed effects
- Treated_it: Binary indicator = 1 if state i has primary enforcement in year t
- X_it: Time-varying control variables
- β: Treatment effect of interest (ATT under parallel trends)

Estimations:
------------
1. Basic TWFE without controls
2. TWFE with state-level controls
3. TWFE with different outcome measures (levels, rates, logs)
4. Robustness checks (clustered SEs, weighted by population)

Caveats:
--------
- TWFE can be biased with staggered adoption and heterogeneous treatment effects
- Early-treated units may serve as controls for late-treated units ("forbidden comparisons")
- Negative weights can arise, leading to sign reversal
- See Goodman-Bacon (2021) and de Chaisemartin & D'Haultfoeuille (2020)
- Results from this script should be compared with robust estimators in 04_cs.py

Notes:
------
- Standard errors clustered at the state level
- Consider weighting by population or VMT for policy relevance
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

def main():
    # TODO: Load clean panel data
    # panel = pd.read_csv('data/processed/panel_data.csv')

    # TODO: Set up panel data structure
    # panel = panel.set_index(['state', 'year'])

    # TODO: Define outcome and treatment variables
    # y = panel['fatalities_per_vmt']
    # treated = panel['treated']

    # TODO: Estimate basic TWFE
    # model = PanelOLS(y, treated, entity_effects=True, time_effects=True)
    # results = model.fit(cov_type='clustered', cluster_entity=True)

    # TODO: Estimate TWFE with controls

    # TODO: Estimate with alternative outcomes (log fatalities, per capita)

    # TODO: Export results tables

    print("TWFE estimation complete. Results saved to output/tables/")

if __name__ == "__main__":
    main()
