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
- data/processed/panel_data.csv: Clean panel dataset from 01_clean.py

Outputs:
--------
- output/tables/cs_att_gt.csv: Group-time average treatment effects ATT(g,t)
- output/tables/cs_aggregations.csv: Aggregated treatment effect estimates
- output/figures/cs_event_study.png: Event study plot from CS estimator

Background:
-----------
Callaway & Sant'Anna (2021) address the problems with TWFE by:
1. Only using valid comparisons (never-treated or not-yet-treated as controls)
2. Estimating separate ATT(g,t) for each cohort g at each time t
3. Providing flexible aggregation schemes for summary measures

Key Parameters:
---------------
- g (cohort): The year a unit first received treatment
- t (time): The calendar year of the outcome
- ATT(g,t): Average treatment effect for cohort g at time t

Aggregations:
-------------
1. Simple weighted average: Overall ATT across all (g,t)
2. Dynamic/event-time: ATT(e) where e = t - g (event time)
3. Cohort-specific: ATT(g) averaging over post-treatment periods
4. Calendar time: ATT(t) averaging over treated cohorts

Control Group Options:
----------------------
- "never_treated": Only use never-treated units as controls (conservative)
- "not_yet_treated": Use never-treated + not-yet-treated (more power)

Implementation:
---------------
Using the csdid Python package which implements the estimator.

References:
-----------
Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with
multiple time periods. Journal of Econometrics, 225(2), 200-230.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from csdid import ATTgt  # Main CS estimator

def main():
    # TODO: Load clean panel data
    # panel = pd.read_csv('data/processed/panel_data.csv')

    # TODO: Set up data for csdid
    # Requires: unit id, time period, treatment group (first treatment time), outcome

    # TODO: Estimate ATT(g,t) - group-time average treatment effects
    # att_gt = ATTgt(
    #     data=panel,
    #     yname='fatalities_per_vmt',
    #     tname='year',
    #     idname='state',
    #     gname='treatment_year',  # 0 for never-treated
    #     control_group='not_yet_treated'
    # )
    # results = att_gt.fit()

    # TODO: Compute aggregations
    # - Simple average ATT
    # - Dynamic effects (event study)
    # - Cohort-specific effects

    # TODO: Create summary table

    # TODO: Plot event study from CS estimates
    # fig, ax = plt.subplots(figsize=(10, 6))
    # plt.savefig('output/figures/cs_event_study.png', dpi=300, bbox_inches='tight')

    # TODO: Compare with TWFE estimates

    # TODO: Export results

    print("Callaway & Sant'Anna estimation complete. Results saved to output/")

if __name__ == "__main__":
    main()
