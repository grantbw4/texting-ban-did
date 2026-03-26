# The Effect of Primary Enforcement Handheld Device Bans on Traffic Fatalities

A causal inference portfolio project using staggered difference-in-differences methods to estimate the effect of primary enforcement handheld device bans on traffic fatalities in the United States.

## Key Finding

**Primary enforcement handheld device bans have no statistically significant effect on traffic fatality rates.**

| Estimator | ATT | Std. Error | p-value | 95% CI |
|-----------|-----|------------|---------|--------|
| TWFE | +0.019 | 0.035 | 0.59 | [-0.05, 0.09] |
| Callaway-Sant'Anna | +0.004 | 0.037 | 0.90 | [-0.07, 0.08] |

This null finding is consistent with recent rigorous studies in the literature (Bhargava & Pathania, 2013; Highway Loss Data Institute, 2010) and contributes to the growing evidence that handheld bans, while intuitively appealing, may not meaningfully reduce traffic fatalities.

## Overview

Many U.S. states have enacted laws prohibiting the use of handheld devices while driving, but the enforcement mechanism varies: some states allow "primary enforcement" (officers can stop drivers solely for device use), while others require "secondary enforcement" (officers can only cite device use after stopping for another violation). This project estimates the causal effect of adopting **primary enforcement** handheld device bans on traffic fatalities using modern staggered difference-in-differences techniques.

## Methods

The staggered adoption of primary enforcement laws across states and years creates a natural experiment. However, standard two-way fixed effects (TWFE) estimators can be biased when treatment effects are heterogeneous across time and units. This project implements:

1. **Traditional TWFE** - As a baseline comparison
2. **Goodman-Bacon Decomposition** - To diagnose potential bias in TWFE estimates
3. **Callaway & Sant'Anna (2021)** - A robust estimator for staggered DiD
4. **Event Study Analysis** - To examine treatment effect dynamics and pre-trends

## Repository Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # Original data files
│   └── processed/        # Cleaned analysis-ready data
├── code/
│   ├── 01_clean.py       # Data cleaning and preparation
│   ├── 02_twfe.py        # Two-way fixed effects estimation
│   ├── 03_goodman_bacon.py   # Goodman-Bacon decomposition
│   ├── 04_cs.py          # Callaway & Sant'Anna estimator
│   └── 05_event_study.py # Event study analysis
├── output/
│   ├── figures/          # Generated plots
│   └── tables/           # Regression tables and summaries
└── docs/                 # GitHub Pages site with writeup
```

## Data Sources

- **Traffic Fatalities**: Fatality Analysis Reporting System (FARS), National Highway Traffic Safety Administration
- **State Laws**: Insurance Institute for Highway Safety (IIHS) and state legislative records
- **Controls**: State population, vehicle miles traveled, unemployment rates, etc.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

Run scripts in numerical order:

```bash
python code/01_clean.py
python code/02_twfe.py
python code/03_goodman_bacon.py
python code/04_cs.py
python code/05_event_study.py
```

## Results Summary

### Pre-Trends Test
The parallel trends assumption is supported by the data:
- **Wald statistic**: 7.10 (df=4)
- **p-value**: 0.13
- No significant pre-treatment differences between treated and control states

### Goodman-Bacon Decomposition
The decomposition reveals methodological concerns with TWFE in this setting:
- **61%** of TWFE weight comes from problematic "Later vs Earlier Treated" comparisons
- **190 total** 2x2 comparisons across 10 treatment cohorts
- Despite these concerns, the robust CS estimator confirms the null finding

### Why the Null Result?

Several factors may explain why handheld bans don't reduce fatalities:

1. **Outcome dilution**: Distracted driving causes ~8-10% of fatal crashes; handheld phone use is a subset. A 20% reduction in phone-related crashes would only produce a ~1.6% reduction in total fatalities.

2. **Compliance and enforcement**: Laws don't equal behavior change. Enforcement is difficult, citation rates are low, and drivers may simply hide phone use.

3. **Substitution effects**: Drivers may switch to hands-free devices, which research suggests are equally distracting (cognitive vs. manual distraction).

4. **Underreporting**: Distracted driving is severely underreported in crash data, making it difficult to detect effects.

## Limitations

- **Outcome measure**: Uses all traffic fatalities rather than distraction-specific crashes
- **Always-treated states excluded**: Early adopters (NY, CA, NJ, CT) treated before 2010 serve as controls
- **Compliance unmeasured**: Cannot separate law passage from actual behavior change
- **State heterogeneity**: Effects may exist in some states but be masked by aggregation

## References

- Bhargava, S., & Pathania, V. S. (2013). Driving under the (cellular) influence: The link between cell phone use and vehicle crashes. *American Economic Journal: Economic Policy*, 5(3), 92-125.
- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277.
- Highway Loss Data Institute. (2010). Texting laws and collision claim frequencies. *HLDI Bulletin*, 27(11).

## Author

Grant Williams

## License

MIT
