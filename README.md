# The Effect of Primary Enforcement Handheld Device Bans on Traffic Fatalities

A causal inference portfolio project using staggered difference-in-differences methods to estimate the effect of primary enforcement handheld device bans on traffic fatalities in the United States.

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

## References

- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277.

## Author

Grant Williams

## License

MIT
