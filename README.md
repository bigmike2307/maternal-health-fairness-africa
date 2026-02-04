# Maternal Health Fairness Analysis Across African Countries

Investigating algorithmic fairness in maternal health risk prediction models using Demographic and Health Survey (DHS) data from multiple African countries.

## Overview

This project examines how machine learning models for maternal health risk prediction perform across different demographic subgroups in African populations. We evaluate fairness metrics across:

- **Wealth quintiles** (poorest to richest)
- **Urban/rural residence**
- **Educational attainment levels**
- **Geographic regions within countries**

### Research Questions

1. Do maternal health risk prediction models perform equally well across wealth quintiles?
2. Are there systematic disparities in model accuracy between urban and rural populations?
3. How does model performance vary across different African countries?
4. What fairness interventions can reduce identified disparities?

## Countries Analyzed

| Country | DHS Survey Year | Sample Size |
|---------|-----------------|-------------|
| Nigeria | 2018 | ~40,000 women |
| Kenya | 2022 | ~15,000 women |
| Ghana | 2022 | ~9,000 women |
| Uganda | 2016 | ~18,000 women |
| Tanzania | 2022 | ~13,000 women |

## Project Structure

```
maternal-health-fairness-africa/
├── data/                    # DHS datasets (not tracked in git)
│   ├── raw/                 # Original DHS files
│   └── processed/           # Cleaned datasets
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_fairness_evaluation.ipynb
│   └── 05_cross_country_comparison.ipynb
├── src/
│   ├── data_loader.py       # DHS data loading utilities
│   ├── preprocessing.py     # Data cleaning functions
│   ├── features.py          # Feature engineering
│   ├── models.py            # ML model definitions
│   └── fairness_metrics.py  # Fairness evaluation functions
├── results/
│   ├── figures/             # Visualization outputs
│   └── tables/              # Statistical results
├── docs/
│   └── methodology.md       # Detailed methodology
├── requirements.txt
└── README.md
```

## Methodology

### Target Variable

**High-risk pregnancy indicator** constructed from DHS variables:
- Maternal age (<18 or >35)
- High parity (>4 previous births)
- Short birth interval (<24 months)
- Previous pregnancy complications
- Anemia status

### Features

Demographics, socioeconomic indicators, healthcare access, and reproductive history variables from DHS Women's Recode files.

### Fairness Metrics

We evaluate models using multiple fairness definitions:

| Metric | Definition |
|--------|------------|
| **Demographic Parity** | P(Ŷ=1\|A=a) = P(Ŷ=1\|A=b) |
| **Equalized Odds** | Equal TPR and FPR across groups |
| **Predictive Parity** | Equal PPV across groups |
| **Calibration** | P(Y=1\|Ŷ=p, A=a) = p for all groups |

### Models

- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting (XGBoost)
- Fairness-constrained models (Fairlearn)

## Key Findings

*[To be updated after analysis]*

## Installation

```bash
git clone https://github.com/[username]/maternal-health-fairness-africa.git
cd maternal-health-fairness-africa
pip install -r requirements.txt
```

## Data Access

DHS data requires registration at [dhsprogram.com](https://dhsprogram.com/data/). 

1. Create an account at the DHS Program website
2. Submit a data request with your project description
3. Download approved datasets to `data/raw/`

**Note:** DHS data files are not included in this repository due to data use agreements.

## Usage

```python
from src.data_loader import load_dhs_data
from src.fairness_metrics import evaluate_fairness

# Load Nigeria DHS data
df = load_dhs_data('data/raw/NG_2018_DHS.dta')

# Evaluate model fairness across wealth quintiles
fairness_report = evaluate_fairness(
    model=trained_model,
    X_test=X_test,
    y_test=y_test,
    sensitive_feature='wealth_quintile'
)
```

## Results Visualization

![Fairness Comparison](results/figures/fairness_comparison_placeholder.png)
*Placeholder: Comparison of model performance across wealth quintiles*

## Contributing

Contributions welcome! Please read the methodology documentation before submitting changes.

## Citation

If you use this work, please cite:

```bibtex
@misc{maternal_health_fairness_africa,
  author = {[Your Name]},
  title = {Maternal Health Fairness Analysis Across African Countries},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/[username]/maternal-health-fairness-africa}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- DHS Program for providing open access to survey data
- USAID for funding the DHS surveys
- Fairlearn team for fairness evaluation tools

## Contact

Michael Ogungbe || mailmichaelogungbe@gmail.com

---

**Research Context:** This work addresses a critical gap in AI fairness research, which has predominantly focused on Western populations. By examining algorithmic fairness within African healthcare contexts, we aim to ensure that AI-driven health interventions do not exacerbate existing health disparities.
