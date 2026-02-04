"""
Maternal Health Fairness Analysis - Source Module
"""

from .data_loader import (
    load_dhs_data,
    load_multiple_countries,
    create_high_risk_indicator,
    create_demographic_groups,
    get_sample_summary
)

from .fairness_metrics import (
    compute_group_metrics,
    evaluate_fairness,
    fairness_report,
    compare_fairness_across_countries
)

__version__ = "0.1.0"
