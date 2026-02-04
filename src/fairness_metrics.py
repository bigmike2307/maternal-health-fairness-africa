"""
Fairness Metrics Module
-----------------------
Evaluate machine learning model fairness across demographic groups.

This module provides functions to compute various fairness metrics and 
generate comprehensive fairness reports for maternal health risk prediction models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import warnings


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification metrics for a single group.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_prob : array-like, optional
        Predicted probabilities for positive class
    
    Returns
    -------
    dict
        Dictionary of computed metrics
    """
    # Handle edge cases
    if len(y_true) == 0:
        return {metric: np.nan for metric in 
                ['accuracy', 'precision', 'recall', 'f1', 'tpr', 'fpr', 'ppv', 'npv', 'auc', 'n']}
    
    metrics = {
        'n': len(y_true),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Confusion matrix derived metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Sensitivity)
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    metrics['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    # Positive prediction rate (for demographic parity)
    metrics['positive_rate'] = y_pred.mean()
    
    # Base rate (actual positive rate in group)
    metrics['base_rate'] = y_true.mean()
    
    # AUC if probabilities provided
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = np.nan
    else:
        metrics['auc'] = np.nan
    
    return metrics


def evaluate_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_feature: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    reference_group: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate fairness metrics across groups defined by a sensitive feature.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    sensitive_feature : array-like
        Group membership for each sample
    y_prob : array-like, optional
        Predicted probabilities
    reference_group : str, optional
        Reference group for computing disparities. If None, uses majority group.
    
    Returns
    -------
    dict
        Comprehensive fairness evaluation results
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive_feature = np.array(sensitive_feature)
    
    if y_prob is not None:
        y_prob = np.array(y_prob)
    
    groups = np.unique(sensitive_feature)
    
    # Compute metrics per group
    group_metrics = {}
    for group in groups:
        mask = sensitive_feature == group
        group_metrics[group] = compute_group_metrics(
            y_true[mask],
            y_pred[mask],
            y_prob[mask] if y_prob is not None else None
        )
    
    # Determine reference group (majority if not specified)
    if reference_group is None:
        reference_group = max(groups, key=lambda g: group_metrics[g]['n'])
    
    ref_metrics = group_metrics[reference_group]
    
    # Compute fairness metrics
    fairness_metrics = {}
    
    # 1. Demographic Parity Difference
    # Measures difference in positive prediction rates
    positive_rates = {g: m['positive_rate'] for g, m in group_metrics.items()}
    fairness_metrics['demographic_parity'] = {
        'values': positive_rates,
        'max_difference': max(positive_rates.values()) - min(positive_rates.values()),
        'ratio': min(positive_rates.values()) / max(positive_rates.values()) if max(positive_rates.values()) > 0 else np.nan
    }
    
    # 2. Equalized Odds Difference
    # Measures difference in TPR and FPR
    tprs = {g: m['tpr'] for g, m in group_metrics.items()}
    fprs = {g: m['fpr'] for g, m in group_metrics.items()}
    fairness_metrics['equalized_odds'] = {
        'tpr_values': tprs,
        'fpr_values': fprs,
        'tpr_difference': max(tprs.values()) - min(tprs.values()),
        'fpr_difference': max(fprs.values()) - min(fprs.values()),
    }
    
    # 3. Predictive Parity
    # Equal PPV across groups
    ppvs = {g: m['ppv'] for g, m in group_metrics.items()}
    fairness_metrics['predictive_parity'] = {
        'values': ppvs,
        'max_difference': max(ppvs.values()) - min(ppvs.values()),
    }
    
    # 4. Accuracy Equity
    accuracies = {g: m['accuracy'] for g, m in group_metrics.items()}
    fairness_metrics['accuracy_equity'] = {
        'values': accuracies,
        'max_difference': max(accuracies.values()) - min(accuracies.values()),
    }
    
    # 5. Disparate Impact Ratio
    # Ratio of positive rates (4/5ths rule threshold is 0.8)
    if ref_metrics['positive_rate'] > 0:
        disparate_impact = {
            g: m['positive_rate'] / ref_metrics['positive_rate']
            for g, m in group_metrics.items()
        }
    else:
        disparate_impact = {g: np.nan for g in groups}
    
    fairness_metrics['disparate_impact'] = {
        'values': disparate_impact,
        'reference_group': reference_group,
        'min_ratio': min(disparate_impact.values()),
        'passes_4_5_rule': min(disparate_impact.values()) >= 0.8 if not np.isnan(min(disparate_impact.values())) else False
    }
    
    return {
        'group_metrics': group_metrics,
        'fairness_metrics': fairness_metrics,
        'reference_group': reference_group,
        'groups': list(groups),
        'n_total': len(y_true)
    }


def fairness_report(
    evaluation: Dict[str, Any],
    name: str = "Model"
) -> str:
    """
    Generate a human-readable fairness report.
    
    Parameters
    ----------
    evaluation : dict
        Output from evaluate_fairness()
    name : str
        Name of the model/analysis
    
    Returns
    -------
    str
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"FAIRNESS EVALUATION REPORT: {name}")
    lines.append("=" * 60)
    lines.append(f"\nTotal samples: {evaluation['n_total']:,}")
    lines.append(f"Reference group: {evaluation['reference_group']}")
    lines.append(f"Groups analyzed: {', '.join(map(str, evaluation['groups']))}")
    
    # Group-level metrics
    lines.append("\n" + "-" * 60)
    lines.append("GROUP-LEVEL PERFORMANCE")
    lines.append("-" * 60)
    
    gm = evaluation['group_metrics']
    header = f"{'Group':<15} {'N':>8} {'Acc':>7} {'TPR':>7} {'FPR':>7} {'PPV':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    
    for group, metrics in gm.items():
        lines.append(
            f"{str(group):<15} {metrics['n']:>8,} {metrics['accuracy']:>7.3f} "
            f"{metrics['tpr']:>7.3f} {metrics['fpr']:>7.3f} {metrics['ppv']:>7.3f}"
        )
    
    # Fairness metrics summary
    fm = evaluation['fairness_metrics']
    
    lines.append("\n" + "-" * 60)
    lines.append("FAIRNESS METRICS SUMMARY")
    lines.append("-" * 60)
    
    lines.append(f"\n1. Demographic Parity (equal positive prediction rates)")
    lines.append(f"   Max difference: {fm['demographic_parity']['max_difference']:.4f}")
    lines.append(f"   Min/Max ratio: {fm['demographic_parity']['ratio']:.4f}")
    
    lines.append(f"\n2. Equalized Odds (equal TPR and FPR)")
    lines.append(f"   TPR difference: {fm['equalized_odds']['tpr_difference']:.4f}")
    lines.append(f"   FPR difference: {fm['equalized_odds']['fpr_difference']:.4f}")
    
    lines.append(f"\n3. Predictive Parity (equal PPV)")
    lines.append(f"   Max difference: {fm['predictive_parity']['max_difference']:.4f}")
    
    lines.append(f"\n4. Disparate Impact")
    lines.append(f"   Min ratio: {fm['disparate_impact']['min_ratio']:.4f}")
    lines.append(f"   Passes 4/5ths rule: {fm['disparate_impact']['passes_4_5_rule']}")
    
    lines.append(f"\n5. Accuracy Equity")
    lines.append(f"   Max difference: {fm['accuracy_equity']['max_difference']:.4f}")
    
    # Interpretation
    lines.append("\n" + "-" * 60)
    lines.append("INTERPRETATION")
    lines.append("-" * 60)
    
    issues = []
    if fm['demographic_parity']['max_difference'] > 0.1:
        issues.append("- Significant demographic parity gap (>0.1)")
    if fm['equalized_odds']['tpr_difference'] > 0.1:
        issues.append("- Unequal true positive rates across groups (>0.1)")
    if fm['equalized_odds']['fpr_difference'] > 0.1:
        issues.append("- Unequal false positive rates across groups (>0.1)")
    if not fm['disparate_impact']['passes_4_5_rule']:
        issues.append("- Fails 4/5ths rule for disparate impact")
    
    if issues:
        lines.append("\n⚠️  FAIRNESS CONCERNS IDENTIFIED:")
        for issue in issues:
            lines.append(f"   {issue}")
    else:
        lines.append("\n✓ No major fairness concerns detected (thresholds: 0.1)")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


def compare_fairness_across_countries(
    evaluations: Dict[str, Dict],
    metric: str = 'demographic_parity'
) -> pd.DataFrame:
    """
    Compare fairness metrics across multiple countries/models.
    
    Parameters
    ----------
    evaluations : dict
        Dictionary mapping country/model names to evaluation results
    metric : str
        Fairness metric to compare
    
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    rows = []
    for name, eval_result in evaluations.items():
        fm = eval_result['fairness_metrics']
        if metric in fm:
            row = {'name': name}
            if 'max_difference' in fm[metric]:
                row['max_difference'] = fm[metric]['max_difference']
            if 'ratio' in fm[metric]:
                row['ratio'] = fm[metric]['ratio']
            if 'values' in fm[metric]:
                for group, value in fm[metric]['values'].items():
                    row[f'{metric}_{group}'] = value
            rows.append(row)
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Demo with synthetic data
    print("Fairness Metrics Module Demo")
    print("=" * 50)
    
    np.random.seed(42)
    n = 1000
    
    # Simulate predictions with bias
    y_true = np.random.binomial(1, 0.3, n)
    wealth = np.random.choice(['Poor', 'Middle', 'Rich'], n, p=[0.4, 0.35, 0.25])
    
    # Add bias: model performs worse for 'Poor' group
    y_pred = y_true.copy()
    poor_mask = wealth == 'Poor'
    y_pred[poor_mask] = np.where(
        np.random.random(poor_mask.sum()) < 0.2,  # 20% error rate for poor
        1 - y_true[poor_mask],
        y_true[poor_mask]
    )
    
    y_prob = np.clip(y_pred + np.random.normal(0, 0.1, n), 0, 1)
    
    # Evaluate
    results = evaluate_fairness(y_true, y_pred, wealth, y_prob)
    print(fairness_report(results, "Demo Model"))
