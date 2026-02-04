"""
DHS Data Loader
---------------
Utilities for loading and preprocessing Demographic and Health Survey (DHS) data
for maternal health fairness analysis.

DHS data files are typically in Stata (.dta) format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import warnings


# DHS variable mappings (standard recode names)
DHS_VARIABLE_MAPPING = {
    # Demographics
    'v012': 'age',
    'v025': 'residence_type',  # 1=urban, 2=rural
    'v106': 'education_level',  # 0=none, 1=primary, 2=secondary, 3=higher
    'v190': 'wealth_index',  # 1=poorest to 5=richest
    'v024': 'region',
    
    # Reproductive history
    'v201': 'total_children_born',
    'v213': 'currently_pregnant',
    'v228': 'pregnancy_terminated',
    'v238': 'birth_interval_months',
    
    # Healthcare access
    'v467a': 'distance_health_facility_problem',
    'v481': 'health_insurance',
    'm14': 'antenatal_visits',
    'm15': 'place_of_delivery',
    
    # Health indicators
    'v453': 'anemia_level',  # 1=severe, 2=moderate, 3=mild, 4=not anemic
    'v445': 'bmi',
}


def load_dhs_data(
    filepath: str,
    country_code: Optional[str] = None,
    variables: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load DHS data from Stata (.dta) file.
    
    Parameters
    ----------
    filepath : str
        Path to the DHS .dta file
    country_code : str, optional
        Country code to add as column (e.g., 'NG' for Nigeria)
    variables : list, optional
        List of DHS variable names to load. If None, loads common maternal health variables.
    
    Returns
    -------
    pd.DataFrame
        Loaded and minimally processed DHS data
    
    Example
    -------
    >>> df = load_dhs_data('data/raw/NGIR7BFL.DTA', country_code='NG')
    """
    import pyreadstat
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"DHS file not found: {filepath}")
    
    # Default variables for maternal health analysis
    if variables is None:
        variables = list(DHS_VARIABLE_MAPPING.keys())
    
    # Read Stata file
    try:
        df, meta = pyreadstat.read_dta(
            filepath,
            usecols=variables,
            apply_value_formats=False  # Keep numeric codes
        )
    except Exception as e:
        # If specific columns fail, try loading all and filtering
        warnings.warn(f"Could not load specific columns, loading all: {e}")
        df, meta = pyreadstat.read_dta(filepath, apply_value_formats=False)
        available_vars = [v for v in variables if v in df.columns]
        df = df[available_vars]
    
    # Add country identifier
    if country_code:
        df['country'] = country_code
    
    # Rename columns to readable names
    rename_dict = {k: v for k, v in DHS_VARIABLE_MAPPING.items() if k in df.columns}
    df = df.rename(columns=rename_dict)
    
    return df


def load_multiple_countries(
    data_dir: str,
    country_files: Dict[str, str]
) -> pd.DataFrame:
    """
    Load and combine DHS data from multiple countries.
    
    Parameters
    ----------
    data_dir : str
        Base directory containing DHS files
    country_files : dict
        Mapping of country codes to filenames
        e.g., {'NG': 'NGIR7BFL.DTA', 'KE': 'KEIR8AFL.DTA'}
    
    Returns
    -------
    pd.DataFrame
        Combined data from all countries
    
    Example
    -------
    >>> countries = {
    ...     'NG': 'NGIR7BFL.DTA',
    ...     'KE': 'KEIR8AFL.DTA',
    ...     'GH': 'GHIR8AFL.DTA'
    ... }
    >>> df = load_multiple_countries('data/raw/', countries)
    """
    data_dir = Path(data_dir)
    dfs = []
    
    for country_code, filename in country_files.items():
        filepath = data_dir / filename
        print(f"Loading {country_code}: {filename}")
        
        try:
            df = load_dhs_data(filepath, country_code=country_code)
            dfs.append(df)
            print(f"  Loaded {len(df):,} records")
        except Exception as e:
            warnings.warn(f"Failed to load {country_code}: {e}")
    
    if not dfs:
        raise ValueError("No data files could be loaded")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal records: {len(combined):,}")
    
    return combined


def create_high_risk_indicator(df: pd.DataFrame) -> pd.Series:
    """
    Create binary high-risk pregnancy indicator based on WHO criteria.
    
    Risk factors:
    - Maternal age <18 or >35
    - High parity (>4 previous births)
    - Short birth interval (<24 months)
    - Severe/moderate anemia
    - Previous pregnancy loss
    
    Parameters
    ----------
    df : pd.DataFrame
        DHS data with required columns
    
    Returns
    -------
    pd.Series
        Binary indicator (1 = high risk, 0 = not high risk)
    """
    risk_factors = pd.DataFrame(index=df.index)
    
    # Age risk
    if 'age' in df.columns:
        risk_factors['age_risk'] = ((df['age'] < 18) | (df['age'] > 35)).astype(int)
    
    # High parity
    if 'total_children_born' in df.columns:
        risk_factors['parity_risk'] = (df['total_children_born'] > 4).astype(int)
    
    # Short birth interval
    if 'birth_interval_months' in df.columns:
        risk_factors['interval_risk'] = (df['birth_interval_months'] < 24).astype(int)
    
    # Anemia (1=severe, 2=moderate are high risk)
    if 'anemia_level' in df.columns:
        risk_factors['anemia_risk'] = (df['anemia_level'].isin([1, 2])).astype(int)
    
    # Previous pregnancy loss
    if 'pregnancy_terminated' in df.columns:
        risk_factors['loss_risk'] = (df['pregnancy_terminated'] == 1).astype(int)
    
    # High risk if ANY factor present
    high_risk = (risk_factors.sum(axis=1) >= 1).astype(int)
    
    return high_risk


def create_demographic_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create categorical demographic groups for fairness analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        DHS data
    
    Returns
    -------
    pd.DataFrame
        Original data with additional group columns
    """
    df = df.copy()
    
    # Wealth groups (for fairness analysis)
    if 'wealth_index' in df.columns:
        wealth_labels = {1: 'Poorest', 2: 'Poorer', 3: 'Middle', 4: 'Richer', 5: 'Richest'}
        df['wealth_group'] = df['wealth_index'].map(wealth_labels)
        
        # Binary: poor vs non-poor
        df['wealth_binary'] = (df['wealth_index'] <= 2).map({True: 'Poor', False: 'Non-poor'})
    
    # Residence
    if 'residence_type' in df.columns:
        df['residence'] = df['residence_type'].map({1: 'Urban', 2: 'Rural'})
    
    # Education groups
    if 'education_level' in df.columns:
        edu_labels = {0: 'None', 1: 'Primary', 2: 'Secondary', 3: 'Higher'}
        df['education_group'] = df['education_level'].map(edu_labels)
        
        # Binary: educated vs not
        df['education_binary'] = (df['education_level'] >= 2).map({True: 'Secondary+', False: 'Below Secondary'})
    
    return df


def get_sample_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics by country and demographic groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed DHS data
    
    Returns
    -------
    pd.DataFrame
        Summary statistics
    """
    summaries = []
    
    if 'country' in df.columns:
        for country in df['country'].unique():
            country_df = df[df['country'] == country]
            summary = {
                'country': country,
                'n_total': len(country_df),
                'n_urban': (country_df.get('residence_type', pd.Series()) == 1).sum(),
                'n_rural': (country_df.get('residence_type', pd.Series()) == 2).sum(),
                'mean_age': country_df.get('age', pd.Series()).mean(),
                'pct_poorest_quintile': (country_df.get('wealth_index', pd.Series()) == 1).mean() * 100,
            }
            summaries.append(summary)
    
    return pd.DataFrame(summaries)


if __name__ == "__main__":
    # Example usage
    print("DHS Data Loader")
    print("=" * 50)
    print("\nExample usage:")
    print(">>> df = load_dhs_data('data/raw/NGIR7BFL.DTA', country_code='NG')")
    print(">>> df = create_demographic_groups(df)")
    print(">>> df['high_risk'] = create_high_risk_indicator(df)")
