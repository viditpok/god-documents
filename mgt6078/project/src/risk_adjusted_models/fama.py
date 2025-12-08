import numpy as np
import pandas as pd
from linearmodels.panel import FamaMacBeth
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Directory search

# absolute path to this file
THIS_FILE = Path(__file__).resolve()

# project root = go up two levels: risk_adjusted_models -> src -> project_root
PROJECT_ROOT = THIS_FILE.parents[2]

RISK_INPUTS_DIR = PROJECT_ROOT / "data_processed" / "risk_inputs"

stock_monthly_fama = RISK_INPUTS_DIR / "stock_monthly_data_fama.csv"
stock_monthly_ff = RISK_INPUTS_DIR / "stock_monthly_data_ff.csv"

print("="*120)
print("IMPLEMENTING EXTENDED TABLE 3: Fama-MacBeth Cross-Sectional Regressions (using linearmodels)")
print("Including OLS, TS, LASSO, PLS, and XGB mispricing signals")
print("="*120)

# Set random seed for reproducibility
np.random.seed(42)

# Load existing data
print("\nLoading existing data...")
stock_data = pd.read_csv(stock_monthly_ff)
stock_data['date'] = pd.to_datetime(stock_data['date'])
print(f"✓ Loaded {len(stock_data)} stock-month observations")

stock_data_full =pd.read_csv(stock_monthly_fama)
stock_data_full['date'] = pd.to_datetime(stock_data_full['date'], errors='coerce', infer_datetime_format=True)
print(f"✓ Read stock_monthly_data_with_characteristics.csv")

print("\nCharacteristics summary:")
print(stock_data_full[['mispricing_ols', 'mispricing_ts', 'mispricing_lasso', 'mispricing_pls', 'mispricing_xgb',
                        'beta', 'book_market', 'momentum', 'accruals', 'sue', 'gross_profitability', 
                        'earnings_yield']].describe())

print("\n" + "="*120)
print("Running Fama-MacBeth Regressions using linearmodels...")
print("="*120)

def create_quintile_dummies(data, characteristics):
    """
    Create quintile dummy variables for each characteristic
    Returns data with Q1-Q5 dummies for each characteristic
    """
    data_with_dummies = data.copy()
    
    for char in characteristics:
        # Create quintiles for this characteristic
        data_with_dummies[f'{char}_quintile'] = pd.qcut(
            data_with_dummies[char], 
            q=5, 
            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
            duplicates='drop'
        )
        
        # Create dummy variables (Q1 is omitted as base)
        for q in ['Q2', 'Q3', 'Q4', 'Q5']:
            data_with_dummies[f'{char}_{q}'] = (
                data_with_dummies[f'{char}_quintile'] == q
            ).astype(float)  # Convert to float for regression
    
    return data_with_dummies

def run_fama_macbeth_linearmodels(data, specification, signal_type='ols', industry_control=True):
    """
    Run Fama-MacBeth cross-sectional regressions using linearmodels
    
    Specifications:
    1: Mispricing only (+ industry)
    2: Add beta, size, book-to-market
    3: Add past returns (ST reversal, momentum, LT reversal)
    4: Add accruals, SUE, gross profitability, earnings yield
    5: All controls (OLS signal)
    6: All controls (TS signal)
    7: All controls (LASSO signal)
    8: All controls (PLS signal)
    9: All controls (XGB signal)
    """
    
    # Define characteristics to include in each specification
    if specification == 1:
        characteristics = [f'mispricing_{signal_type}']
    elif specification == 2:
        characteristics = [f'mispricing_{signal_type}', 'beta', 'market_cap', 'book_market']
    elif specification == 3:
        characteristics = [f'mispricing_{signal_type}', 'beta', 'market_cap', 'book_market',
                          'st_reversal', 'momentum', 'lt_reversal']
    elif specification in [4, 5, 6, 7, 8, 9]:
        characteristics = [f'mispricing_{signal_type}', 'beta', 'market_cap', 'book_market',
                          'st_reversal', 'momentum', 'lt_reversal',
                          'accruals', 'sue', 'gross_profitability', 'earnings_yield']
    
    # Prepare data
    reg_data = data.copy()
    
    # Create quintile dummies for all characteristics at once
    reg_data = create_quintile_dummies(reg_data, characteristics)
    
    # Create industry dummies if needed
    if industry_control:
        reg_data = pd.get_dummies(reg_data, columns=['industry'], prefix='industry', drop_first=True)
    
    # Create a time period identifier (required by linearmodels)
    # Convert date to period (e.g., 1, 2, 3, ...)
    reg_data['time_period'] = pd.factorize(reg_data['date'])[0]
    
    # Set up multi-index for panel data (gvkey, time_period)
    # Important: time dimension must be second level of index
    reg_data = reg_data.set_index(['gvkey', 'time_period'])
    
    # Build list of regressors
    regressors = []
    regressor_labels = {}
    
    for char in characteristics:
        for q in ['Q2', 'Q3', 'Q4', 'Q5']:
            var_name = f'{char}_{q}'
            if var_name in reg_data.columns:
                regressors.append(var_name)
                # Create cleaner label for output
                if q == 'Q5':
                    regressor_labels[var_name] = char
    
    # Add industry dummies
    if industry_control:
        industry_cols = [col for col in reg_data.columns if col.startswith('industry_')]
        regressors.extend(industry_cols)
    
    # Prepare dependent and independent variables
    y = reg_data['return']
    X = reg_data[regressors]
    
    # Drop missing values
    valid_idx = y.notna() & X.notna().all(axis=1)
    y = y[valid_idx]
    X = X[valid_idx]
    
    # Run Fama-MacBeth regression
    try:
        mod = FamaMacBeth(y, X)
        res = mod.fit(cov_type='kernel')  # Use kernel covariance for robustness
        
        return res, characteristics, regressor_labels
    except Exception as e:
        print(f"    Error in regression: {e}")
        return None, characteristics, regressor_labels

def extract_fm_results(fm_results, characteristics, regressor_labels):
    """
    Extract results from FamaMacBeth model
    Focus on Q5 coefficients for each characteristic
    """
    if fm_results is None:
        return {}
    
    results = {}
    
    # Extract Q5 coefficients for each characteristic
    for char in characteristics:
        q5_var = f'{char}_Q5'
        
        if q5_var in fm_results.params.index:
            results[char] = {
                'coefficient': fm_results.params[q5_var],
                't_statistic': fm_results.tstats[q5_var],
                'std_error': fm_results.std_errors[q5_var],
                'pvalue': fm_results.pvalues[q5_var]
            }
    
    # Add model statistics
    results['_model_stats'] = {
        'n_obs': fm_results.nobs,
        'rsquared': fm_results.rsquared if hasattr(fm_results, 'rsquared') else None
    }
    
    return results

# Run all specifications
print("\nRunning Fama-MacBeth regressions for all specifications...")

all_results = {}

# Panel A: Full sample
print("\n--- Panel A: Full Sample (1987-2012) ---")
data_full = stock_data_full.copy()

specifications = {
    1: {'signal': 'ols', 'name': 'Spec 1: Mispricing + Industry'},
    2: {'signal': 'ols', 'name': 'Spec 2: Add Beta, Size, B/M'},
    3: {'signal': 'ols', 'name': 'Spec 3: Add Past Returns'},
    4: {'signal': 'ols', 'name': 'Spec 4: Add Earnings Vars'},
    5: {'signal': 'ols', 'name': 'Spec 5: Kitchen Sink (OLS)'},
    6: {'signal': 'ts', 'name': 'Spec 6: Kitchen Sink (TS)'},
    7: {'signal': 'lasso', 'name': 'Spec 7: Kitchen Sink (LASSO)'},
    8: {'signal': 'pls', 'name': 'Spec 8: Kitchen Sink (PLS)'},
    9: {'signal': 'xgb', 'name': 'Spec 9: Kitchen Sink (XGB)'}
}

for spec_num, spec_info in specifications.items():
    if spec_num <5:
        continue
    print(f"\n  Running {spec_info['name']}...")
    
    signal = spec_info['signal']
    
    try:
        fm_results, characteristics, regressor_labels = run_fama_macbeth_linearmodels(
            data_full, 
            specification=spec_num if spec_num <= 5 else 5,  # Specs 6-9 use same vars as 5
            signal_type=signal,
            industry_control=True
        )
        
        stats = extract_fm_results(fm_results, characteristics, regressor_labels)
        
        all_results[f'full_spec{spec_num}'] = {
            'fm_results': fm_results,
            'characteristics': characteristics,
            'stats': stats,
            'signal': signal,
            'regressor_labels': regressor_labels
        }
        
        # Save detailed results
        if fm_results is not None:
            # Save parameter estimates
            params_df = pd.DataFrame({
                'variable': fm_results.params.index,
                'coefficient': fm_results.params.values,
                't_statistic': fm_results.tstats.values,
                'p_value': fm_results.pvalues.values,
                'std_error': fm_results.std_errors.values
            })
            params_df.to_csv(f'data_processed/results/fama/fama_macbeth_results_spec{spec_num}_full.csv', index=False)
            print(f"    ✓ Saved detailed results to fama_macbeth_results_spec{spec_num}_full.csv")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        all_results[f'full_spec{spec_num}'] = {
            'fm_results': None,
            'characteristics': [],
            'stats': {},
            'signal': signal,
            'regressor_labels': {}
        }

# Panel B: Subsample 1993-2012
print("\n--- Panel B: Subsample (1993-2012) ---")
data_subsample = stock_data_full[stock_data_full['date'] >= '1993-01-01'].copy()

for spec_num, spec_info in specifications.items():
    print(f"\n  Running {spec_info['name']}...")
    
    signal = spec_info['signal']
    
    try:
        fm_results, characteristics, regressor_labels = run_fama_macbeth_linearmodels(
            data_subsample, 
            specification=spec_num if spec_num <= 5 else 5,  # Specs 6-9 use same vars as 5
            signal_type=signal,
            industry_control=True
        )
        
        stats = extract_fm_results(fm_results, characteristics, regressor_labels)
        
        all_results[f'subsample_spec{spec_num}'] = {
            'fm_results': fm_results,
            'characteristics': characteristics,
            'stats': stats,
            'signal': signal,
            'regressor_labels': regressor_labels
        }
        
        # Save detailed results
        if fm_results is not None:
            # Save parameter estimates
            params_df = pd.DataFrame({
                'variable': fm_results.params.index,
                'coefficient': fm_results.params.values,
                't_statistic': fm_results.tstats.values,
                'p_value': fm_results.pvalues.values,
                'std_error': fm_results.std_errors.values
            })
          #  params_df.to_csv(f'data_processed/results/fama/fama_macbeth_results_spec{spec_num}_subsample.csv', index=False)
            print(f"    ✓ Saved detailed results to fama_macbeth_results_spec{spec_num}_subsample.csv")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        all_results[f'subsample_spec{spec_num}'] = {
            'fm_results': None,
            'characteristics': [],
            'stats': {},
            'signal': signal,
            'regressor_labels': {}
        }

def save_table3_to_csv(results_dict, panel_name):
    """Save formatted Table 3 to CSV"""
    
    # Characteristic labels for display
    char_labels = {
        'mispricing_ols': 'Mispricing Signal (ols)',
        'mispricing_ts': 'Mispricing Signal (ts)',
        'mispricing_lasso': 'Mispricing Signal (lasso)',
        'mispricing_pls': 'Mispricing Signal (pls)',
        'mispricing_xgb': 'Mispricing Signal (xgb)',
        'beta': 'Beta',
        'market_cap': 'Market capitalization',
        'book_market': 'Book/market',
        'st_reversal': 'Short-term reversal',
        'momentum': 'Momentum',
        'lt_reversal': 'Long-term reversal',
        'accruals': 'Accruals',
        'sue': 'SUE',
        'gross_profitability': 'Gross profitability',
        'earnings_yield': 'Earnings yield'
    }
    
    # Get all unique characteristics across specifications
    all_chars = []
    for spec_num in range(1, 10):
        key = f'{panel_name.lower()}_spec{spec_num}'
        if key in results_dict and results_dict[key]['stats']:
            chars = results_dict[key]['characteristics']
            for char in chars:
                if char not in all_chars:
                    all_chars.append(char)
    
    # Build CSV data
    csv_data = []
    
    # Header row
    header = {'Variable': 'Variable'}
    for spec_num in range(1, 10):
        header[f'Spec_{spec_num}'] = f'Spec {spec_num}'
    csv_data.append(header)
    
    # Add each characteristic with coefficient, t-stat, p-value, and significance rows
    for char in all_chars:
        label = char_labels.get(char, f'{char}')
        
        coef_row = {'Variable': f'{label} (Q5) - Coefficient'}
        tstat_row = {'Variable': f'{label} (Q5) - t-statistic'}
        pval_row = {'Variable': f'{label} (Q5) - p-value'}
        sig_row = {'Variable': f'{label} (Q5) - Significance'}
        
        for spec_num in range(1, 10):
            key = f'{panel_name.lower()}_spec{spec_num}'
            
            if key in results_dict and char in results_dict[key]['stats']:
                stat_dict = results_dict[key]['stats'][char]
                coef = stat_dict['coefficient']
                t_stat = stat_dict['t_statistic']
                p_val = stat_dict['pvalue']
                
                # Significance stars
                sig = '***' if abs(t_stat) > 2.576 else '**' if abs(t_stat) > 1.96 else '*' if abs(t_stat) > 1.645 else ''
                
                coef_row[f'Spec_{spec_num}'] = coef
                tstat_row[f'Spec_{spec_num}'] = t_stat
                pval_row[f'Spec_{spec_num}'] = p_val
                sig_row[f'Spec_{spec_num}'] = sig if sig else 'NS'
            else:
                coef_row[f'Spec_{spec_num}'] = np.nan
                tstat_row[f'Spec_{spec_num}'] = np.nan
                pval_row[f'Spec_{spec_num}'] = np.nan
                sig_row[f'Spec_{spec_num}'] = ''
        
        csv_data.append(coef_row)
        csv_data.append(tstat_row)
        csv_data.append(pval_row)
        csv_data.append(sig_row)
    
    # Add model statistics
    n_obs_row = {'Variable': 'Number of observations'}
    rsq_row = {'Variable': 'R-squared'}
    industry_row = {'Variable': 'Industry control'}
    
    for spec_num in range(1, 10):
        key = f'{panel_name.lower()}_spec{spec_num}'
        if key in results_dict and results_dict[key]['stats'] and '_model_stats' in results_dict[key]['stats']:
            n_obs = results_dict[key]['stats']['_model_stats']['n_obs']
            rsq = results_dict[key]['stats']['_model_stats'].get('rsquared')
            
            n_obs_row[f'Spec_{spec_num}'] = n_obs
            rsq_row[f'Spec_{spec_num}'] = rsq if rsq is not None else np.nan
        else:
            n_obs_row[f'Spec_{spec_num}'] = np.nan
            rsq_row[f'Spec_{spec_num}'] = np.nan
        
        industry_row[f'Spec_{spec_num}'] = 'Yes'
    
    csv_data.append(n_obs_row)
    csv_data.append(rsq_row)
    csv_data.append(industry_row)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(csv_data)
    filename = f'results/tables/fama_macbeth_{panel_name.lower()}_extended.csv'
    df.to_csv(filename, index=False)
    
    return df, filename

def print_table3_results(results_dict, panel_name):
    """Print formatted Table 3 results with coefficients and t-statistics"""
    
    print(f"\n{'='*150}")
    print(f"Panel {panel_name}: Fama-MacBeth Cross-Sectional Regressions (Extended with LASSO, PLS, XGB)")
    print("="*150)
    
    # Characteristic labels for display
    char_labels = {
        'mispricing_ols': 'Mispricing Signal (M)',
        'mispricing_ts': 'Mispricing Signal (M)',
        'mispricing_lasso': 'Mispricing Signal (M)',
        'mispricing_pls': 'Mispricing Signal (M)',
        'mispricing_xgb': 'Mispricing Signal (M)',
        'beta': 'Beta',
        'market_cap': 'Market capitalization',
        'book_market': 'Book/market',
        'st_reversal': 'Short-term reversal',
        'momentum': 'Momentum',
        'lt_reversal': 'Long-term reversal',
        'accruals': 'Accruals',
        'sue': 'SUE',
        'gross_profitability': 'Gross profitability',
        'earnings_yield': 'Earnings yield'
    }
    
    # Get all unique characteristics across specifications
    all_chars = []
    for spec_num in range(1, 10):
        key = f'{panel_name.lower()}_spec{spec_num}'
        if key in results_dict and results_dict[key]['stats']:
            chars = results_dict[key]['characteristics']
            for char in chars:
                if char not in all_chars:
                    all_chars.append(char)
    
    # Print header
    print(f"\n{'Variable':<35} {'Spec 1':>10} {'Spec 2':>10} {'Spec 3':>10} {'Spec 4':>10} {'Spec 5':>10} {'Spec 6':>10} {'Spec 7':>10} {'Spec 8':>10} {'Spec 9':>10}")
    print("-" * 150)
    
    # Print each characteristic with coefficient and t-stat rows
    for char in all_chars:
        label = char_labels.get(char, f'{char}')
        
        # Coefficient row
        coef_line = f"{label + ' (Q5)':<35}"
        tstat_line = f"{'':<35}"
        
        for spec_num in range(1, 10):
            key = f'{panel_name.lower()}_spec{spec_num}'
            
            if key in results_dict and char in results_dict[key]['stats']:
                stat_dict = results_dict[key]['stats'][char]
                coef = stat_dict['coefficient']
                t_stat = stat_dict['t_statistic']
                
                # Significance stars
                sig = '***' if abs(t_stat) > 2.576 else '**' if abs(t_stat) > 1.96 else '*' if abs(t_stat) > 1.645 else ''
                
                coef_line += f" {coef:>7.4f}{sig:<3}"
                tstat_line += f" [{t_stat:>7.2f}]"
            else:
                coef_line += f"          "
                tstat_line += f"          "
        
        print(coef_line)
        print(tstat_line)
    
    # Footer information
    print("-" * 150)
    
    # Number of observations
    print(f"\n{'Number of observations':<35}", end='')
    for spec_num in range(1, 10):
        key = f'{panel_name.lower()}_spec{spec_num}'
        if key in results_dict and results_dict[key]['stats'] and '_model_stats' in results_dict[key]['stats']:
            n_obs = results_dict[key]['stats']['_model_stats']['n_obs']
            print(f" {n_obs:>9}", end='')
        else:
            print(f"          ", end='')
    print()
    
    # R-squared
    print(f"{'Adj. R-squared':<35}", end='')
    for spec_num in range(1, 10):
        key = f'{panel_name.lower()}_spec{spec_num}'
        if key in results_dict and results_dict[key]['stats'] and '_model_stats' in results_dict[key]['stats']:
            rsq = results_dict[key]['stats']['_model_stats'].get('rsquared')
            if rsq is not None:
                print(f" {rsq:>9.3f}", end='')
            else:
                print(f"          ", end='')
        else:
            print(f"          ", end='')
    print()
    
    # Industry control
    print(f"{'Industry control':<35}", end='')
    for spec_num in range(1, 10):
        print(f" {'Yes':>9}", end='')
    print()
    
    print("\nSignal Types:")
    print("  Spec 5: OLS signal")
    print("  Spec 6: TS signal")
    print("  Spec 7: LASSO signal")
    print("  Spec 8: PLS signal")
    print("  Spec 9: XGB signal")
    print("\n*, **, and *** indicate statistical significance at the 10%, 5%, and 1% level, respectively.")
    print("Standard errors are Fama-MacBeth standard errors with kernel covariance.")

# Print and save results for both panels
print_table3_results(all_results, 'Full')
df_full, file_full = save_table3_to_csv(all_results, 'Full')
print(f"\n✓ Saved Panel A results to {file_full}")

print_table3_results(all_results, 'Subsample')
df_subsample, file_subsample = save_table3_to_csv(all_results, 'Subsample')
print(f"\n✓ Saved Panel B results to {file_subsample}")

# Create summary statistics table
print("\n" + "="*150)
print("SUMMARY: MISPRICING SIGNAL (Q5) COEFFICIENTS - ALL SIGNALS")
print("="*150)

summary_data = []

for panel in ['full', 'subsample']:
    panel_label = 'Full Sample' if panel == 'full' else 'Subsample 1993-2012'
    
    for spec_num in range(1, 10):
        key = f'{panel}_spec{spec_num}'
        if key in all_results and all_results[key]['stats']:
            signal = all_results[key]['signal']
            char = f'mispricing_{signal}'
            
            if char in all_results[key]['stats']:
                stat_dict = all_results[key]['stats'][char]
                sig = '***' if abs(stat_dict['t_statistic']) > 2.576 else '**' if abs(stat_dict['t_statistic']) > 1.96 else '*' if abs(stat_dict['t_statistic']) > 1.645 else ''
                
                summary_data.append({
                    'Panel': panel_label,
                    'Specification': spec_num,
                    'Signal': signal.upper(),
                    'Coefficient': f"{stat_dict['coefficient']:.4f}{sig}",
                    't-statistic': f"[{stat_dict['t_statistic']:.2f}]",
                    'p-value': f"{stat_dict['pvalue']:.4f}",
                    'Significant (5%)': 'Yes' if abs(stat_dict['t_statistic']) > 1.96 else 'No'
                })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv('table3_summary_mispricing_coefficients_extended.csv', index=False)
    print("\n✓ Saved table3_summary_mispricing_coefficients_extended.csv")
else:
    print("\nNo results to summarize.")

print("\n" + "="*150)
print("ADDITIONAL FILES GENERATED:")
print("="*150)
print("""
1. fama_macbeth_results_spec[1-9]_full.csv - Full regression results (full sample)
2. fama_macbeth_results_spec[1-9]_subsample.csv - Full regression results (subsample)
3. fama_macbeth_full_extended.csv - Formatted Table 3 Panel A (CSV format)
4. fama_macbeth_subsample_extended.csv - Formatted Table 3 Panel B (CSV format)
5. table3_summary_mispricing_coefficients_extended.csv - Summary of key results

All 5 mispricing signals (OLS, TS, LASSO, PLS, XGB) have been tested.
""")

print("\n" + "="*150)
print("EXTENDED TABLE 3 IMPLEMENTATION COMPLETE (using linearmodels.FamaMacBeth)")
print("="*150)