import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_months = 310  # March 1987 to December 2012
n_stocks = 2000  # Average number of stocks per month
n_industries = 38

# absolute path to this file
THIS_FILE = Path(__file__).resolve()

# project root = go up two levels: risk_adjusted_models -> src -> project_root
PROJECT_ROOT = THIS_FILE.parents[2]

RISK_INPUTS_DIR = PROJECT_ROOT / "data_processed" / "risk_inputs"

print("Generating synthetic data for Table 4 analysis...")
ff = RISK_INPUTS_DIR / "factor_returns_ff.csv"
stock_monthly_ff = RISK_INPUTS_DIR / "stock_monthly_data_ff.csv"
industry_returns = RISK_INPUTS_DIR / "industry_returns.csv"
#Change  
factor_returns = pd.read_csv(ff)
print("✓ Saved factor_returns.csv")

stock_data = pd.read_csv(stock_monthly_ff)
print(f"✓ Read stock_monthly_data.csv ({len(stock_data)} observations)")

industry_returns = pd.read_csv(industry_returns)
print("✓ Read industry_returns.csv")

stock_data['date'] = pd.to_datetime(stock_data['date'], dayfirst=True, errors='coerce')
industry_returns['date'] = pd.to_datetime(industry_returns['date'], errors='coerce')
factor_returns['date'] = pd.to_datetime(factor_returns['date'], errors='coerce')

# Merge industry returns back to stock data
stock_data = stock_data.merge(industry_returns, on=['date', 'industry'], how='left')

# Calculate industry-adjusted returns
stock_data['return_ind_adj'] = stock_data['return'] - stock_data['industry_return']

print("\nProcessing portfolios and running regressions...")

def create_quintile_portfolios(data, signal_col, weight_type='equal'):
    """
    Sort stocks into quintiles based on mispricing signal and create portfolios
    """
    portfolios = []
    
    for date in data['date'].unique():
        month_data = data[data['date'] == date].copy()
        
        # Sort into quintiles
        month_data['quintile'] = pd.qcut(month_data[signal_col], 
                                         q=5, 
                                         labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                         duplicates='drop')
        
        # Calculate portfolio returns
        for quintile in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            quintile_data = month_data[month_data['quintile'] == quintile]
            
            if len(quintile_data) > 0:
                if weight_type == 'equal':
                    port_return = quintile_data['return_ind_adj'].mean()
                else:  # value-weighted
                    weights = quintile_data['market_cap'] / quintile_data['market_cap'].sum()
                    port_return = (quintile_data['return_ind_adj'] * weights).sum()
                
                portfolios.append({
                    'date': date,
                    'quintile': quintile,
                    'return': port_return
                })
    
    return pd.DataFrame(portfolios)

def run_factor_regressions(portfolio_returns, factor_returns, n_factors=6):
    """
    Run time series regressions for each quintile
    """
    results = []
    
    # Merge portfolio returns with factors
    merged = portfolio_returns.merge(factor_returns, on='date', how='inner')
    
    # Factor list
    if n_factors == 3:
        factors = ['Mkt_RF', 'SMB', 'HML']
    elif n_factors == 4:
        factors = ['Mkt_RF', 'SMB', 'HML', 'Mom']
    elif n_factors == 5:
        factors = ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']
    elif n_factors == 6:
        factors = ['Mkt_RF', 'SMB', 'HML', 'Mom', 'ST_Rev', 'LT_Rev']
    else:  # 8 factors
        factors = ['Mkt_RF', 'SMB', 'HML', 'Mom', 'ST_Rev', 'LT_Rev', 'CMA', 'RMW']
    
    # Run regression for each quintile
    for quintile in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        quintile_data = merged[merged['quintile'] == quintile].copy()
        
        if len(quintile_data) > 30:  # Need enough observations
            y = quintile_data['return'].values
            X = quintile_data[factors].values
            X = sm.add_constant(X)
            
            model = sm.OLS(y, X)
            fit = model.fit()
            
            result = {
                'quintile': quintile,
                'alpha': fit.params[0],
                't_alpha': fit.tvalues[0],
            }
            
            for i, factor in enumerate(factors):
                result[f'beta_{factor}'] = fit.params[i+1]
                result[f't_{factor}'] = fit.tvalues[i+1]
            
            result['r_squared'] = fit.rsquared
            result['n_obs'] = len(quintile_data)
            results.append(result)
    
    return pd.DataFrame(results)

def calculate_spread_statistics(portfolio_returns, factor_returns, n_factors=6):
    """
    Calculate Q5-Q1 spread and run regression
    """
    # Merge and pivot
    merged = portfolio_returns.merge(factor_returns, on='date', how='inner')
    pivoted = merged.pivot(index='date', columns='quintile', values='return')
    
    # Calculate spread
    pivoted['spread'] = pivoted['Q5'] - pivoted['Q1']
    
    # Merge with factors
    spread_data = pivoted[['spread']].merge(factor_returns, on='date', how='inner')
    
    # Factor list
    if n_factors == 3:
        factors = ['Mkt_RF', 'SMB', 'HML']
    elif n_factors == 4:
        factors = ['Mkt_RF', 'SMB', 'HML', 'Mom']
    elif n_factors == 5:
        factors = ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']
    elif n_factors == 6:
        factors = ['Mkt_RF', 'SMB', 'HML', 'Mom', 'ST_Rev', 'LT_Rev']
    else:  # 8 factors
        factors = ['Mkt_RF', 'SMB', 'HML', 'Mom', 'ST_Rev', 'LT_Rev', 'CMA', 'RMW']
    
    # Run regression
    y = spread_data['spread'].values
    X = spread_data[factors].values
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X)
    fit = model.fit()
    
    # Calculate fraction positive
    frac_positive = (spread_data['spread'] > 0).mean()
    
    result = {
        'quintile': 'Q5-Q1',
        'mean_spread': spread_data['spread'].mean(),
        't_spread': spread_data['spread'].mean() / (spread_data['spread'].std() / np.sqrt(len(spread_data))),
        'frac_positive': frac_positive,
        'alpha': fit.params[0],
        't_alpha': fit.tvalues[0],
    }
    
    for i, factor in enumerate(factors):
        result[f'beta_{factor}'] = fit.params[i+1]
        result[f't_{factor}'] = fit.tvalues[i+1]
    
    result['r_squared'] = fit.rsquared
    result['n_obs'] = len(spread_data)
    
    return result

# Process for ALL FIVE mispricing signals
all_results = {}

for signal_type in ['ols', 'ts', 'lasso', 'pls', 'xgb']:
    signal_col = f'mispricing_{signal_type}'
    
    print(f"\n{'='*60}")
    print(f"Processing {signal_type.upper()} mispricing signal")
    print(f"{'='*60}")
    
    for weight_type in ['equal', 'value']:
        print(f"\n  {weight_type.capitalize()}-weighted portfolios:")
        
        # Create portfolios
        portfolios = create_quintile_portfolios(stock_data, signal_col, weight_type)
        
        # Save portfolio returns
        filename = f'data_processed/results/portfolio_returns_{signal_type}_{weight_type}.csv'
        portfolios.to_csv(filename, index=False)
        print(f"    ✓ Saved {filename}")
        
        # Calculate average returns by quintile
        avg_returns = portfolios.groupby('quintile')['return'].mean()
        
        # Run regressions for all factor models
        results_3f = run_factor_regressions(portfolios, factor_returns, n_factors=3)
        results_4f = run_factor_regressions(portfolios, factor_returns, n_factors=4)
        results_5f = run_factor_regressions(portfolios, factor_returns, n_factors=5)
        results_6f = run_factor_regressions(portfolios, factor_returns, n_factors=6)
        results_8f = run_factor_regressions(portfolios, factor_returns, n_factors=8)
        
        # Calculate spread statistics
        spread_3f = calculate_spread_statistics(portfolios, factor_returns, n_factors=3)
        spread_4f = calculate_spread_statistics(portfolios, factor_returns, n_factors=4)
        spread_5f = calculate_spread_statistics(portfolios, factor_returns, n_factors=5)
        spread_6f = calculate_spread_statistics(portfolios, factor_returns, n_factors=6)
        spread_8f = calculate_spread_statistics(portfolios, factor_returns, n_factors=8)
        
        # Store results
        key = f'{signal_type}_{weight_type}'
        all_results[key] = {
            'avg_returns': avg_returns,
            'results_3f': results_3f,
            'results_4f': results_4f,
            'results_5f': results_5f,
            'results_6f': results_6f,
            'results_8f': results_8f,
            'spread_3f': spread_3f,
            'spread_4f': spread_4f,
            'spread_5f': spread_5f,
            'spread_6f': spread_6f,
            'spread_8f': spread_8f
        }
        
        print(f"    ✓ Completed regressions for all factor models")

# ============================================================================
# FORMATTED TABLE OUTPUT (Academic Paper Style)
# ============================================================================

def format_sig(val, t_stat):
    """Add significance stars based on t-statistic"""
    sig = '***' if abs(t_stat) > 2.576 else '**' if abs(t_stat) > 1.96 else '*' if abs(t_stat) > 1.645 else ''
    return sig

def print_academic_table(signals, weight_type, all_results, factor_models_to_show=['6f', '8f']):
    """
    Print results in academic paper table format
    signals: list of signal types to include (e.g., ['ols', 'ts'] or ['lasso', 'pls', 'xgb'])
    """
    
    panel_name = 'Equal-weighted' if weight_type == 'equal' else 'Value-weighted'
    
    print("\n" + "="*200)
    print(f"Panel: {panel_name} portfolios (with industry control)")
    print("="*200)
    
    # Calculate column width
    col_width = 20
    
    # Print header with signal names
    header_line = " " * 40  # Space for row labels
    for signal in signals:
        signal_label = signal.upper().center(col_width * 6)
        header_line += signal_label + "  "
    print("\n" + header_line)
    
    # Print sub-header with quintile labels
    subheader = " " * 40
    for signal in signals:
        subheader += "Q1 (overvalued)".center(col_width)
        subheader += "Q2".center(col_width)
        subheader += "Q3".center(col_width)
        subheader += "Q4".center(col_width)
        subheader += "Q5 (undervalued)".center(col_width)
        subheader += "Q5-Q1 (spread)".center(col_width)
        subheader += "  "
    print(subheader)
    
    # Print column headers
    colheader = " " * 40
    for signal in signals:
        for _ in range(6):  # 5 quintiles + 1 spread
            colheader += "Coef.".rjust(9) + "[t-stat]".rjust(11)
        colheader += "  "
    print(colheader)
    print("-" * 200)
    
    # Industry-adjusted returns
    print(f"{'Industry-adjusted returns':<40}", end='')
    for signal in signals:
        key = f'{signal}_{weight_type}'
        avg_returns = all_results[key]['avg_returns']
        
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            ret = avg_returns[q]
            t_stat = ret / (avg_returns.std() / np.sqrt(n_months))
            sig = format_sig(ret, t_stat)
            print(f"{ret:9.4f}{sig:<3}{t_stat:>8.2f}]".rjust(17), end='')
        
        # Spread
        spread = all_results[key]['spread_3f']['mean_spread']
        t_spread = all_results[key]['spread_3f']['t_spread']
        sig = format_sig(spread, t_spread)
        print(f"{spread:9.4f}{sig:<3}{t_spread:>8.2f}]".rjust(17), end='  ')
    print()
    
    # Print each factor model
    for model_key in factor_models_to_show:
        n_factors = int(model_key[0])
        
        if n_factors == 3:
            model_name = "Three-factor model"
            factors = ['Mkt_RF', 'SMB', 'HML']
        elif n_factors == 4:
            model_name = "Four-factor model"
            factors = ['Mkt_RF', 'SMB', 'HML', 'Mom']
        elif n_factors == 5:
            model_name = "Five-factor model"
            factors = ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']
        elif n_factors == 6:
            model_name = "Six-factor model"
            factors = ['Mkt_RF', 'SMB', 'HML', 'Mom', 'ST_Rev', 'LT_Rev']
        else:  # 8
            model_name = "Eight-factor model"
            factors = ['Mkt_RF', 'SMB', 'HML', 'Mom', 'ST_Rev', 'LT_Rev', 'CMA', 'RMW']
        
        print(f"\n{model_name}")
        
        # Alpha row
        print(f"{'Alpha':<40}", end='')
        for signal in signals:
            key = f'{signal}_{weight_type}'
            results = all_results[key][f'results_{model_key}']
            spread = all_results[key][f'spread_{model_key}']
            
            for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                q_data = results[results['quintile'] == q].iloc[0]
                alpha = q_data['alpha']
                t_alpha = q_data['t_alpha']
                sig = format_sig(alpha, t_alpha)
                print(f"{alpha:9.4f}{sig:<3}{t_alpha:>8.2f}]".rjust(17), end='')
            
            # Spread alpha
            alpha = spread['alpha']
            t_alpha = spread['t_alpha']
            sig = format_sig(alpha, t_alpha)
            print(f"{alpha:9.4f}{sig:<3}{t_alpha:>8.2f}]".rjust(17), end='  ')
        print()
        
        # Factor rows
        for factor in factors:
            print(f"{factor:<40}", end='')
            for signal in signals:
                key = f'{signal}_{weight_type}'
                results = all_results[key][f'results_{model_key}']
                spread = all_results[key][f'spread_{model_key}']
                
                for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                    q_data = results[results['quintile'] == q].iloc[0]
                    beta = q_data[f'beta_{factor}']
                    t_beta = q_data[f't_{factor}']
                    sig = format_sig(beta, t_beta)
                    print(f"{beta:9.4f}{sig:<3}{t_beta:>8.2f}]".rjust(17), end='')
                
                # Spread factor loading
                beta = spread[f'beta_{factor}']
                t_beta = spread[f't_{factor}']
                sig = format_sig(beta, t_beta)
                print(f"{beta:9.4f}{sig:<3}{t_beta:>8.2f}]".rjust(17), end='  ')
            print()
        
        # R-squared row
        print(f"{'R-squared':<40}", end='')
        for signal in signals:
            key = f'{signal}_{weight_type}'
            results = all_results[key][f'results_{model_key}']
            spread = all_results[key][f'spread_{model_key}']
            
            for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                q_data = results[results['quintile'] == q].iloc[0]
                r2 = q_data['r_squared']
                print(f"{r2:20.2f}", end='')
            
            r2 = spread['r_squared']
            print(f"{r2:20.2f}  ", end='')
        print()
        
        # Number of observations row
        print(f"{'Number of observations':<40}", end='')
        for signal in signals:
            key = f'{signal}_{weight_type}'
            results = all_results[key][f'results_{model_key}']
            spread = all_results[key][f'spread_{model_key}']
            
            for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                q_data = results[results['quintile'] == q].iloc[0]
                n_obs = int(q_data['n_obs'])
                print(f"{n_obs:20d}", end='')
            
            n_obs = int(spread['n_obs'])
            print(f"{n_obs:20d}  ", end='')
        print()

# Print tables in the academic format
print("\n" + "="*200)
print("TABLE 4: Time-Series Regressions of Quintile Portfolios Formed on Mispricing Signals")
print("="*200)

# Panel A: Equal-weighted - OLS and TS
print("\n" + "="*200)
print("Panel A: OLS and TS Signals")
print("="*200)
print_academic_table(['ols', 'ts'], 'equal', all_results, ['6f', '8f'])

# Panel B: Equal-weighted - LASSO, PLS, XGB
print("\n" + "="*200)
print("Panel B: LASSO, PLS, and XGB Signals")
print("="*200)
print_academic_table(['lasso', 'pls', 'xgb'], 'equal', all_results, ['6f', '8f'])

# Panel C: Value-weighted - OLS and TS
print("\n" + "="*200)
print("Panel C: OLS and TS Signals (Value-weighted)")
print("="*200)
print_academic_table(['ols', 'ts'], 'value', all_results, ['6f', '8f'])

# Panel D: Value-weighted - LASSO, PLS, XGB
print("\n" + "="*200)
print("Panel D: LASSO, PLS, and XGB Signals (Value-weighted)")
print("="*200)
print_academic_table(['lasso', 'pls', 'xgb'], 'value', all_results, ['6f', '8f'])

# ============================================================================
# SAVE TO CSV IN TABLE FORMAT
# ============================================================================

print("\n" + "="*200)
print("Saving formatted table to CSV...")
print("="*200)

def create_formatted_csv_table(signals, weight_type, all_results, factor_models):
    """Create a CSV in the same format as the printed table"""
    rows = []
    
    panel_name = 'Equal-weighted' if weight_type == 'equal' else 'Value-weighted'
    
    # Header row with signal names
    header1 = ['', '']
    for signal in signals:
        header1.extend([signal.upper()] * 12)  # 6 quintiles × 2 columns each
    
    # Sub-header row with quintile labels
    header2 = ['', '']
    for signal in signals:
        for label in ['Q1 (overvalued)', 'Q2', 'Q3', 'Q4', 'Q5 (undervalued)', 'Q5-Q1 (spread)']:
            header2.extend([label, ''])
    
    # Column header row
    header3 = ['Panel', 'Variable']
    for signal in signals:
        for _ in range(6):
            header3.extend(['Coefficient', 't-statistic'])
    
    rows.append(header1)
    rows.append(header2)
    rows.append(header3)
    
    # Industry-adjusted returns
    row = [panel_name, 'Industry-adjusted returns']
    for signal in signals:
        key = f'{signal}_{weight_type}'
        avg_returns = all_results[key]['avg_returns']
        
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            ret = avg_returns[q]
            t_stat = ret / (avg_returns.std() / np.sqrt(n_months))
            sig = format_sig(ret, t_stat)
            row.extend([f"{ret:.4f}{sig}", f"[{t_stat:.2f}]"])
        
        spread = all_results[key]['spread_3f']['mean_spread']
        t_spread = all_results[key]['spread_3f']['t_spread']
        sig = format_sig(spread, t_spread)
        row.extend([f"{spread:.4f}{sig}", f"[{t_spread:.2f}]"])
    rows.append(row)
    
    # Each factor model
    for model_key in factor_models:
        n_factors = int(model_key[0])
        
        if n_factors == 3:
            model_name = "Three-factor model"
            factors = ['Mkt_RF', 'SMB', 'HML']
        elif n_factors == 4:
            model_name = "Four-factor model"
            factors = ['Mkt_RF', 'SMB', 'HML', 'Mom']
        elif n_factors == 5:
            model_name = "Five-factor model"
            factors = ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']
        elif n_factors == 6:
            model_name = "Six-factor model"
            factors = ['Mkt_RF', 'SMB', 'HML', 'Mom', 'ST_Rev', 'LT_Rev']
        else:
            model_name = "Eight-factor model"
            factors = ['Mkt_RF', 'SMB', 'HML', 'Mom', 'ST_Rev', 'LT_Rev', 'CMA', 'RMW']
        
        # Model name row
        rows.append(['', model_name] + [''] * (len(signals) * 12))
        
        # Alpha row
        row = ['', 'Alpha']
        for signal in signals:
            key = f'{signal}_{weight_type}'
            results = all_results[key][f'results_{model_key}']
            spread = all_results[key][f'spread_{model_key}']
            
            for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                q_data = results[results['quintile'] == q].iloc[0]
                alpha = q_data['alpha']
                t_alpha = q_data['t_alpha']
                sig = format_sig(alpha, t_alpha)
                row.extend([f"{alpha:.4f}{sig}", f"[{t_alpha:.2f}]"])
            
            alpha = spread['alpha']
            t_alpha = spread['t_alpha']
            sig = format_sig(alpha, t_alpha)
            row.extend([f"{alpha:.4f}{sig}", f"[{t_alpha:.2f}]"])
        rows.append(row)
        
        # Factor rows
        for factor in factors:
            row = ['', factor]
            for signal in signals:
                key = f'{signal}_{weight_type}'
                results = all_results[key][f'results_{model_key}']
                spread = all_results[key][f'spread_{model_key}']
                
                for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                    q_data = results[results['quintile'] == q].iloc[0]
                    beta = q_data[f'beta_{factor}']
                    t_beta = q_data[f't_{factor}']
                    sig = format_sig(beta, t_beta)
                    row.extend([f"{beta:.4f}{sig}", f"[{t_beta:.2f}]"])
                
                beta = spread[f'beta_{factor}']
                t_beta = spread[f't_{factor}']
                sig = format_sig(beta, t_beta)
                row.extend([f"{beta:.4f}{sig}", f"[{t_beta:.2f}]"])
            rows.append(row)
        
        # R-squared row
        row = ['', 'R-squared']
        for signal in signals:
            key = f'{signal}_{weight_type}'
            results = all_results[key][f'results_{model_key}']
            spread = all_results[key][f'spread_{model_key}']
            
            for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                q_data = results[results['quintile'] == q].iloc[0]
                r2 = q_data['r_squared']
                row.extend([f"{r2:.2f}", ''])
            
            r2 = spread['r_squared']
            row.extend([f"{r2:.2f}", ''])
        rows.append(row)
        
        # N obs row
        row = ['', 'Number of observations']
        for signal in signals:
            key = f'{signal}_{weight_type}'
            results = all_results[key][f'results_{model_key}']
            spread = all_results[key][f'spread_{model_key}']
            
            for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                q_data = results[results['quintile'] == q].iloc[0]
                n_obs = int(q_data['n_obs'])
                row.extend([str(n_obs), ''])
            
            n_obs = int(spread['n_obs'])
            row.extend([str(n_obs), ''])
        rows.append(row)
    
    return pd.DataFrame(rows)

# Create formatted tables for CSV export
table_ols_ts_equal = create_formatted_csv_table(['ols', 'ts'], 'equal', all_results, ['6f', '8f'])
table_lasso_pls_xgb_equal = create_formatted_csv_table(['lasso', 'pls', 'xgb'], 'equal', all_results, ['6f', '8f'])
table_ols_ts_value = create_formatted_csv_table(['ols', 'ts'], 'value', all_results, ['6f', '8f'])
table_lasso_pls_xgb_value = create_formatted_csv_table(['lasso', 'pls', 'xgb'], 'value', all_results, ['6f', '8f'])

# Save to CSV
table_ols_ts_equal.to_csv('data_processed/results/formatted_table_OLS_TS_equal_weighted.csv', index=False, header=False)
table_lasso_pls_xgb_equal.to_csv('data_processed/results/formatted_table_LASSO_PLS_XGB_equal_weighted.csv', index=False, header=False)
table_ols_ts_value.to_csv('data_processed/results/formatted_table_OLS_TS_value_weighted.csv', index=False, header=False)
table_lasso_pls_xgb_value.to_csv('data_processed/results/formatted_table_LASSO_PLS_XGB_value_weighted.csv', index=False, header=False)

print("✓ Saved formatted_table_OLS_TS_equal_weighted.csv")
print("✓ Saved formatted_table_LASSO_PLS_XGB_equal_weighted.csv")
print("✓ Saved formatted_table_OLS_TS_value_weighted.csv")
print("✓ Saved formatted_table_LASSO_PLS_XGB_value_weighted.csv")

# Also create comprehensive consolidated CSV
consolidated_rows = []

for signal_type in ['ols', 'ts', 'lasso', 'pls', 'xgb']:
    for weight_type in ['equal', 'value']:
        key = f'{signal_type}_{weight_type}'
        
        for n_factors in [3, 4, 5, 6, 8]:
            model_key = f'{n_factors}f'
            results_key = f'results_{model_key}'
            spread_key = f'spread_{model_key}'
            
            # Get factor list for this model
            if n_factors == 3:
                factors = ['Mkt_RF', 'SMB', 'HML']
            elif n_factors == 4:
                factors = ['Mkt_RF', 'SMB', 'HML', 'Mom']
            elif n_factors == 5:
                factors = ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']
            elif n_factors == 6:
                factors = ['Mkt_RF', 'SMB', 'HML', 'Mom', 'ST_Rev', 'LT_Rev']
            else:
                factors = ['Mkt_RF', 'SMB', 'HML', 'Mom', 'ST_Rev', 'LT_Rev', 'CMA', 'RMW']
            
            # Add quintile results
            for _, row in all_results[key][results_key].iterrows():
                row_data = {
                    'Signal': signal_type.upper(),
                    'Weighting': weight_type.capitalize(),
                    'N_Factors': n_factors,
                    'Quintile': row['quintile'],
                    'Alpha': row['alpha'],
                    't_Alpha': row['t_alpha'],
                    'R_squared': row['r_squared'],
                    'N_obs': row['n_obs']
                }
                
                for factor in factors:
                    row_data[f'Beta_{factor}'] = row[f'beta_{factor}']
                    row_data[f't_{factor}'] = row[f't_{factor}']
                
                consolidated_rows.append(row_data)
            
            # Add spread results
            spread = all_results[key][spread_key]
            spread_row = {
                'Signal': signal_type.upper(),
                'Weighting': weight_type.capitalize(),
                'N_Factors': n_factors,
                'Quintile': 'Q5-Q1',
                'Alpha': spread['alpha'],
                't_Alpha': spread['t_alpha'],
                'R_squared': spread['r_squared'],
                'N_obs': spread['n_obs']
            }
            
            for factor in factors:
                spread_row[f'Beta_{factor}'] = spread[f'beta_{factor}']
                spread_row[f't_{factor}'] = spread[f't_{factor}']
            
            spread_row['Mean_Spread'] = spread['mean_spread']
            spread_row['t_Spread'] = spread['t_spread']
            spread_row['Frac_Positive'] = spread['frac_positive']
            
            consolidated_rows.append(spread_row)

consolidated_df = pd.DataFrame(consolidated_rows)
base_cols = ['Signal', 'Weighting', 'N_Factors', 'Quintile', 'Alpha', 't_Alpha', 'R_squared', 'N_obs']
factor_cols = [col for col in consolidated_df.columns if col.startswith('Beta_') or col.startswith('t_')]
spread_cols = ['Mean_Spread', 't_Spread', 'Frac_Positive']
ordered_cols = base_cols + sorted(factor_cols) + [col for col in spread_cols if col in consolidated_df.columns]
consolidated_df = consolidated_df[ordered_cols]

# consolidated_df.to_csv('comprehensive_results_all_signals_all_factors.csv', index=False)
# print("✓ Saved comprehensive_results_all_signals_all_factors.csv")

print("\n" + "="*200)
print("SUMMARY OF OUTPUT FILES:")
print("="*200)
print("""
FORMATTED ACADEMIC-STYLE TABLES (CSV):
1. formatted_table_OLS_TS_equal_weighted.csv
2. formatted_table_LASSO_PLS_XGB_equal_weighted.csv
3. formatted_table_OLS_TS_value_weighted.csv
4. formatted_table_LASSO_PLS_XGB_value_weighted.csv

COMPREHENSIVE RESULTS (all signals, all factors):
5. comprehensive_results_all_signals_all_factors.csv

INDIVIDUAL PORTFOLIO RETURNS:
- portfolio_returns_[signal]_[weighting].csv

All regressions completed and formatted tables saved!
""")

print("\n" + "="*200)
print("Analysis complete!")
print("="*200)