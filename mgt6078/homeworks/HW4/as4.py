import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import f
import statsmodels.api as sm


DATA_DIR = Path('data')
RESULTS_DIR = Path('results')
START_DATE = pd.Timestamp('1963-07-31')


def log_header(text: str) -> None:
    print(f'\n### {text}')


def preview_dataset(name: str, df: pd.DataFrame) -> None:
    print(f'\n{name} columns: {list(df.columns)}')
    print(df.head(10))


def load_ff5(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, skiprows=4)
    raw = raw.rename(columns={raw.columns[0]: 'date'})
    raw['date'] = raw['date'].astype(str).str.strip()
    raw = raw[raw['date'].str.fullmatch(r'\d{6}')]
    num_cols = raw.columns.drop('date')
    raw[num_cols] = raw[num_cols].replace('.', np.nan)
    raw[num_cols] = raw[num_cols].apply(
        lambda c: pd.to_numeric(c, errors='coerce'))
    raw['date'] = pd.to_datetime(
        raw['date'], format='%Y%m').dt.to_period('M').dt.to_timestamp('M')
    raw = raw.set_index('date').dropna(how='all')
    raw = raw.apply(lambda col: col / 100.0)
    raw = raw.rename(columns=lambda c: c.strip().replace('-', '_'))
    return raw


def load_momentum(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, skiprows=13)
    raw = raw.rename(columns={raw.columns[0]: 'date'})
    raw['date'] = raw['date'].astype(str).str.strip()
    raw = raw[raw['date'].str.fullmatch(r'\d{6}')]
    raw['date'] = pd.to_datetime(
        raw['date'], format='%Y%m').dt.to_period('M').dt.to_timestamp('M')
    raw = raw.set_index('date')
    raw = raw.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    raw = raw / 100.0
    raw = raw.rename(columns={'Mom': 'MOM'})
    return raw


def load_portfolios(path: Path, skiprows: int) -> pd.DataFrame:
    raw = pd.read_csv(path, skiprows=skiprows)
    raw = raw.rename(columns={raw.columns[0]: 'date'})
    raw['date'] = raw['date'].astype(str).str.strip()
    mask = raw['date'].str.fullmatch(r'\d{6}')
    raw = raw[mask]
    raw['date'] = pd.to_datetime(
        raw['date'], format='%Y%m').dt.to_period('M').dt.to_timestamp('M')
    raw = raw.set_index('date')
    raw = raw.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    raw = raw / 100.0
    cleaned_columns = {c: c.strip().replace(' ', '_') for c in raw.columns}
    raw = raw.rename(columns=cleaned_columns)
    return raw


def load_hml_dev(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.xlsx':
        sheet = pd.read_excel(path, sheet_name='HML Devil', header=17)
        sheet = sheet.rename(
            columns={'Unnamed: 0': 'date', 'Unnamed: 24': 'HML_DEV'})
    else:
        sheet = pd.read_csv(path, skiprows=18)
        sheet = sheet.rename(columns={'DATE': 'date', 'USA': 'HML_DEV'})
    sheet['date'] = pd.to_datetime(sheet['date'], errors='coerce')
    sheet = sheet[['date', 'HML_DEV']]
    sheet['HML_DEV'] = (
        sheet['HML_DEV']
        .astype(str)
        .str.replace('%', '', regex=False)
        .str.replace(',', '', regex=False)
    )
    sheet['HML_DEV'] = pd.to_numeric(
        sheet['HML_DEV'], errors='coerce') / (1.0 if path.suffix.lower() == '.xlsx' else 100.0)
    sheet = sheet.dropna()
    sheet['date'] = sheet['date'].dt.to_period('M').dt.to_timestamp('M')
    sheet = sheet.set_index('date').sort_index()
    return sheet


def align_dataframes(dfs: dict) -> tuple[dict, pd.DatetimeIndex]:
    aligned_index = None
    for df in dfs.values():
        aligned_index = df.index if aligned_index is None else aligned_index.intersection(
            df.index)
    aligned_index = aligned_index[aligned_index >= START_DATE]
    aligned_dict = {name: df.loc[aligned_index].copy()
                    for name, df in dfs.items()}
    full_range = pd.period_range(
        aligned_index.min(), aligned_index.max(), freq='M').to_timestamp('M')
    if len(full_range.difference(aligned_index)) > 0:
        missing = full_range.difference(aligned_index)
        print(
            f'warning: missing dates detected -> {list(missing)[:5]} ... total {len(missing)}')
    print(f'aligned sample from {aligned_index.min().date()} to {aligned_index.max().date()} '
          f'({len(aligned_index)} months)')
    return aligned_dict, aligned_index


def build_factor_panels(ff5: pd.DataFrame, mom: pd.DataFrame, hml_dev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ff5_panel = ff5[['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']].copy()
    combined = pd.concat(
        [
            ff5_panel[['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']],
            mom[['MOM']],
            hml_dev[['HML_DEV']]
        ],
        axis=1
    )
    combined = combined.dropna()
    aqr6_panel = pd.concat(
        [
            ff5_panel[['Mkt_RF', 'SMB', 'RMW', 'CMA']],
            hml_dev[['HML_DEV']],
            mom[['MOM']],
            ff5_panel[['RF']]
        ],
        axis=1
    )
    aqr6_panel = aqr6_panel[['Mkt_RF', 'SMB',
                             'HML_DEV', 'RMW', 'CMA', 'MOM', 'RF']].dropna()
    return ff5_panel.loc[combined.index], aqr6_panel.loc[combined.index], combined


def prepare_excess_returns(returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    rf_aligned = rf.loc[returns.index]
    excess = returns.sub(rf_aligned, axis=0)
    return excess


def run_factor_model(excess_returns: pd.DataFrame, factors: pd.DataFrame) -> dict:
    y, x = excess_returns.align(factors, join='inner', axis=0)
    x = sm.add_constant(x)
    alphas = {}
    tstats = {}
    betas = []
    residuals = pd.DataFrame(index=y.index)
    r2 = {}
    for col in y.columns:
        model = sm.OLS(y[col], x).fit()
        alphas[col] = model.params['const']
        tstats[col] = model.tvalues['const']
        betas.append(model.params.drop('const'))
        residuals[col] = model.resid
        r2[col] = model.rsquared
    betas_df = pd.DataFrame(betas, index=y.columns)
    return {
        'alphas': pd.Series(alphas),
        'tstats': pd.Series(tstats),
        'betas': betas_df,
        'residuals': residuals,
        'r2': pd.Series(r2)
    }

def build_factor_regression_table(factors: pd.DataFrame, factor_order: list[str]) -> pd.DataFrame:
    """
    Build an AQR-style regression table:
      - Each factor is regressed on all *other* factors in factor_order.
      - Two rows per factor:
          <FACTOR>    : coefficients (intercept annualized in %)
          <FACTOR>_t  : t-stats in parentheses style "(x.xx)"
      - Columns: Intercept, factors in factor_order, R2 (in %).
    The resulting CSV will look very close to the AQR tables.
    """
    factors = factors.dropna().copy()
    rows = {}

    for target in factor_order:
        y = factors[target]
        X = sm.add_constant(factors[factor_order].drop(columns=target))
        model = sm.OLS(y, X).fit()

        # Coefficient row
        coef_row = {}
        # intercept: annualized percent
        coef_row['Intercept'] = f"{model.params['const'] * 12.0 * 100.0:.1f}%"

        # initialize all factor columns as blank, then fill where applicable
        for fct in factor_order:
            if fct == target:
                coef_row[fct] = ""  # blank on diagonal
            else:
                coef_row[fct] = f"{model.params[fct]:.2f}"

        coef_row['R2'] = f"{model.rsquared * 100.0:.0f}%"

        # t-stat row
        t_row = {}
        t_row['Intercept'] = f"({model.tvalues['const']:.2f})"
        for fct in factor_order:
            if fct == target:
                t_row[fct] = ""
            else:
                t_row[fct] = f"({model.tvalues[fct]:.2f})"
        t_row['R2'] = ""

        # store two rows: factor and factor_t (like the article)
        rows[target] = coef_row
        rows[f"{target}_t"] = t_row

    table = pd.DataFrame.from_dict(rows, orient='index')

    # enforce column order: Intercept, factors..., R2
    table = table[['Intercept'] + factor_order + ['R2']]

    return table



def grs_test(alphas: pd.Series, residuals: pd.DataFrame, factors: pd.DataFrame) -> dict:
    alpha_vec = alphas.values.reshape(-1, 1)
    sigma_e = residuals.cov().values
    mean_f = factors.mean().values.reshape(-1, 1)
    sigma_f = factors.cov().values
    inv_sigma_e = np.linalg.pinv(sigma_e)
    inv_sigma_f = np.linalg.pinv(sigma_f)
    T = len(residuals)
    N = residuals.shape[1]
    K = factors.shape[1]
    df2 = T - N - K
    if df2 <= 0:
        raise ValueError('insufficient degrees of freedom for GRS test')
    alpha_term = float(np.squeeze(alpha_vec.T @ inv_sigma_e @ alpha_vec))
    denom = float(np.squeeze(1 + (mean_f.T @ inv_sigma_f @ mean_f)))
    grs_stat = ((T - N - K) / N) * (alpha_term / denom)
    p_val = 1 - f.cdf(grs_stat, N, df2)
    return {'grs': grs_stat, 'df1': N, 'df2': df2, 'p_value': float(p_val)}


def export_table(df: pd.DataFrame, path: Path, index_label: str | None = None) -> None:
    df.to_csv(path, index_label=index_label)
    print(f'saved {path}')


def main() -> None:
    log_header('stage 1: understand assignment')
    print('pipeline configured for MGT 6078 assignment 4')

    log_header('stage 2: inspect and align data')
    print('available files:')
    for item in sorted(DATA_DIR.iterdir()):
        print(f' - {item.name}')
    ff5 = load_ff5(DATA_DIR / 'F-F_Research_Data_5_Factors_2x3.csv')
    print('ff5 factors converted from percent to decimal')
    preview_dataset('ff5 factors preview', ff5)
    mom = load_momentum(DATA_DIR / 'F-F_Momentum_Factor.csv')
    print('momentum factor converted from percent to decimal')
    preview_dataset('momentum factor preview', mom)
    port25 = load_portfolios(DATA_DIR / '25_Portfolios_5x5.csv', skiprows=15)
    print('25 portfolios converted from percent to decimal')
    preview_dataset('25 portfolios preview', port25)
    port10 = load_portfolios(
        DATA_DIR / '10_Industry_Portfolios.csv', skiprows=11)
    print('10 industry portfolios converted from percent to decimal')
    preview_dataset('10 industry portfolios preview', port10)
    hml_path = DATA_DIR / 'The Devil in HMLs Details Factors Monthly.xlsx'
    if not hml_path.exists():
        hml_path = DATA_DIR / 'The Devil in HMLs Details Factors Monthly.csv'
    hml_dev = load_hml_dev(hml_path)
    print('hml dev series loaded in decimal')
    preview_dataset('hml dev preview', hml_dev)
    data_dict, aligned_index = align_dataframes(
        {'ff5': ff5, 'mom': mom, 'port25': port25,
            'port10': port10, 'hml_dev': hml_dev}
    )
    print(f'aligned dataframes shapes:')
    for name, df in data_dict.items():
        print(f' - {name}: {df.shape}')

    log_header('stage 3: build factor panels')
    ff5_panel, aqr6_panel, combined_factors = build_factor_panels(
        data_dict['ff5'], data_dict['mom'], data_dict['hml_dev']
    )
    print(f'ff5 panel shape {ff5_panel.shape}')
    print(f'aqr6 panel shape {aqr6_panel.shape}')
    print(f'combined factor set shape {combined_factors.shape}')

    log_header('stage 4: build test asset panels')
    excess_25 = prepare_excess_returns(
        data_dict['port25'].loc[combined_factors.index], ff5_panel['RF'])
    excess_10 = prepare_excess_returns(
        data_dict['port10'].loc[combined_factors.index], ff5_panel['RF'])
    print(f'25 portfolio excess returns shape {excess_25.shape}')
    print(f'10 industry excess returns shape {excess_10.shape}')

    
    log_header('stage 5: replicate AQR tables 1-4 (factor-on-factor regressions)')

    order_t1 = ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']
    order_t2 = ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    order_t3 = ['Mkt_RF', 'SMB', 'HML_DEV', 'RMW', 'CMA']
    order_t4 = ['Mkt_RF', 'SMB', 'HML_DEV', 'RMW', 'CMA', 'MOM']

    factors_t1 = ff5_panel.loc[combined_factors.index, order_t1]
    factors_t2 = ff5_panel.loc[combined_factors.index, order_t1].join(
        aqr6_panel.loc[combined_factors.index, ['MOM']]
    )[order_t2]
    factors_t3 = aqr6_panel.loc[combined_factors.index, order_t3]
    factors_t4 = aqr6_panel.loc[combined_factors.index, order_t4]

    table_1 = build_factor_regression_table(factors_t1, order_t1)
    table_2 = build_factor_regression_table(factors_t2, order_t2)
    table_3 = build_factor_regression_table(factors_t3, order_t3)
    table_4 = build_factor_regression_table(factors_t4, order_t4)

    print('AQR regression tables computed')

    RESULTS_DIR.mkdir(exist_ok=True)
    export_table(table_1, RESULTS_DIR / 'aqr_table1_ff5.csv', index_label='factor')
    export_table(table_2, RESULTS_DIR / 'aqr_table2_ff5_plus_mom.csv', index_label='factor')
    export_table(table_3, RESULTS_DIR / 'aqr_table3_hmldev.csv', index_label='factor')
    export_table(table_4, RESULTS_DIR / 'aqr_table4_hmldev_plus_mom.csv', index_label='factor')


    log_header('stage 6: run factor regressions')
    ff5_factors_only = ff5_panel[['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']]
    aqr6_factors_only = aqr6_panel[[
        'Mkt_RF', 'SMB', 'HML_DEV', 'RMW', 'CMA', 'MOM']]
    results_ff5_25 = run_factor_model(excess_25, ff5_factors_only)
    print('completed ff5 regressions on 25 portfolios')
    results_aqr6_25 = run_factor_model(excess_25, aqr6_factors_only)
    print('completed aqr6 regressions on 25 portfolios')
    results_ff5_10 = run_factor_model(excess_10, ff5_factors_only)
    print('completed ff5 regressions on 10 industries')
    results_aqr6_10 = run_factor_model(excess_10, aqr6_factors_only)
    print('completed aqr6 regressions on 10 industries')

    log_header('stage 7: compute grs statistics')
    grs_ff5_25 = grs_test(results_ff5_25['alphas'], results_ff5_25['residuals'],
                          ff5_factors_only.loc[results_ff5_25['residuals'].index])
    print(
        f"grs ff5 + 25 portfolios: stat {grs_ff5_25['grs']:.6f}, p {grs_ff5_25['p_value']:.6f}")
    grs_aqr6_25 = grs_test(results_aqr6_25['alphas'], results_aqr6_25['residuals'],
                           aqr6_factors_only.loc[results_aqr6_25['residuals'].index])
    print(
        f"grs aqr6 + 25 portfolios: stat {grs_aqr6_25['grs']:.6f}, p {grs_aqr6_25['p_value']:.6f}")
    grs_ff5_10 = grs_test(results_ff5_10['alphas'], results_ff5_10['residuals'],
                          ff5_factors_only.loc[results_ff5_10['residuals'].index])
    print(
        f"grs ff5 + 10 industries: stat {grs_ff5_10['grs']:.6f}, p {grs_ff5_10['p_value']:.6f}")
    grs_aqr6_10 = grs_test(results_aqr6_10['alphas'], results_aqr6_10['residuals'],
                           aqr6_factors_only.loc[results_aqr6_10['residuals'].index])
    print(
        f"grs aqr6 + 10 industries: stat {grs_aqr6_10['grs']:.6f}, p {grs_aqr6_10['p_value']:.6f}")

    log_header('stage 8: export outputs')
    RESULTS_DIR.mkdir(exist_ok=True)
    export_table(table_1, RESULTS_DIR / 'table_1.csv', index_label='factor')
    export_table(table_2, RESULTS_DIR / 'table_2.csv', index_label='factor')
    export_table(table_3, RESULTS_DIR / 'table_3.csv', index_label='factor')
    export_table(table_4, RESULTS_DIR / 'table_4.csv', index_label='factor')
    alpha_ff5_25 = pd.DataFrame(
        {
            'alpha_monthly_pct': (results_ff5_25['alphas'] * 100).round(4),
            't_stat_alpha': results_ff5_25['tstats'].round(3)
        }
    )
    alpha_aqr6_25 = pd.DataFrame(
        {
            'alpha_monthly_pct': (results_aqr6_25['alphas'] * 100).round(4),
            't_stat_alpha': results_aqr6_25['tstats'].round(3)
        }
    )
    alpha_ff5_10 = pd.DataFrame(
        {
            'alpha_monthly_pct': (results_ff5_10['alphas'] * 100).round(4),
            't_stat_alpha': results_ff5_10['tstats'].round(3)
        }
    )
    alpha_aqr6_10 = pd.DataFrame(
        {
            'alpha_monthly_pct': (results_aqr6_10['alphas'] * 100).round(4),
            't_stat_alpha': results_aqr6_10['tstats'].round(3)
        }
    )
    export_table(alpha_ff5_25, RESULTS_DIR /
                 'alphas_tstats_ff5_25.csv', index_label='portfolio')
    export_table(alpha_aqr6_25, RESULTS_DIR /
                 'alphas_tstats_aqr6_25.csv', index_label='portfolio')
    export_table(alpha_ff5_10, RESULTS_DIR /
                 'alphas_tstats_ff5_10.csv', index_label='portfolio')
    export_table(alpha_aqr6_10, RESULTS_DIR /
                 'alphas_tstats_aqr6_10.csv', index_label='portfolio')
    grs_summary = pd.DataFrame(
        [
            {'model': 'FF5', 'assets': '25 Portfolios', **grs_ff5_25},
            {'model': 'AQR6', 'assets': '25 Portfolios', **grs_aqr6_25},
            {'model': 'FF5', 'assets': '10 Industries', **grs_ff5_10},
            {'model': 'AQR6', 'assets': '10 Industries', **grs_aqr6_10},
        ]
    )
    grs_summary['grs'] = grs_summary['grs'].round(4)

    def format_p(val: float) -> str:
        return '<1e-6' if val < 1e-6 else f'{val:.4f}'
    grs_summary['p_value'] = grs_summary['p_value'].apply(format_p)
    export_table(grs_summary, RESULTS_DIR /
                 'grs_summary.csv', index_label='row')

    log_header('stage 9: transparency checkpoints')
    print('alphas ff5 25 preview:')
    print(results_ff5_25['alphas'].head())
    print('alphas aqr6 25 preview:')
    print(results_aqr6_25['alphas'].head())
    print('alphas ff5 10 preview:')
    print(results_ff5_10['alphas'].head())
    print('alphas aqr6 10 preview:')
    print(results_aqr6_10['alphas'].head())

    log_header('stage 10: stop')
    print('assignment 4 pipeline complete, results stored in results/')


if __name__ == '__main__':
    main()
