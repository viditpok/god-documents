import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# Black-Scholes option pricing with spread
def black_scholes(S, K, T, r, sigma, option_type='call', spread_factor=0.05):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
        # Adjust for bid-ask spread when selling
        price = price * (1 - spread_factor)
        return price, delta
    except:
        return 0, 0

# Volatility smile model
def get_smile_iv(iv30, K, S, df, current_idx, lookback=252):
    moneyness = K / S
    b = 5  # Smile curvature parameter
    iv = iv30 * (1 + b * (moneyness - 1)**2)
    
    # Historical IV stats for this moneyness
    start_idx = max(0, current_idx - lookback)
    historical_ivs = []
    for i in range(start_idx, current_idx):
        S_i = df['Close'].iloc[i]
        K_i = S_i * moneyness
        iv30_i = df['IV30'].iloc[i]
        if pd.notna(iv30_i):
            historical_ivs.append(iv30_i * (1 + b * (K_i / S_i - 1)**2))
    
    if not historical_ivs:
        return iv, False, iv, 0.01
    
    mean_iv = np.mean(historical_ivs)
    std_iv = np.std(historical_ivs) if len(historical_ivs) > 1 else 0.01
    is_high_iv = iv > mean_iv + 0.9 * std_iv
    return iv, is_high_iv, mean_iv, std_iv

# Heston model simulation
def simulate_heston(S0, v0, r, kappa, theta, xi, rho, T, dt, steps, num_paths=10):
    S_paths = np.zeros((num_paths, steps + 1))
    v_paths = np.zeros((num_paths, steps + 1))
    S_paths[:, 0] = S0
    v_paths[:, 0] = v0
    
    for path in range(num_paths):
        z1 = np.random.normal(0, 1, steps)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, steps)
        
        for t in range(steps):
            v_paths[path, t + 1] = v_paths[path, t] + kappa * (theta - v_paths[path, t]) * dt + xi * np.sqrt(max(v_paths[path, t], 0)) * np.sqrt(dt) * z2[t]
            v_paths[path, t + 1] = max(v_paths[path, t + 1], 0)
            S_paths[path, t + 1] = S_paths[path, t] * np.exp((r - 0.5 * v_paths[path, t]) * dt + np.sqrt(max(v_paths[path, t], 0)) * np.sqrt(dt) * z1[t])
    
    # Average paths
    S = np.mean(S_paths, axis=0)
    v = np.mean(v_paths, axis=0)
    iv = np.sqrt(v) * 100
    
    # Cap IVs to historical maximum
    max_historical_iv = df['IV30'].iloc[max(0, current_idx - 252):current_idx].max()
    iv = np.minimum(iv, max_historical_iv)
    iv = np.clip(iv, 0, 100)  # Ensure no negative or extreme IVs
    
    return S, iv, S_paths, v_paths

# Plot volatility smile for last trade
def plot_vol_smile(S, iv30, call_strike, put_strike, upper_breakeven, lower_breakeven, df, current_idx, entry_date):
    try:
        strikes = np.linspace(S * 0.8, S * 1.2, 50)
        ivs = []
        mean_ivs = []
        for K in strikes:
            iv, _, mean_iv, _ = get_smile_iv(iv30, K, S, df, current_idx)
            ivs.append(iv)
            mean_ivs.append(mean_iv)
        
        plt.figure(figsize=(12, 8))
        plt.plot(strikes, ivs, label='Volatility Smile', color='blue', linewidth=2)
        plt.plot(strikes, mean_ivs, label='Historical Mean IV', color='green', linestyle='--', linewidth=1.5)
        plt.axvline(call_strike, color='red', linestyle=':', label=f'Call Strike ({call_strike:.2f})')
        plt.axvline(put_strike, color='purple', linestyle=':', label=f'Put Strike ({put_strike:.2f})')
        plt.axvline(S, color='black', linestyle='--', label=f'Stock Price ({S:.2f})')
        plt.axvline(upper_breakeven, color='orange', linestyle='-.', label=f'Upper Breakeven ({upper_breakeven:.2f})')
        plt.axvline(lower_breakeven, color='orange', linestyle='-.', label=f'Lower Breakeven ({lower_breakeven:.2f})')
        
        # Annotate IV at call and put strikes
        call_iv, _, _, _ = get_smile_iv(iv30, call_strike, S, df, current_idx)
        put_iv, _, _, _ = get_smile_iv(iv30, put_strike, S, df, current_idx)
        plt.text(call_strike, call_iv + 1, f'IV: {call_iv:.2f}%', color='red', fontsize=10, ha='right')
        plt.text(put_strike, put_iv + 1, f'IV: {put_iv:.2f}%', color='purple', fontsize=10, ha='left')
        
        plt.title(f'Volatility Smile for GOOGL Last Trade on {entry_date.date()}')
        plt.xlabel('Strike Price ($)')
        plt.ylabel('Implied Volatility (%)')
        plt.grid(True)
        plt.legend()
        filename = 'vol_smile_last_trade.png'
        plt.savefig(filename)
        plt.close()
        
    except Exception as e:
        print(f"Failed to plot volatility smile: {e}")

# Plot future Heston paths
def plot_future_heston_paths(S0, v0, theta, df, last_date):
    try:
        T = 30 / 365
        r = 0.04
        kappa = 3.0
        xi = 0.2
        rho = -0.7
        dt = T / 30
        steps = 30
        num_paths = 10
        
        S, iv, S_paths, v_paths = simulate_heston(S0, v0, r, kappa, theta, xi, rho, T, dt, steps, num_paths)
        
        days = np.arange(steps + 1)
        plt.figure(figsize=(12, 8))
        
        # Price paths
        plt.subplot(2, 1, 1)
        for i in range(S_paths.shape[0]):
            plt.plot(days, S_paths[i], alpha=0.3, color='blue')
        plt.plot(days, S, label='Average Price Path', color='red', linewidth=2)
        plt.title(f'Future Heston Price Paths for GOOGL from {last_date.date()}')
        plt.xlabel('Days')
        plt.ylabel('Stock Price ($)')
        plt.grid(True)
        plt.legend()
        
        # IV paths
        plt.subplot(2, 1, 2)
        for i in range(v_paths.shape[0]):
            iv_path = np.sqrt(np.maximum(v_paths[i], 0)) * 100
            plt.plot(days, iv_path, alpha=0.3, color='blue')
        plt.plot(days, iv, label='Average IV Path', color='red', linewidth=2)
        plt.title('Future Heston Implied Volatility Paths')
        plt.xlabel('Days')
        plt.ylabel('Implied Volatility (%)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        filename = 'future_heston_paths.png'
        plt.savefig(filename)
        plt.close()
        
    except Exception as e:
        print(f"Failed to plot future Heston paths: {e}")

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['IV30'] = df['IV30'].where(df['IV30'] < 100)
    df['IV30'] = df['IV30'].fillna(method='ffill').fillna(20)  # Handle missing IV30
    return df

# Calculate IV30 statistics
def calculate_iv30_stats(df, current_idx, lookback=252):
    start_idx = max(0, current_idx - lookback)
    iv30_data = df['IV30'].iloc[start_idx:current_idx]
    
    if len(iv30_data) == 0 or iv30_data.isna().all():
        return 20, 0.01, 20, 0, (20/100)**2
    
    mean_iv30 = iv30_data.mean()
    std_iv30 = iv30_data.std() if len(iv30_data) > 1 else 0.01
    top_quartile = iv30_data.quantile(0.75)
    trend_start = max(0, current_idx - 5)
    iv30_trend = df['IV30'].iloc[trend_start:current_idx]
    iv30_change = (iv30_trend.iloc[-1] / iv30_trend.iloc[0] - 1) * 100 if len(iv30_trend) > 1 and iv30_trend.iloc[0] != 0 else 0
    variance_data = (iv30_data / 100)**2
    theta = variance_data.mean() if not variance_data.empty else (mean_iv30 / 100)**2
    return mean_iv30, std_iv30, top_quartile, iv30_change, theta

# Simulate short strangle trade
def short_strangle_trade(df, current_idx, S, iv30, mean_iv30, std_iv30, top_quartile, iv30_change, portfolio_value):
    current_idx  
    T = 30 / 365
    r = 0.04
    call_strike = round(S * 1.027, 2)
    put_strike = round(S * 0.960, 2)
    position_size = 0.02 * portfolio_value
    contracts = max(1, int(position_size / (S * 100)))
    commission_per_contract = 0.50
    
    # Volatility smile IVs
    call_iv, call_high_iv, call_mean_iv, call_std_iv = get_smile_iv(iv30, call_strike, S, df, current_idx)
    put_iv, put_high_iv, put_mean_iv, put_std_iv = get_smile_iv(iv30, put_strike, S, df, current_idx)
    
    # Entry conditions
    iv_threshold = mean_iv30 + 0.9 * std_iv30
    if (pd.isna(iv30) or iv30 <= iv_threshold or iv30_change > 10 or
        not call_high_iv or not put_high_iv):
        return None, None, "IV30 too low, missing, rising too fast, or strikes not high IV"
    
    call_price, call_delta = black_scholes(S, call_strike, T, r, call_iv / 100, 'call', spread_factor=0.05)
    put_price, put_delta = black_scholes(S, put_strike, T, r, put_iv / 100, 'put', spread_factor=0.05)
    
    if call_price == 0 or put_price == 0:
        return None, None, "Invalid option pricing"
    
    total_premium = (call_price + put_price) * 100 * contracts
    commissions = contracts * 2 * commission_per_contract 
    net_premium = total_premium - commissions
    net_delta = (call_delta + put_delta) * contracts * 100
    delta_hedge_shares = -net_delta
    premium_per_share = call_price + put_price
    upper_breakeven = call_strike + premium_per_share
    lower_breakeven = put_strike - premium_per_share
    stop_loss_iv = mean_iv30 + 1.5 * std_iv30
    price_stop_upper = upper_breakeven * 1.02
    price_stop_lower = lower_breakeven * 0.98
    
    # Heston simulation for future paths
    v0 = (iv30 / 100)**2
    kappa = 3.0
    xi = 0.2
    rho = -0.7
    dt = T / 30
    steps = 30
    theta = (mean_iv30 / 100)**2
    S_sim, iv_sim, S_paths, v_paths = simulate_heston(S, v0, r, kappa, theta, xi, rho, T, dt, steps)
    
    # Simulate trade using Heston paths
    profit_target = total_premium * 0.7
    early_exit_target = total_premium * 0.5
    exit_idx = current_idx
    exit_price = S
    exit_reason = "Held to expiration"
    exit_pnl = 0
    max_pnl = 0
    trailing_stop = float('inf')
    
    for i in range(steps + 1):
        S_i = S_sim[i]
        iv30_i = iv_sim[i]
        days_remaining = max(0, 30 - i)
        T_i = days_remaining / 365
        
        # Stop-loss checks
        if pd.notna(iv30_i) and iv30_i > stop_loss_iv:
            exit_idx = current_idx + i
            exit_price = S_i
            exit_reason = "IV30 stop-loss triggered"
            break
        if S_i > price_stop_upper or S_i < price_stop_lower:
            exit_idx = current_idx + i
            exit_price = S_i
            exit_reason = "Price stop-loss triggered"
            break
        
        # Smile-based IV for current step
        call_iv_i, _, _, _ = get_smile_iv(iv30_i, call_strike, S_i, df, current_idx)
        put_iv_i, _, _, _ = get_smile_iv(iv30_i, put_strike, S_i, df, current_idx)
        
        # Mean reversion exit
        if call_iv_i < call_mean_iv + 0.5 * call_std_iv or put_iv_i < put_mean_iv + 0.5 * put_std_iv:
            exit_idx = current_idx + i
            exit_price = S_i
            exit_reason = "Volatility smile mean reversion"
            break
        
        # Current option value (buy back at ask)
        if T_i > 0:
            call_price_i, _ = black_scholes(S_i, call_strike, T_i, r, call_iv_i / 100, 'call', spread_factor=-0.05)
            put_price_i, _ = black_scholes(S_i, put_strike, T_i, r, put_iv_i / 100, 'put', spread_factor=-0.05)
            current_value = (call_price_i + put_price_i) * 100 * contracts
            current_pnl = total_premium - current_value - commissions
            
            # Trailing stop
            if current_pnl > max_pnl:
                max_pnl = current_pnl
                if max_pnl >= total_premium * 0.5:
                    trailing_stop = max_pnl * 0.75
            if current_pnl < trailing_stop:
                exit_idx = current_idx + i
                exit_price = S_i
                exit_reason = "Trailing stop triggered"
                exit_pnl = current_pnl
                break
            
            # Exit checks
            if current_pnl >= profit_target:
                exit_idx = current_idx + i
                exit_price = S_i
                exit_reason = "Profit target reached"
                exit_pnl = current_pnl
                break
            if days_remaining <= 9 and current_pnl >= early_exit_target:
                exit_idx = current_idx + i
                exit_price = S_i
                exit_reason = "Early exit (50% at 21 days)"
                exit_pnl = current_pnl
                break
    
    # Final PnL
    if exit_pnl == 0:
        S_exit = S_sim[-1]
        if S_exit <= put_strike or S_exit >= call_strike:
            intrinsic_call = max(0, S_exit - call_strike) if S_exit > call_strike else 0
            intrinsic_put = max(0, put_strike - S_exit) if S_exit < put_strike else 0
            exit_pnl = total_premium - (intrinsic_call + intrinsic_put) * 100 * contracts - commissions
        else:
            exit_pnl = total_premium - commissions
    
    exit_idx = min(exit_idx, len(df) - 1)
    exit_date = df['Date'].iloc[current_idx] + timedelta(days=30 * i / steps) if i < steps else df['Date'].iloc[exit_idx]
    
    trade = {
        'entry_date': df['Date'].iloc[current_idx],
        'stock_price': S,
        'iv30': iv30,
        'call_strike': call_strike,
        'put_strike': put_strike,
        'contracts': contracts,
        'total_premium': total_premium,
        'commissions': commissions,
        'exit_date': exit_date,
        'exit_price': exit_price,
        'exit_pnl': exit_pnl,
        'exit_reason': exit_reason,
        'delta_hedge_shares': delta_hedge_shares,
        'upper_breakeven': upper_breakeven,
        'lower_breakeven': lower_breakeven
    }
    
    return trade, exit_idx + 1, None

# Backtest strategy
def backtest_strategy(df, initial_portfolio=50000, lookback=252):
    portfolio_value = initial_portfolio
    portfolio_values = [initial_portfolio]
    dates = [df['Date'].iloc[lookback]]
    trades = []
    last_trade = None
    global current_idx
    current_idx = lookback
    
    while current_idx < len(df):
        S = df['Close'].iloc[current_idx]
        iv30 = df['IV30'].iloc[current_idx]
        mean_iv30, std_iv30, top_quartile, iv30_change, theta = calculate_iv30_stats(df, current_idx, lookback)
        
        trade, exit_idx, error = short_strangle_trade(df, current_idx, S, iv30, mean_iv30, std_iv30, top_quartile, iv30_change, portfolio_value)
        
        if trade:
            portfolio_value += trade['exit_pnl']
            portfolio_values.append(portfolio_value)
            dates.append(trade['exit_date'])
            trades.append(trade)
            last_trade = trade  # Track last trade
            current_idx = exit_idx
        else:
            current_idx += 1
        
        portfolio_value = max(portfolio_value, 0)
    
    # Metrics
    cumulative_pnl = sum(t['exit_pnl'] for t in trades)
    num_trades = len(trades)
    win_rate = sum(1 for t in trades if t['exit_pnl'] > 0) / num_trades * 100 if num_trades > 0 else 0
    avg_pnl = cumulative_pnl / num_trades if num_trades > 0 else 0
    portfolio_series = np.array(portfolio_values)
    max_drawdown = 0
    peak = initial_portfolio
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)
    years = (df['Date'].iloc[-1] - df['Date'].iloc[lookback]).days / 365.25
    total_return = (portfolio_value - initial_portfolio) / initial_portfolio * 100
    annualized_return = ((portfolio_value / initial_portfolio) ** (1 / years) - 1) * 100 if years > 0 else 0
    gross_profits = sum(t['exit_pnl'] for t in trades if t['exit_pnl'] > 0)
    gross_losses = abs(sum(t['exit_pnl'] for t in trades if t['exit_pnl'] < 0))
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
    
    metrics = {
        'initial_portfolio': initial_portfolio,
        'final_portfolio': portfolio_value,
        'cumulative_pnl': cumulative_pnl,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor
    }
    
    return trades, metrics, portfolio_values, dates, last_trade

# Save trades to Excel
def save_trades_to_excel(trades, filename='googl_trades.xlsx'):
    trade_data = [{
        'Entry Date': t['entry_date'].date(),
        'Stock Price': t['stock_price'],
        'IV30': t['iv30'],
        'Call Strike': t['call_strike'],
        'Put Strike': t['put_strike'],
        # 'Contracts': t['contracts'],
        'Total Premium': t['total_premium'],
        # 'Commissions': t['commissions'],
        'PnL': t['exit_pnl'],
        'Exit Reason': t['exit_reason'],
        'Delta Hedge Shares': t['delta_hedge_shares'],
        'Upper Breakeven': t['upper_breakeven'],
        'Lower Breakeven': t['lower_breakeven']
    } for t in trades]
    df_trades = pd.DataFrame(trade_data)
    df_trades.to_excel(filename, index=False)
    

# Plot cumulative returns
def plot_cumulative_returns(dates, portfolio_values, initial_portfolio):
    returns = [(pv / initial_portfolio - 1) * 100 for pv in portfolio_values]
    plt.figure(figsize=(10, 6))
    plt.plot(dates, returns, label='Cumulative Returns (%)')
    plt.title('Cumulative Returns of GOOGL Mean Reversion Strategy (2014–2025)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig('cumulative_returns.png')
    plt.close()
    

# Plot PnL histogram
def plot_pnl_histogram(trades):
    pnls = [t['exit_pnl'] for t in trades]
    plt.figure(figsize=(10, 6))
    plt.hist(pnls, bins=50, color='blue', alpha=0.7, label='PnL Distribution')
    plt.axvline(0, color='red', linestyle='--', label='Break-even')
    plt.title('PnL Distribution of GOOGL Mean Reversion Strategy Trades')
    plt.xlabel('PnL ($)')
    plt.ylabel('Number of Trades')
    plt.grid(True)
    plt.legend()
    plt.savefig('pnl_histogram.png')
    plt.close()
    

# Main function
def main():
    global df, current_idx
    file_path = '/Users/anjalikaarora/Downloads/HistoricalPrices_GOOGL.csv'
    df = load_data(file_path)
    trades, metrics, portfolio_values, dates, last_trade = backtest_strategy(df)
    
    # Print metrics
    print("GOOGL Mean Reversion Options Strategy Backtest (2014–2025)")
    print(f"Initial Portfolio: ${metrics['initial_portfolio']:.2f}")
    print(f"Final Portfolio: ${metrics['final_portfolio']:.2f}")
    print(f"Cumulative Profit/Loss: ${metrics['cumulative_pnl']:.2f}")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Average PnL per Trade: ${metrics['avg_pnl']:.2f}")
    # print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}" if metrics['profit_factor'] != float('inf') else "Profit Factor: Undefined (no losses)")
    
    # Save trades
    save_trades_to_excel(trades)
    
    # Plot charts
    plot_cumulative_returns(dates, portfolio_values, metrics['initial_portfolio'])
    plot_pnl_histogram(trades)
    
    # Plot volatility smile for last trade
    if last_trade:
        last_idx = df[df['Date'] == last_trade['entry_date']].index[0]
        plot_vol_smile(
            last_trade['stock_price'],
            last_trade['iv30'],
            last_trade['call_strike'],
            last_trade['put_strike'],
            last_trade['upper_breakeven'],
            last_trade['lower_breakeven'],
            df,
            last_idx,
            last_trade['entry_date']
        )
    
    # Plot future Heston paths
    last_date = df['Date'].iloc[-1]
    last_S = df['Close'].iloc[-1]
    last_iv30 = df['IV30'].iloc[-1]
    _, _, _, _, theta = calculate_iv30_stats(df, len(df) - 1, lookback=252)
    v0 = (last_iv30 / 100)**2
    plot_future_heston_paths(last_S, v0, theta, df, last_date)

if __name__ == "__main__":
    main()