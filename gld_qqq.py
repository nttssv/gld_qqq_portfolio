import warnings
import yfinance as yf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
from scipy.optimize import minimize
import sys

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress specific matplotlib warnings
plt.rcParams['figure.max_open_warning'] = 0

# Suppress scipy optimization warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.optimize')
warnings.filterwarnings('ignore', message='Values in x were outside bounds during a minimize step')

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout')

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pandas')

# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')
warnings.filterwarnings('ignore', category=UserWarning, module='yfinance')

# Set numpy error handling globally
np.seterr(divide='ignore', invalid='ignore')

# Create output directory if it doesn't exist
output_dir = os.path.dirname(os.path.abspath(__file__))

def optimize_portfolio(returns, max_lookback_years=3, min_data_points=60):
    """
    Optimize portfolio weights using MPT to maximize Sharpe ratio.
    
    Args:
        returns: DataFrame of historical returns (must be in chronological order)
        max_lookback_years: Maximum number of years of historical data to use
        min_data_points: Minimum number of data points required for optimization
        
    Returns:
        Optimal weights as a numpy array
    """
    # Ensure we have enough data
    if len(returns) < min_data_points:
        raise ValueError(f"Not enough data points. Need at least {min_data_points}, got {len(returns)}")
    
    # Limit lookback period
    max_lookback = min(len(returns), 252 * max_lookback_years)
    returns = returns.iloc[-max_lookback:]
    
    n_assets = returns.shape[1]
    
    # Objective function (negative Sharpe ratio)
    def objective(weights, returns):
        portfolio_returns = (returns * weights).sum(axis=1)
        # Use annualized Sharpe ratio
        sharpe = -portfolio_returns.mean() / (portfolio_returns.std() + 1e-10) * np.sqrt(252)
        return sharpe
    
    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0.1, 0.9) for _ in range(n_assets)]  # No short selling, 10-90% allocation
    
    # Initial guess (equal weights)
    init_weights = np.ones(n_assets) / n_assets
    
    # Run optimization from multiple starting points to avoid local minima
    best_sharpe = -np.inf
    best_weights = init_weights
    
    for _ in range(5):  # Try 5 different starting points
        # Generate random weights that sum to 1
        weights = np.random.random(n_assets)
        weights /= weights.sum()
        
        # Run optimization
        result = minimize(
            objective, 
            weights,
            args=(returns,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        # Check if this is the best solution so far
        if result.success and -result.fun > best_sharpe:
            best_sharpe = -result.fun
            best_weights = result.x
    
    return best_weights

def calculate_drawdown(portfolio_values):
    """
    Calculate drawdown series from portfolio values.
    
    Args:
        portfolio_values: Series of portfolio values (not returns)
        
    Returns:
        Series of drawdown percentages
    """
    # Convert to numpy array for faster computation
    values = np.array(portfolio_values)
    
    # Calculate the running maximum (peak) at each point
    peak = np.maximum.accumulate(values)
    
    # Calculate drawdown as percentage
    drawdown = (values - peak) / peak
    
    return pd.Series(drawdown, index=portfolio_values.index)

def calculate_rolling_sharpe(returns, window=36, risk_free_rate=0.0):
    """
    Calculate rolling Sharpe ratio using only historical data.
    
    Args:
        returns: Series of returns
        window: Lookback window in months
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Series of rolling Sharpe ratios
    """
    # Ensure we have enough data
    if len(returns) < window:
        return pd.Series(index=returns.index, dtype=float)
        
    rolling_mean = returns.rolling(window=window, min_periods=window).mean() * 12
    rolling_std = returns.rolling(window=window, min_periods=window).std() * np.sqrt(12)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std
        
    return rolling_sharpe

# Parameters
start_value = 100000
start_date = "1999-01-01"  # Start from January 1999 to ensure we have enough history
analysis_start_date = "2000-10-01"  # Start analysis from October 2000
portfolio_start_date = "2000-10-01"  # All portfolios start at $100,000 on this date
end_date = pd.Timestamp.now().strftime("%Y-%m-%d")  # Current date
tickers = ["QQQ", "GC=F"]  # Using GC=F (Gold Futures)
weights = np.array([0.5, 0.5])
data_export = []  # To store data for export

# Download data
print("Downloading data...")
# Download adjusted close prices for each ticker separately
data_frames = []
for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']
    print(f"\n{ticker} data points:")
    print(f"Start date: {df.index[0].strftime('%Y-%m-%d')}")
    print(f"End date: {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Number of data points: {len(df)}")
    
    # Check for October 2000 data specifically
    oct_2000_data = df['2000-09-15':'2000-10-31']
    if not oct_2000_data.empty:
        print(f"\nOctober 2000 data for {ticker}:")
        print(oct_2000_data)
    
    df.name = ticker
    data_frames.append(df)

# Combine into a single DataFrame
close_prices = pd.concat(data_frames, axis=1)

# Forward fill any missing data points
close_prices = close_prices.ffill().dropna()

# Adjust GC=F prices to be more comparable to GLD (divide by 10)
if 'GC=F' in close_prices:
    close_prices['GC=F'] = close_prices['GC=F'] / 10

data = close_prices

# Resample monthly
data_monthly = data.resample("ME").last()

# Rebalance dates: end of Mar, Jun, Sep, Dec
rebalance_months = [3, 6, 9, 12]
rebalance_dates = [d for d in data_monthly.index if d.month in rebalance_months]

# Portfolio simulation with MPT optimization
portfolio_value = start_value
portfolio_history = []
allocation_history = []

# Track last valid allocation for error recovery
last_valid_alloc = None

def validate_prices(prices, date):
    """Validate price data and check for potential data issues."""
    if prices.isna().any():
        print(f"Warning: Missing price data on {date.strftime('%Y-%m-%d')}")
        return False
            
    if (prices <= 0).any():
        print(f"Warning: Non-positive price on {date.strftime('%Y-%m-%d')}: {prices}")
        return False
            
    # Check for extreme price movements that might indicate data errors
    if len(portfolio_history) > 1:
        prev_prices = portfolio_history[-1]
        for asset in prices.index:
            if asset in prev_prices:
                ret = (prices[asset] - prev_prices[asset]) / prev_prices[asset]
                if abs(ret) > 0.5:  # More than 50% move
                    print(f"Warning: Large price movement for {asset} on {date.strftime('%Y-%m-%d')}: {ret*100:.1f}%")
                        
    return True

# Start with equal weights, but calculate initial allocation based on prices
initial_weights = np.array([0.5, 0.5])

# Find the first valid date where we have data for both assets
first_valid_idx = data_monthly.first_valid_index()
if first_valid_idx is not None:
    alloc = initial_weights * portfolio_value / data_monthly.loc[first_valid_idx].values
else:
    # Fallback to equal allocation if no valid data
    alloc = np.array([portfolio_value / 2] * 2)

# Convert daily data to returns for MPT optimization
# Use the same data source as the portfolio simulation for consistency
daily_returns = data_monthly.pct_change().dropna()

# Filter daily returns to start from October 2000 (the portfolio start date)
portfolio_start_mask_daily = daily_returns.index >= '2000-10-01'
daily_returns = daily_returns[portfolio_start_mask_daily]

# Start from the first date after we have both assets
start_idx = data_monthly.index.get_loc(first_valid_idx) if first_valid_idx is not None else 1

# Add data validation for the initial period
if start_idx >= len(data_monthly):
    raise ValueError("Not enough data to start the simulation")
        
print(f"\nStarting simulation with {len(data_monthly) - start_idx} months of data")
print(f"First date: {data_monthly.index[start_idx].strftime('%Y-%m-%d')}")
print(f"Last date: {data_monthly.index[-1].strftime('%Y-%m-%d')}")

for i in range(start_idx, len(data_monthly)):
    date = data_monthly.index[i]
    prices = data_monthly.iloc[i]
    
    # Validate prices before processing
    if not validate_prices(prices, date):
        continue
        
    # Check for extreme price movements
    if i > 0:
        prev_prices = data_monthly.iloc[i-1]
        returns = (prices - prev_prices) / prev_prices
        if (abs(returns) > 0.5).any():  # More than 50% move
            print(f"\nLarge price movement detected on {date.strftime('%Y-%m-%d')}:")
            for ticker, ret in returns.items():
                print(f"  {ticker}: {ret*100:.2f}% (${prev_prices[ticker]:.2f} -> ${prices[ticker]:.2f})")
    
    # Ensure alloc is a numpy array for consistent indexing
    alloc_array = np.array(alloc)
    
    # Calculate portfolio value before any rebalancing
    portfolio_value = float(np.sum(alloc_array * prices.values))
    
    # Calculate current weights using numpy array operations
    current_weights = (alloc_array * prices.values) / portfolio_value
    
    # Calculate comparison portfolio values using October 2000 as the starting point
    # Find the October 2000 prices for both assets
    oct_2000_mask = (data_monthly.index >= '2000-10-01') & (data_monthly.index < '2000-11-01')
    if oct_2000_mask.any():
        oct_2000_idx = data_monthly.index[oct_2000_mask][0]
        oct_qqq_price = float(data_monthly.loc[oct_2000_idx, 'QQQ'])
        oct_gld_price = float(data_monthly.loc[oct_2000_idx, 'GC=F'])
    else:
        # Fallback to first available prices if October 2000 not found
        oct_qqq_price = float(data_monthly['QQQ'].iloc[0])
        oct_gld_price = float(data_monthly['GC=F'].iloc[0])

    # Calculate the number of shares we could buy with $100,000 in October 2000
    qqq_shares = start_value / oct_qqq_price
    gld_shares = start_value / oct_gld_price

    # Calculate current value of those shares - use proper column names
    current_qqq_price = float(prices['QQQ'])
    current_gld_price = float(prices['GC=F'])

    qqq_only = qqq_shares * current_qqq_price
    gld_only = gld_shares * current_gld_price
    fifty_fifty = (qqq_only + gld_only) / 2
    
    # Print debug information for the first few months
    if len(portfolio_history) < 3:  # Show first 3 months for better understanding
        print("\n" + "="*80, file=sys.stderr)
        print(f"Date: {date}", file=sys.stderr)
        print(f"October 2000 QQQ price: ${oct_qqq_price:.4f}, Current QQQ price: ${current_qqq_price:.4f}", file=sys.stderr)
        print(f"October 2000 GLD price: ${oct_gld_price:.4f}, Current GLD price: ${current_gld_price:.4f}", file=sys.stderr)
        print(f"QQQ shares: {qqq_shares:.4f}, GLD shares: {gld_shares:.4f}", file=sys.stderr)
        print(f"QQQ only value: ${qqq_only:,.2f} ({(qqq_only/start_value-1)*100:,.2f}% return)", file=sys.stderr)
        print(f"GLD only value: ${gld_only:,.2f} ({(gld_only/start_value-1)*100:,.2f}% return)", file=sys.stderr)
        print(f"50/50 value: ${fifty_fifty:,.2f} ({(fifty_fifty/start_value-1)*100:,.2f}% return)", file=sys.stderr)
        print("-" * 50, file=sys.stderr)
        
        # Print the first row of data_monthly for verification
        if len(portfolio_history) == 0:
            print("\nFirst row of data_monthly:", file=sys.stderr)
            print(data_monthly.iloc[[0]], file=sys.stderr)
            
        # Force flush the output
        sys.stderr.flush()
    
    # Ensure all portfolios start at exactly $100,000 in October 2000
    if date.strftime('%Y-%m') == '2000-10':
        # In October 2000, all portfolios start at $100,000
        qqq_only = start_value
        gld_only = start_value
        fifty_fifty = start_value
    # For all other months, the calculated values above are correct
    
    # Record history before rebalancing
    record = {
        "Date": date,
        "Portfolio_Value": portfolio_value,
        "QQQ_Price": current_qqq_price,  # Use the actual QQQ price, not indexed
        "GC=F_Price": current_gld_price,  # Use the actual GC=F price, not indexed
        "QQQ_Weight": current_weights[0],
        "GC=F_Weight": current_weights[1],
        "QQQ_Allocation_USD": float(alloc_array[0] * current_qqq_price),
        "GC=F_Allocation_USD": float(alloc_array[1] * current_gld_price),
        "QQQ_Only": qqq_only,
        "GLD_Only": gld_only,
        "FiftyFifty": fifty_fifty
    }
    portfolio_history.append(record)
    data_export.append(record)
    
    # Rebalance if quarter-end
    if date in rebalance_dates:
        # Calculate new allocation based on current portfolio value
        # Use past 3 years of daily returns for MPT optimization (exclusive of current date)
        lookback_start = daily_returns.index[0]
        lookback_end = date - pd.Timedelta(days=1)  # Exclude current date
        
        if lookback_end <= lookback_start:
            print(f"Warning: Not enough historical data for optimization on {date}")
            continue  # Skip rebalancing this period
            
        returns_for_opt = daily_returns.loc[lookback_start:lookback_end].copy()
        
        # Ensure we have enough data points (at least 3 months of daily data)
        min_required_days = 60
        if len(returns_for_opt) < min_required_days:
            print(f"Warning: Only {len(returns_for_opt)} days of data available for optimization on {date} (need at least {min_required_days})")
            continue  # Skip rebalancing this period
            
        # Ensure we have data for both assets
        if returns_for_opt.isna().any().any():
            print(f"Warning: Missing data in optimization window ending {date}")
            continue  # Skip rebalancing this period
            
        try:
            # Get optimal weights using MPT
            new_weights = optimize_portfolio(returns_for_opt)
            
            # Smooth transition between old and new weights (20% of the way each rebalance)
            smooth_weights = current_weights * 0.8 + new_weights * 0.2
            
            # Calculate new allocation in shares based on current portfolio value
            # This ensures we're not creating money out of thin air
            alloc = smooth_weights * portfolio_value / prices
            
            # Store the last valid allocation in case of future errors
            last_valid_alloc = alloc.copy()
            
            # Verify the new allocation sums to the current portfolio value
            new_portfolio_value = (alloc * prices).sum()
            if not np.isclose(new_portfolio_value, portfolio_value, rtol=1e-5):
                # If there's a discrepancy due to numerical precision, adjust the allocation
                alloc = alloc * (portfolio_value / new_portfolio_value)
                
        except Exception as e:
            print(f"Error during optimization on {date}: {str(e)}")
            if last_valid_alloc is not None:
                alloc = last_valid_alloc  # Revert to last known good allocation
                print("Reverted to last valid allocation")
            else:
                # Fallback to current allocation if no valid allocation exists
                # Fallback to previous weights if optimization fails
                alloc = alloc * portfolio_value / prices
        else:
            # Not enough data yet, keep current weights but update dollar amounts
            alloc = pd.Series(current_weights * portfolio_value / prices.values, index=prices.index)

# Convert to DataFrame and filter by analysis start date
results = pd.DataFrame(portfolio_history)
results['Date'] = pd.to_datetime(results['Date'])
results = results[results['Date'] >= analysis_start_date].set_index("Date")

# Add drawdown to the results
cummax = results["Portfolio_Value"].cummax()
results['Drawdown'] = (results["Portfolio_Value"] / cummax - 1)

# Export price and allocation data to CSV
export_df = pd.DataFrame(data_export)

# Ensure we only include rows that exist in both dataframes
min_length = min(len(export_df), len(results))
export_df = export_df.iloc[:min_length].copy()
results = results.iloc[:min_length].copy()

# Add drawdown to the export dataframe
export_df['Drawdown'] = results['Drawdown'].values

# Set initial investment to the starting value of $100,000
initial_investment = 100000

# Export detailed portfolio data
detailed_df = pd.DataFrame(portfolio_history)
detailed_df.to_csv(os.path.join(output_dir, 'portfolio_holdings_detailed.csv'), index=True)
print(f"\nDetailed portfolio data with drawdown exported to {os.path.join(output_dir, 'portfolio_holdings_detailed.csv')}")

# Add the comparison portfolios to results for charting
results['QQQ_Only'] = detailed_df.set_index('Date')['QQQ_Only']
results['GLD_Only'] = detailed_df.set_index('Date')['GLD_Only']
results['FiftyFifty'] = detailed_df.set_index('Date')['FiftyFifty']
results.to_csv(os.path.join(output_dir, 'results.csv'), index=True)

# Performance metrics
ending_value = results["Portfolio_Value"].iloc[-1]
total_return = (ending_value / start_value - 1) * 100
years = (results.index[-1] - results.index[0]).days / 365.25
cagr = (ending_value / start_value) ** (1/years) - 1
returns = results["Portfolio_Value"].pct_change().dropna()
vol = returns.std() * np.sqrt(12)
sharpe = (returns.mean() * 12) / vol  # annualized Sharpe ratio (rf=0)

# Calculate drawdown using the dedicated function
drawdown = calculate_drawdown(results["Portfolio_Value"])
max_dd = drawdown.min()
max_dd_date = drawdown.idxmin()
max_dd_start = results["Portfolio_Value"][:max_dd_date].idxmax()
max_dd_duration = (max_dd_date - max_dd_start).days

print(f"Initial Investment: ${start_value:,.0f}")
print(f"Ending Value: ${ending_value:,.0f}")
print(f"Total Return: {total_return:.2f}%")
print(f"CAGR: {cagr:.2%}")
print(f"Annualized Volatility: {vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2%}")
print(f"Drawdown Period: {max_dd_duration} days")
print(f"Rebalances: {len(rebalance_dates)}")

# Save CSV
rebalance_file = os.path.join(output_dir, "qqq_gld_quarterly_rebalance.csv")
results.to_csv(rebalance_file)
print(f"Rebalanced portfolio data exported to {rebalance_file}")

# Calculate additional metrics
rolling_sharpe = calculate_rolling_sharpe(returns)

# Calculate max drawdown and its duration
max_dd = drawdown.min()
max_dd_date = drawdown.idxmin()
max_dd_start = results["Portfolio_Value"][:max_dd_date].idxmax()
max_dd_duration = (max_dd_date - max_dd_start).days

# Print drawdown statistics
print(f"\nDrawdown Analysis:")
print(f"Maximum Drawdown: {max_dd:.2%} on {max_dd_date.strftime('%Y-%m-%d')}")
print(f"Drawdown Period: {max_dd_start.strftime('%Y-%m-%d')} to {max_dd_date.strftime('%Y-%m-%d')} ({max_dd_duration} days)")
print(f"Recovery Date: {results[results['Portfolio_Value'] >= results['Portfolio_Value'].loc[max_dd_start]].index[1].strftime('%Y-%m-%d')}" if any(results['Portfolio_Value'] >= results['Portfolio_Value'].loc[max_dd_start]) else "Portfolio has not yet recovered from max drawdown")
print(f"Average Drawdown: {drawdown[drawdown < 0].mean():.2%}")
print(f"Drawdown Frequency: {len(drawdown[drawdown < -0.05])/len(drawdown)*100:.1f}% of months with >5% drawdown")

# Create figure with GridSpec for better layout control
fig = plt.figure(figsize=(18, 20))  # Reduced overall figure height
gs = GridSpec(5, 1, height_ratios=[2, 1, 1, 2, 1], hspace=0.5)  # Reduced heatmap height

# Get the initial investment and calculate the value of 100% QQQ and 100% GC=F
initial_investment = results["Portfolio_Value"].iloc[0]
# Use the same calculation method as in the main simulation for consistency
qqq_only = results["QQQ_Only"]
gold_only = results["GLD_Only"]

# Portfolio Value
ax1 = fig.add_subplot(gs[0])
ax1.plot(results.index, results["Portfolio_Value"], linewidth=2, color='#1f77b4', label='MPT Optimized')
ax1.plot(results.index, qqq_only, '--', linewidth=1.5, color='#2ca02c', alpha=0.7, label='100% QQQ')
ax1.plot(results.index, gold_only, '--', linewidth=1.5, color='#ff7f0e', alpha=0.7, label='100% GC=F')
ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
ax1.set_title('Portfolio Performance Over Time', fontsize=12, pad=15)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower right', framealpha=0.9, facecolor='white', frameon=True, edgecolor='lightgray')

# Add performance annotation
end_value = results["Portfolio_Value"].iloc[-1]
ax1.annotate(f'MPT: ${end_value:,.0f}', 
             xy=(results.index[-1], end_value),
             xytext=(10, 10),
             textcoords='offset points',
             ha='left',
             va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
             


# Add annotation for 100% QQQ
qqq_end_value = results["QQQ_Only"].iloc[-1]
ax1.annotate(f'100% QQQ: ${qqq_end_value:,.0f}', 
             xy=(results.index[-1], qqq_end_value),
             xytext=(10, -20),
             textcoords='offset points',
             ha='left',
             va='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.8))

# Add annotation for 100% GLD
gld_end_value = results["GLD_Only"].iloc[-1]
ax1.annotate(f'100% GLD: ${gld_end_value:,.0f}', 
             xy=(results.index[-1], gld_end_value),
             xytext=(10, 30),
             textcoords='offset points',
             ha='left',
             va='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.8))

ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1000:,.0f}K'))

# Add key metrics as text in the plot
metrics_text = (
    f"CAGR: {cagr*100:.1f}%\n"
    f"Sharpe: {sharpe:.2f}\n"
    f"Max DD: {drawdown.min()*100:.1f}%\n"
    f"Volatility: {vol*100:.1f}%"
)

# Position the text in the upper left corner
ax1.text(
    0.02, 0.98, metrics_text,
    transform=ax1.transAxes,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray'),
    fontsize=11,
    family='monospace'
)

# Get drawdown values for visualization (same as in CSV)
drawdown_vis = calculate_drawdown(results["Portfolio_Value"])

# Add period of max drawdown - using the same drawdown calculation as in the CSV
max_dd_idx = drawdown_vis.idxmin()
max_dd_start_idx = results["Portfolio_Value"][:max_dd_idx].idxmax()
max_dd_period = (max_dd_idx - max_dd_start_idx).days

ax1.axvspan(max_dd_start_idx, max_dd_idx, color='red', alpha=0.1)
ax1.text(
    max_dd_start_idx + (max_dd_idx - max_dd_start_idx)/2, 
    results["Portfolio_Value"].max() * 0.8,
    f'Max DD Period\n{max_dd_period} days',
    ha='center',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray')
)

# Drawdown - use the same calculation as in the CSV
drawdown_vis = calculate_drawdown(results["Portfolio_Value"])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.fill_between(drawdown_vis.index, drawdown_vis * 100, 0, color='#ff7f0e', alpha=0.3)
ax2.set_ylabel("Drawdown (%)", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))

# Rolling Sharpe Ratio (3-year)
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.plot(rolling_sharpe.index, rolling_sharpe, color='#2ca02c')
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax3.set_ylabel("3Y Rolling Sharpe", fontsize=12)
ax3.grid(True, alpha=0.3)

# Create monthly returns table with proper formatting
monthly_returns = results['Portfolio_Value'].resample('ME').last().pct_change()
monthly_returns = monthly_returns[monthly_returns.notna()] * 100  # Convert to percentage
monthly_returns = monthly_returns.to_frame('Return')
monthly_returns['Year'] = monthly_returns.index.year
monthly_returns['Month'] = monthly_returns.index.month_name().str[:3]

# Ensure we have all months for each year
years = monthly_returns['Year'].unique()
all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Monthly returns heatmap
ax4 = fig.add_subplot(gs[3])

# Ensure we're using the same time period as other charts
start_date = results.index[0].strftime('%Y-%m')
end_date = results.index[-1].strftime('%Y-%m')

# Get monthly returns for the same period
monthly_returns = results['Portfolio_Value'].resample('ME').last().pct_change().dropna()
monthly_returns = monthly_returns.to_frame('Return')
monthly_returns['Year'] = monthly_returns.index.year
monthly_returns['Month'] = monthly_returns.index.month

# Pivot with months as rows and years as columns
monthly_returns_pivot = monthly_returns.pivot(index='Month', columns='Year', values='Return') * 100

# Reorder months from January to December
monthly_returns_pivot = monthly_returns_pivot.reindex(range(1, 13))

# Create heatmap without colorbar
heatmap = sns.heatmap(monthly_returns_pivot, cmap='RdYlGn', center=0, annot=True, fmt='.1f', 
                     linewidths=0.5, cbar=False, ax=ax4,
                     annot_kws={'size': 9})  # Increased annotation font size

# Format y-axis to show month names with larger font
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
ax4.set_yticks([x - 0.5 for x in range(1, 13)])
ax4.set_yticklabels(month_names, rotation=0, fontsize=10)

# Format x-axis with years with larger font
ax4.set_xticks([x + 0.5 for x in range(len(monthly_returns_pivot.columns))])
ax4.set_xticklabels(monthly_returns_pivot.columns.astype(int), rotation=45, ha='right', fontsize=10)

# Add value annotations with larger font
for i in range(monthly_returns_pivot.shape[0]):
    for j in range(monthly_returns_pivot.shape[1]):
        val = monthly_returns_pivot.iloc[i, j]
        if pd.notna(val):
            # Use black text for better visibility
            color = 'black' if abs(val) < 10 else 'white'
            ax4.text(j + 0.5, i + 0.5, f'{val:.1f}', ha='center', va='center', 
                    color=color, fontsize=9, fontweight='bold')  # Increased font size

# Adjust title and labels with larger font
ax4.set_title(f"Monthly Returns (%) - {start_date} to {end_date}", fontsize=14, pad=15)
ax4.set_ylabel('Month', fontsize=12)
ax4.set_xlabel('Year', fontsize=12)

# Adjust layout and saveplot with quarterly rebalance markers
ax5 = fig.add_subplot(gs[4], sharex=ax1)
results_df = pd.DataFrame(portfolio_history).set_index('Date')

# Plot allocation percentages
line1, = ax5.plot(results_df.index, results_df['QQQ_Weight']*100, label='QQQ', color='#1f77b4', linewidth=2)
line2, = ax5.plot(results_df.index, results_df['GC=F_Weight']*100, label='GC=F', color='gold', linewidth=2)

# Add horizontal lines at 25%, 50%, 75%
for pct in [25, 50, 75]:
    ax5.axhline(y=pct, color='gray', linestyle='--', alpha=0.3)

# Mark rebalance points
rebalance_dates = [d for d in results_df.index if d.month in [3,6,9,12] and d.day > 25]
for date in rebalance_dates:
    if date in results_df.index:
        ax5.axvline(x=date, color='red', alpha=0.2, linestyle='-', zorder=0)

# Add current allocation text
last_date = results_df.index[-1].strftime('%Y-%m-%d')
last_qqq = results_df['QQQ_Weight'].iloc[-1] * 100
last_gld = results_df['GC=F_Weight'].iloc[-1] * 100

ax5.text(0.02, 0.98, f'Current Allocation ({last_date}):\nQQQ: {last_qqq:.1f}%\nGC=F: {last_gld:.1f}%',
         transform=ax5.transAxes, va='top', ha='left', 
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5'),
         fontsize=9, linespacing=1.5)

ax5.set_ylabel('Allocation %', fontsize=12)
ax5.set_ylim(0, 100)
ax5.grid(True, alpha=0.3)
ax5.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
ax5.set_title('Quarterly Asset Allocation with Rebalance Points (red lines)', fontsize=12, pad=15)

# Format x-axis to show years
years = mdates.YearLocator(2)
year_fmt = mdates.DateFormatter('%Y')
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(year_fmt)
data_monthly.to_csv(os.path.join(output_dir, 'data_monthly.csv'), index=True)

# Calculate monthly returns for each asset using the same data as the portfolio simulation
# Use data_monthly which has the same price adjustments as the simulation
# Filter to start from October 2000 (the portfolio start date)
qqq_returns = data_monthly['QQQ'].pct_change().dropna()
gld_returns = data_monthly['GC=F'].pct_change().dropna()

# Filter returns to start from October 2000 (the portfolio start date)
portfolio_start_mask = qqq_returns.index >= '2000-11-01'
qqq_returns = qqq_returns[portfolio_start_mask]
gld_returns = gld_returns[portfolio_start_mask]
qqq_returns.to_csv(os.path.join(output_dir, 'qqq_returns.csv'), index=True)
gld_returns.to_csv(os.path.join(output_dir, 'gld_returns.csv'), index=True)

# Ensure both series have the same index
common_index = qqq_returns.index.intersection(gld_returns.index)
qqq_returns = qqq_returns[common_index]
gld_returns = gld_returns[common_index]

# Calculate 50:50 portfolio returns (rebalanced monthly)
fifty_fifty_returns = (qqq_returns * 0.5) + (gld_returns * 0.5)

# Calculate MPT portfolio returns (using the actual portfolio returns from the backtest)
if len(returns) > 0:
    returns = pd.Series(returns, index=common_index[1:len(returns)+1] if len(returns) < len(common_index) else common_index[:len(returns)])

# Function to calculate portfolio metrics
def calculate_portfolio_metrics(returns, name):
    # Ensure we're working with monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) if not isinstance(returns.index, pd.DatetimeIndex) else returns
    
    # Calculate total months and years
    total_months = len(monthly_returns)
    years = total_months / 12.0
    
    # Calculate total return and CAGR
    total_return = (1 + monthly_returns).prod() - 1
    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # Calculate volatility (annualized)
    vol = monthly_returns.std() * np.sqrt(12)
    
    # Calculate Sharpe ratio (assuming 0 risk-free rate for simplicity)
    sharpe = (monthly_returns.mean() * 12) / (monthly_returns.std() * np.sqrt(12)) if monthly_returns.std() > 0 else 0
    
    # Calculate max drawdown
    cum_returns = (1 + monthly_returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_dd = drawdown.min() if not drawdown.empty else 0
    
    return {
        'Portfolio': name,
        'CAGR (%)': cagr * 100,
        'Volatility (%)': vol * 100,
        'Max DD (%)': max_dd * 100,
        'Sharpe': sharpe
    }

# Get metrics for all portfolios
mpt_metrics = {
    'Portfolio': 'MPT Portfolio',
    'CAGR (%)': cagr * 100,
    'Volatility (%)': vol * 100,
    'Max DD (%)': drawdown.min() * 100,
    'Sharpe': sharpe
}

qqq_metrics = calculate_portfolio_metrics(qqq_returns, '100% QQQ')
gld_metrics = calculate_portfolio_metrics(gld_returns, '100% GLD')
fifty_fifty_metrics = calculate_portfolio_metrics(fifty_fifty_returns, '50% QQQ/50% GLD')

# Add more metrics to each portfolio
def add_metrics(portfolio_returns):
    # Calculate additional metrics
    years = len(portfolio_returns) / 12
    total_return = (1 + portfolio_returns).prod() - 1
    positive_returns = portfolio_returns[portfolio_returns > 0].mean()
    negative_returns = portfolio_returns[portfolio_returns < 0].mean()
    sortino = (portfolio_returns.mean() / portfolio_returns[portfolio_returns < 0].std()) * np.sqrt(12)
    win_rate = len(portfolio_returns[portfolio_returns > 0]) / len(portfolio_returns) * 100
    
    return {
        'Total Return (%)': total_return * 100,
        'Positive Months (%)': win_rate,
        'Avg. Gain (%)': positive_returns * 100 if not np.isnan(positive_returns) else 0,
        'Avg. Loss (%)': negative_returns * 100 if not np.isnan(negative_returns) else 0,
        'Sortino Ratio': sortino
    }

# Add metrics to each portfolio
for port_returns, port_name in [(returns, 'MPT Portfolio'), 
                              (qqq_returns, '100% QQQ'), 
                              (gld_returns, '100% GLD'),
                              (fifty_fifty_returns, '50% QQQ/50% GLD')]:
    metrics = add_metrics(port_returns)
    for k, v in metrics.items():
        if port_name == 'MPT Portfolio':
            mpt_metrics[k] = v
        elif port_name == '100% QQQ':
            qqq_metrics[k] = v
        elif port_name == '100% GLD':
            gld_metrics[k] = v
        else:
            fifty_fifty_metrics[k] = v

# Create a table with all metrics
metrics_df = pd.DataFrame([mpt_metrics, qqq_metrics, gld_metrics, fifty_fifty_metrics])
metrics_df = metrics_df.set_index('Portfolio').T

# Reorder the rows for better readability
row_order = [
    'CAGR (%)', 'Total Return (%)', 'Volatility (%)', 'Max DD (%)',
    'Sharpe', 'Sortino Ratio', 'Positive Months (%)',
    'Avg. Gain (%)', 'Avg. Loss (%)'
]
metrics_df = metrics_df.loc[row_order]

# Adjust layout to make space for the table
plt.subplots_adjust(bottom=0.25)  # Make more space at the bottom

# Create a table axis at the bottom of the figure
ax_table = fig.add_axes([0.1, 0.05, 0.8, 0.15])  # Smaller, more compact table
ax_table.axis('off')

# Make the table more compact
table_props = {
    'cellLoc': 'center',
    'loc': 'center',
    'bbox': [0, 0, 1, 1],
    'cellColours': [['#f7f7f7']*len(metrics_df.columns)]*len(metrics_df.index),
    'colColours': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
    'cellLoc': 'center',
    'edges': 'closed',
    'fontsize': 8  # Smaller font size
}

# Create the table with compact formatting
col_widths = [0.15, 0.15, 0.1, 0.1, 0.1]  # Adjust these values as needed
table = ax_table.table(
    cellText=np.round(metrics_df.values, 2),
    rowLabels=metrics_df.index,
    colLabels=metrics_df.columns,
    colWidths=col_widths,
    **table_props
)

# Style the table
table.auto_set_font_size(False)
#table.auto_set_column_width([i for i in range(len(metrics_df.columns))])
table.scale(0.8, 1)  # Slightly less horizontal scaling
# Set custom column widths to make them narrower


# Style cells
for (row, col), cell in table.get_celld().items():
    if row == 0:  # Header row
        cell.set_text_props(weight='bold', color='white', fontsize=8)
        cell.set_facecolor('#1f77b4')
    elif col == -1:  # Row labels column
        cell.set_text_props(weight='bold', fontsize=8)
    
    cell.set_edgecolor('lightgray')
    cell.set_linewidth(0.3)
    cell.set_height(0.08)  # Make rows more compact

# Add a compact title
ax_table.set_title('Portfolio Performance Comparison', fontsize=9, pad=10, weight='bold')

# Save the plot using absolute paths with tight layout
plt.tight_layout(rect=[0, 0.05, 1, 0.97])  # Adjust the bottom margin

# Save standard resolution
plot_filename = os.path.join(output_dir, "qqq_gold_mpt_optimized.png")
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Portfolio optimization plot saved to {plot_filename}")


# Print risk metrics table
print("\nPortfolio Risk Metrics:")
print("-" * 50)
print(f"{'Start Date:':<25} {results.index[0].strftime('%Y-%m-%d'):<25}")
print(f"{'End Date:':<25} {results.index[-1].strftime('%Y-%m-%d'):<25}")
print(f"{'Total Return (%):':<25} {total_return:.2f}%")
print(f"{'CAGR (%):':<25} {cagr*100:.2f}%")
print(f"{'Annual Volatility (%):':<25} {vol*100:.2f}%")
print(f"{'Max Drawdown (%):':<25} {drawdown.min()*100:.2f}%")
print(f"{'Sharpe Ratio:':<25} {sharpe:.2f}")
print(f"{'Sortino Ratio:':<25} {np.mean(returns > 0) / np.std(returns[returns < 0]) * np.sqrt(12):.2f}")
print(f"{'Winning Months (%):':<25} {len(returns[returns > 0]) / len(returns) * 100:.1f}%")
print(f"{'Best Month (%):':<25} {returns.max()*100:.1f}%")
print(f"{'Worst Month (%):':<25} {returns.min()*100:.1f}%")
print("-" * 50)

# Show the plot
#plt.show(block=True)