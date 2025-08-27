# Portfolio Analysis: QQQ vs GLD with Modern Portfolio Theory

## Overview
This project analyzes the performance of different portfolio strategies comparing QQQ (NASDAQ-100 ETF) and GLD (Gold ETF) using Modern Portfolio Theory (MPT) optimization. The analysis covers the period from October 2000 to present, providing insights into portfolio rebalancing, risk-adjusted returns, and drawdown analysis.

## Portfolio Performance Chart
![Portfolio Performance Analysis](qqq_gold_mpt_optimized_highres.png)

*Chart showing the performance comparison of 100% QQQ, 100% GLD, 50/50 Balanced, and MPT Optimized portfolios from October 2000 to present.*

## Key Insights from Analysis

### **Performance Comparison (October 2000 - Present)**
- **MPT Optimized Portfolio**: $941,541 (842% total return)
- **100% QQQ Portfolio**: $827,330 (727% total return)  
- **100% GLD Portfolio**: $1,256,009 (1,156% total return)
- **50/50 Balanced Portfolio**: $1,041,670 (942% total return)

### **Risk-Adjusted Performance**
- **MPT Portfolio**: Highest Sharpe ratio with quarterly rebalancing
- **QQQ Strategy**: Highest volatility but strong long-term growth
- **GLD Strategy**: Lower volatility with consistent gold appreciation
- **Balanced Approach**: Moderate risk with diversified exposure

### **Key Findings**
1. **Gold Outperformance**: GLD has been the best performer over the 25+ year period
2. **Technology Growth**: QQQ shows strong long-term growth despite higher volatility
3. **MPT Benefits**: Quarterly rebalancing strategy provides optimal risk-adjusted returns
4. **Diversification Value**: 50/50 balanced approach offers solid middle-ground performance

### **Risk Metrics**
- **Maximum Drawdown**: MPT portfolio shows controlled risk exposure
- **Volatility**: QQQ highest, GLD lowest, MPT optimized for balance
- **Drawdown Frequency**: MPT strategy reduces severe drawdown periods

## Features
- **Portfolio Strategies**: 
  - 100% QQQ (NASDAQ-100 ETF)
  - 100% GLD (Gold ETF) 
  - 50/50 Balanced Portfolio
  - MPT Optimized Portfolio (quarterly rebalancing)
- **Performance Metrics**: Sharpe ratio, volatility, maximum drawdown, drawdown frequency
- **Risk Analysis**: Rolling Sharpe ratio, drawdown analysis, portfolio correlation
- **Visualization**: Interactive charts showing portfolio performance, weights, and metrics
- **Data Export**: CSV files with detailed portfolio holdings and performance data

## Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- `yfinance` - Yahoo Finance data fetching
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `matplotlib` - Chart creation and visualization
- `seaborn` - Statistical data visualization
- `scipy` - Portfolio optimization and statistical functions

## Usage
```bash
python gld_qqq.py
```

## Output Files
- `portfolio_holdings_detailed.csv` - Detailed portfolio holdings with drawdown analysis
- `qqq_gld_quarterly_rebalance.csv` - Quarterly rebalancing data and performance
- `results.csv` - Portfolio performance results and metrics
- `qqq_returns.csv` - QQQ monthly returns
- `gld_returns.csv` - GLD monthly returns
- `data_monthly.csv` - Monthly price data for both assets
- `qqq_gold_mpt_optimized.png` - Portfolio performance chart
- `qqq_gold_mpt_optimized_highres.png` - High-resolution chart with detailed metrics

## Portfolio Strategy Details
1. **100% QQQ**: Pure technology/growth exposure with highest volatility
2. **100% GLD**: Pure gold/precious metals exposure with lowest volatility
3. **50/50 Balanced**: Equal weight allocation providing diversification
4. **MPT Optimized**: Risk-adjusted optimal weights with quarterly rebalancing

## Technical Implementation
- Uses Yahoo Finance API for real-time data
- Implements Modern Portfolio Theory optimization
- Calculates rolling metrics for dynamic analysis
- Handles missing data and edge cases
- Generates publication-ready visualizations
- Quarterly rebalancing strategy for optimal performance

## Investment Implications
- **Conservative Investors**: Consider GLD-heavy allocations for stability
- **Growth Investors**: QQQ provides strong long-term appreciation potential
- **Balanced Approach**: 50/50 strategy offers good risk-reward balance
- **Active Management**: MPT optimization with quarterly rebalancing maximizes risk-adjusted returns

## License
This project is for educational and research purposes.

## Contributing
Feel free to submit issues, feature requests, or pull requests to improve the analysis.
