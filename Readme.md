# 📊 Smart Portfolio Optimizer

A comprehensive interactive dashboard for portfolio optimization using Modern Portfolio Theory (Markowitz), active portfolio management (Treynor-Black), value stock screening, Monte Carlo simulation, and portfolio comparison.
🌐 Live Demo
Try it yourself:(https://smart-portfolio-optimizer-0cr4.onrender.com/)

Note: The free tier may take 1-5 minutes to wake up after inactivity.
📊 Dashboard Preview
<img width="928" height="434" alt="image" src="https://github.com/user-attachments/assets/43ef8a0b-ca78-476d-abb3-7044d6d54778" />



## 🚀 Features

### 1. **Markowitz Portfolio Optimization (Passive)**
- Efficient frontier visualization with optimal portfolio
- Multiple optimization goals:
  - Maximum Sharpe Ratio (best risk-adjusted returns)
  - Minimum Volatility (safest portfolio)
  - Target Return (8%, 10%, 12%)
  - Equal Weight (diversification benchmark)
- Comprehensive performance metrics:
  - Annualized Return & Volatility
  - Sharpe & Sortino Ratios
  - Maximum Drawdown & Calmar Ratio
  - **Value at Risk (VaR 95% & 99%)**
  - **Conditional Value at Risk (CVaR 95%)**
  - Skewness & Kurtosis
- Rolling Sharpe ratio (3-month window)
- Sector allocation pie chart
- Cumulative returns vs benchmark

### 2. **Treynor-Black Active Portfolio**
- Calculates alpha and beta for each selected stock
- Identifies stocks with statistically significant alpha (p-value < 0.10)
- Builds active portfolio weighted by alpha/residual variance
- Combines active bets with passive market portfolio
- Shows active weights and final combined portfolio
- Performance metrics for active strategy

### 3. **Value Stock Screener**
- Multi-factor scoring system:
  - P/E Ratio (value factor)
  - P/B Ratio (value factor)
  - Historical returns (momentum)
  - Volatility (risk factor)
  - Dividend yield (income factor)
- Color-coded recommendations:
  - 🟢 Strong Buy (7+ points)
  - 🟡 Buy (5-6 points)
  - 🟠 Hold (3-4 points)
  - 🔴 Sell (0-2 points)
- Correlation heatmap for selected value stocks

### 4. **Monte Carlo Simulation**
- 1,000 simulations over 1-year horizon
- Distribution of final portfolio values
- 95% confidence intervals
- Sample paths visualization
- Key risk metrics:
  - Mean and median final values
  - VaR at 95% and 99% confidence
  - Probability of loss
  - Probability of significant gains (20%+)

### 5. **Portfolio Comparison**
- Compare three strategies side-by-side:
  - Your selected optimization goal
  - Equal weight portfolio
  - Minimum volatility portfolio
- Cumulative returns comparison
- Performance metrics table
- Risk metrics comparison (VaR, CVaR)

### 6. **Additional Features**
- Custom ticker addition
- Download portfolio weights as CSV
- Download performance metrics as CSV
- Interactive controls with real-time updates
- Responsive design with animations

## 📋 Requirements

- Python 3.8 or higher
- Internet connection (for Yahoo Finance data)

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ayeayemyat-miso/smart-portfolio-optimizer
cd smart-portfolio-optimizer
