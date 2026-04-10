# app/dashboard.py - COMPLETE WORKING VERSION FOR RENDER
"""
Smart Portfolio Optimizer - Complete Version with Date Range Selection
Includes: Markowitz, Treynor-Black, Value Screener, Monte Carlo, Portfolio Comparison
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import yfinance as yf
import base64
import io
from scipy import stats
import time
import requests

warnings.filterwarnings('ignore')

from core.data_fetcher import PortfolioDataFetcher
from core.optimizer import PortfolioOptimizer
from core.evaluator import PerformanceEvaluator
from core.treynor_black import TreynorBlackOptimizer

# ============================================================
# RENDER FALLBACK - ALPHA VANTAGE FOR WHEN YAHOO FINANCE IS BLOCKED
# ============================================================

def get_valuation_metrics_fallback(ticker):
    """Fallback function for Render when Yahoo Finance is blocked"""
    api_key = os.environ.get("ALPHA_VANTAGE_KEY")
    if not api_key:
        return None
    
    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data and 'Symbol' in data:
                pe = data.get('PERatio')
                pb = data.get('PriceToBookRatio')
                div = data.get('DividendYield', '0')
                sector = data.get('Sector', 'Unknown')
                
                try:
                    div = float(div) / 100 if div and div != 'None' else 0
                except:
                    div = 0
                
                return {
                    'pe': float(pe) if pe and pe != 'None' else None,
                    'pb': float(pb) if pb and pb != 'None' else None,
                    'dividend': div,
                    'sector': sector,
                    'roe': None
                }
    except Exception as e:
        print(f"Alpha Vantage fallback error for {ticker}: {e}")
    
    return None

# Enhanced valuation function that tries Yahoo Finance first, then Alpha Vantage
def get_valuation_metrics(ticker):
    """Get valuation metrics for a stock with fallback for Render"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        result = {
            'pe': info.get('trailingPE', None),
            'pb': info.get('priceToBook', None),
            'dividend': info.get('dividendYield', 0),
            'sector': info.get('sector', 'Unknown'),
            'roe': info.get('returnOnEquity', None)
        }
        
        # If on Render and Yahoo Finance returned None, try Alpha Vantage
        if os.environ.get("RENDER") and result.get('pe') is None:
            print(f"Yahoo Finance failed for {ticker}, trying Alpha Vantage fallback...")
            fallback = get_valuation_metrics_fallback(ticker)
            if fallback:
                return fallback
        
        return result
    except:
        # If on Render, try Alpha Vantage
        if os.environ.get("RENDER"):
            fallback = get_valuation_metrics_fallback(ticker)
            if fallback:
                return fallback
        return {'pe': None, 'pb': None, 'dividend': 0, 'sector': 'Unknown', 'roe': None}

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Smart Portfolio Optimizer"

# ============================================================
# Expanded Stock Universe (35 well-diversified stocks)
# ============================================================
EXPANDED_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'ADBE', 'CRM', 'ORCL',
    'AMD', 'INTC', 'QCOM', 'TXN', 'JPM', 'V', 'MA', 'BAC', 'GS', 'AXP',
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'WMT', 'HD', 'COST', 
    'MCD', 'PG', 'KO', 'XOM', 'CVX', 'CAT', 'BA', 'GE'
]

OPTIMIZATION_GOALS = [
    {'label': '🚀 Maximum Sharpe Ratio', 'value': 'sharpe'},
    {'label': '🛡️ Minimum Volatility', 'value': 'min_vol'},
    {'label': '🎯 Target Return 6%', 'value': 'target_6'},
    {'label': '🎯 Target Return 8%', 'value': 'target_8'},
    {'label': '🎯 Target Return 10%', 'value': 'target_10'},
    {'label': '🎯 Target Return 12%', 'value': 'target_12'},
    {'label': '🎯 Target Return 15%', 'value': 'target_15'},
    {'label': '🎯 Target Return 20%', 'value': 'target_20'},
    {'label': '⚖️ Equal Weight', 'value': 'equal'}
]

TARGET_RETURNS = {
    'target_6': 0.06, 'target_8': 0.08, 'target_10': 0.10,
    'target_12': 0.12, 'target_15': 0.15, 'target_20': 0.20
}

BENCHMARKS = {
    'S&P 500': '^GSPC', 'Nasdaq 100': '^NDX',
    'Dow Jones': '^DJI', 'Russell 2000': '^RUT'
}

# Global variables
_global_returns = None
_global_sp500 = None
_global_all_tickers = None
_company_name_cache = {}

# ============================================================
# Helper Functions
# ============================================================

def get_company_name(ticker):
    global _company_name_cache
    if ticker in _company_name_cache:
        return _company_name_cache[ticker]
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get('longName', ticker)
        _company_name_cache[ticker] = name
        return name
    except:
        _company_name_cache[ticker] = ticker
        return ticker

def get_company_names_for_tickers(tickers):
    company_names = {}
    for ticker in tickers:
        company_names[ticker] = get_company_name(ticker)
        time.sleep(0.05)
    return company_names

def get_sector_for_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('sector', 'Other')
    except:
        return 'Other'

def compute_var_cvar(returns, confidence=0.95):
    if len(returns) < 2:
        return np.nan, np.nan
    var = np.percentile(returns, (1-confidence)*100)
    cvar = returns[returns <= var].mean()
    return var, cvar

def compute_skewness_kurtosis(returns):
    if len(returns) < 3:
        return np.nan, np.nan
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)
    return skew, kurt

def compute_rolling_sharpe(returns, window=63, rf=0.03/252):
    rolling_returns = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_returns - rf) / rolling_vol
    return rolling_sharpe

def fetch_data_for_dates(start_date, end_date, tickers):
    global _global_returns, _global_sp500, _global_all_tickers
    
    print(f"\n📊 Fetching data from {start_date.date()} to {end_date.date()}")
    print(f"   Fetching {len(tickers)} stocks...")
    
    prices = PortfolioDataFetcher.fetch_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date
    )
    
    if prices.empty:
        print("❌ No data fetched!")
        return None, None, None
    
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    
    daily_returns, _ = PortfolioDataFetcher.calculate_returns(prices)
    if daily_returns.index.tz is not None:
        daily_returns.index = daily_returns.index.tz_localize(None)
    
    benchmark_returns = None
    try:
        sp500 = yf.Ticker("^GSPC")
        sp500_hist = sp500.history(start=start_date, end=end_date)
        if not sp500_hist.empty:
            if sp500_hist.index.tz is not None:
                sp500_hist.index = sp500_hist.index.tz_localize(None)
            benchmark_returns = sp500_hist['Close'].pct_change().dropna()
            benchmark_returns = benchmark_returns.reindex(daily_returns.index, method='ffill')
    except Exception as e:
        print(f"⚠️ Benchmark not available: {e}")
    
    _global_returns = daily_returns
    _global_sp500 = benchmark_returns
    _global_all_tickers = daily_returns.columns.tolist()
    
    print(f"✅ Data loaded: {len(_global_all_tickers)} stocks, {len(daily_returns)} days")
    
    return daily_returns, benchmark_returns, _global_all_tickers

def calculate_value_score(pe, pb, annual_return, annual_vol, dividend):
    score = 0
    if pe and isinstance(pe, (int, float)) and pe > 0:
        if pe < 15: score += 3
        elif pe < 20: score += 2
        elif pe < 25: score += 1
    if pb and isinstance(pb, (int, float)) and pb > 0:
        if pb < 2: score += 2
        elif pb < 3: score += 1
    if annual_return > 0.20: score += 3
    elif annual_return > 0.10: score += 2
    elif annual_return > 0.05: score += 1
    if annual_vol < 0.15: score += 2
    elif annual_vol < 0.25: score += 1
    if dividend and dividend > 0.02: score += 1
    return score

def get_recommendation(score):
    if score >= 7:
        return {'text': 'Strong Buy', 'color': '#155724', 'bg': '#d4edda', 'emoji': '🟢'}
    elif score >= 5:
        return {'text': 'Buy', 'color': '#856404', 'bg': '#fff3cd', 'emoji': '🟡'}
    elif score >= 3:
        return {'text': 'Hold', 'color': '#856404', 'bg': '#fff3cd', 'emoji': '🟠'}
    else:
        return {'text': 'Sell', 'color': '#721c24', 'bg': '#f8d7da', 'emoji': '🔴'}

def create_excel_report(weights_df, metrics, port_returns, selected_tickers, start_date, end_date):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        weights_df.to_excel(writer, sheet_name='Portfolio Weights', index=False)
        metrics_df = pd.DataFrame([metrics]).T
        metrics_df.columns = ['Value']
        metrics_df.index.name = 'Metric'
        metrics_df.to_excel(writer, sheet_name='Performance Metrics')
        info_df = pd.DataFrame({
            'Info': ['Analysis Start Date', 'Analysis End Date', 'Number of Assets', 'Generated On'],
            'Value': [start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), 
                     len(selected_tickers), datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })
        info_df.to_excel(writer, sheet_name='Report Info', index=False)
        output.seek(0)
    return output

# ============================================================
# Layout
# ============================================================
app.layout = html.Div([
    html.Div([
        html.H1("📊 Smart Portfolio Optimizer", 
                style={'text-align': 'center', 'color': '#2c3e50', 'margin-top': '20px'}),
        html.P("Build optimal portfolios using Modern Portfolio Theory | Select any date range for analysis",
               style={'text-align': 'center', 'color': '#7f8c8d', 'margin-bottom': '20px'}),
        html.Hr(),
    ]),
    
    html.Div([
        html.Div([
            html.H3("⚙️ Portfolio Settings"),
            html.Label("📅 Date Range:", style={'font-weight': 'bold', 'margin-top': '10px'}),
            dcc.DatePickerRange(
                id='date-range',
                start_date=(datetime.now() - timedelta(days=2*365)).date(),
                end_date=datetime.now().date(),
                display_format='YYYY-MM-DD',
                style={'width': '100%', 'margin-bottom': '20px'}
            ),
            html.Label("📈 Select Stocks:", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='stock-selector',
                options=[{'label': ticker, 'value': ticker} for ticker in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']],
                value=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
                multi=True,
                placeholder="Search and select stocks...",
                style={'margin-bottom': '20px'}
            ),
            html.Label("🎯 Optimization Goal:", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='goal-selector',
                options=OPTIMIZATION_GOALS,
                value='sharpe',
                style={'margin-bottom': '20px'}
            ),
            html.Label("💰 Risk-Free Rate (%):", style={'font-weight': 'bold'}),
            dcc.Slider(
                id='rf-slider',
                min=0, max=5, step=0.25, value=3,
                marks={i: f'{i}%' for i in range(0, 6)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Label("📊 Benchmark:", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='benchmark-selector',
                options=[{'label': k, 'value': v} for k, v in BENCHMARKS.items()],
                value='^GSPC',
                style={'margin-bottom': '20px'}
            ),
            html.Button("📥 Download Excel Report", id="download-excel-btn", n_clicks=0,
                       style={'background': '#27ae60', 'color': 'white', 'border': 'none',
                              'padding': '10px 20px', 'border-radius': '5px', 
                              'cursor': 'pointer', 'margin-top': '10px', 'width': '100%'}),
            dcc.Download(id="download-excel"),
            html.Div(id='data-status', style={'margin-top': '10px', 'padding': '10px', 
                                              'background': '#e8f4fd', 'border-radius': '8px',
                                              'font-size': '12px', 'text-align': 'center'}),
            html.Div(id='summary-box', style={'margin-top': '20px', 'padding': '15px', 
                                             'background': '#f8f9fa', 'border-radius': '10px'})
        ], style={'width': '28%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'}),
        
        html.Div([
            dcc.Graph(id='frontier', style={'height': '400px'}),
            dcc.Graph(id='weights', style={'height': '400px'}),
            dcc.Graph(id='rolling-sharpe', style={'height': '400px'}),
            dcc.Graph(id='sector-allocation', style={'height': '400px'}),
            html.Div(id='metrics', style={'display': 'flex', 'flex-wrap': 'wrap', 'gap': '10px', 'margin': '10px 0'})
        ], style={'width': '70%', 'display': 'inline-block', 'margin-left': '1%'})
    ]),
    
    html.Hr(),
    
    dcc.Tabs(id='tabs', value='markowitz', children=[
        dcc.Tab(label='📊 Markowitz', value='markowitz'),
        dcc.Tab(label='📈 Treynor-Black', value='active'),
        dcc.Tab(label='💰 Value Screener', value='value'),
        dcc.Tab(label='🔄 Monte Carlo', value='monte'),
        dcc.Tab(label='⚖️ Compare', value='compare'),
    ]),
    html.Div(id='tab-content', style={'padding': '20px', 'margin-top': '20px'})
])

# ============================================================
# Data Loading Callback - SIMPLIFIED FOR RENDER
# ============================================================

@app.callback(
    [Output('data-status', 'children'),
     Output('stock-selector', 'options'),
     Output('stock-selector', 'value')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def load_data_on_date_change(start_date, end_date):
    """Load data when date range changes - Simplified for Render"""
    global _global_returns, _global_sp500, _global_all_tickers
    
    # Default fallback options
    default_options = [{'label': ticker, 'value': ticker} for ticker in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMZN']]
    default_value = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']
    
    if not start_date or not end_date:
        return "⚠️ Please select date range", default_options, default_value
    
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start >= end:
            return "⚠️ End date must be after start date", default_options, default_value
        
        # Try to fetch data
        daily_returns, benchmark_returns, tickers = fetch_data_for_dates(start, end, EXPANDED_TICKERS)
        
        if daily_returns is None or daily_returns.empty:
            return "❌ No data available for selected dates", default_options, default_value
        
        _global_returns = daily_returns
        _global_sp500 = benchmark_returns
        _global_all_tickers = tickers
        
        # Create simple options (just tickers, no company names to avoid errors)
        options = [{'label': ticker, 'value': ticker} for ticker in tickers]
        
        # Default selection
        default_selection = [t for t in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'] if t in tickers]
        if len(default_selection) < 2:
            default_selection = tickers[:5]
        
        status = f"✅ Data loaded: {len(tickers)} stocks | {len(daily_returns)} days"
        
        return status, options, default_selection
        
    except Exception as e:
        print(f"Error in load_data: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)[:50]}", default_options, default_value

# ============================================================
# Excel Download Callback
# ============================================================

@app.callback(
    Output("download-excel", "data"),
    Input("download-excel-btn", "n_clicks"),
    [State('stock-selector', 'value'),
     State('goal-selector', 'value'),
     State('rf-slider', 'value'),
     State('date-range', 'start_date'),
     State('date-range', 'end_date')],
    prevent_initial_call=True
)
def download_excel(n_clicks, selected_tickers, goal, rf, start_date, end_date):
    global _global_returns
    
    if _global_returns is None or not selected_tickers or len(selected_tickers) < 2:
        return None
    
    rets = _global_returns[selected_tickers].dropna()
    rf_dec = rf / 100
    optimizer = PortfolioOptimizer(rets, risk_free_rate=rf_dec)
    
    if goal == 'sharpe':
        weights = optimizer.optimize_max_sharpe()
        title = "Max Sharpe"
    elif goal == 'min_vol':
        weights = optimizer.optimize_min_volatility()
        title = "Min Volatility"
    elif goal in TARGET_RETURNS:
        weights = optimizer.optimize_target_return(TARGET_RETURNS[goal])
        title = f"Target {TARGET_RETURNS[goal]*100:.0f}%"
    else:
        weights = np.array([1/len(selected_tickers)] * len(selected_tickers))
        title = "Equal Weight"
    
    if weights is None:
        weights = np.array([1/len(selected_tickers)] * len(selected_tickers))
    
    ret, risk, sharpe = optimizer.portfolio_performance(weights)
    port_returns = rets.dot(weights)
    cum = (1 + port_returns).cumprod()
    metrics = PerformanceEvaluator.get_all_metrics(port_returns, cum, risk_free_rate=rf_dec)
    
    var95, cvar95 = compute_var_cvar(port_returns, 0.95)
    var99, cvar99 = compute_var_cvar(port_returns, 0.99)
    metrics['VaR (95%)'] = f'{var95*100:.2f}%' if not np.isnan(var95) else 'N/A'
    metrics['CVaR (95%)'] = f'{cvar95*100:.2f}%' if not np.isnan(cvar95) else 'N/A'
    metrics['VaR (99%)'] = f'{var99*100:.2f}%' if not np.isnan(var99) else 'N/A'
    metrics['Portfolio Strategy'] = title
    metrics['Number of Assets'] = len(selected_tickers)
    metrics['Risk-Free Rate'] = f"{rf}%"
    metrics['Analysis Period'] = f"{start_date} to {end_date}"
    
    weights_df = pd.DataFrame({'Asset': selected_tickers, 'Weight (%)': weights * 100})
    weights_df = weights_df.sort_values('Weight (%)', ascending=False)
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.now()
    end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
    
    excel_data = create_excel_report(weights_df, metrics, port_returns, selected_tickers, start_dt, end_dt)
    
    return dcc.send_bytes(excel_data.getvalue(), f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

# ============================================================
# Markowitz Callback
# ============================================================

@app.callback(
    [Output('frontier', 'figure'),
     Output('weights', 'figure'),
     Output('rolling-sharpe', 'figure'),
     Output('sector-allocation', 'figure'),
     Output('summary-box', 'children'),
     Output('metrics', 'children')],
    [Input('stock-selector', 'value'),
     Input('goal-selector', 'value'),
     Input('rf-slider', 'value'),
     Input('benchmark-selector', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_markowitz(selected_tickers, goal, rf, benchmark_symbol, start_date, end_date):
    global _global_returns, _global_sp500
    
    if _global_returns is None:
        empty = go.Figure()
        empty.update_layout(title="Loading data... Please wait")
        return empty, empty, empty, empty, "Loading data...", []
    
    if not selected_tickers or len(selected_tickers) < 2:
        empty = go.Figure()
        empty.update_layout(title="Select at least 2 stocks")
        return empty, empty, empty, empty, "Select at least 2 stocks", []
    
    available_tickers = [t for t in selected_tickers if t in _global_returns.columns]
    if len(available_tickers) < 2:
        empty = go.Figure()
        empty.update_layout(title="Selected stocks not available in data")
        return empty, empty, empty, empty, "Selected stocks not available", []
    
    rets = _global_returns[available_tickers].dropna()
    rf_dec = rf / 100
    optimizer = PortfolioOptimizer(rets, risk_free_rate=rf_dec)
    
    if goal == 'sharpe':
        weights = optimizer.optimize_max_sharpe()
        title = "Maximum Sharpe Ratio Portfolio"
    elif goal == 'min_vol':
        weights = optimizer.optimize_min_volatility()
        title = "Minimum Volatility Portfolio"
    elif goal in TARGET_RETURNS:
        weights = optimizer.optimize_target_return(TARGET_RETURNS[goal])
        title = f"Target Return {TARGET_RETURNS[goal]*100:.0f}% Portfolio"
    else:
        weights = np.array([1/len(available_tickers)] * len(available_tickers))
        title = "Equal Weight Portfolio"
    
    if weights is None:
        weights = np.array([1/len(available_tickers)] * len(available_tickers))
    
    ret, risk, sharpe = optimizer.portfolio_performance(weights)
    
    # Efficient Frontier
    frontier = optimizer.efficient_frontier()
    fig_front = go.Figure()
    if not frontier.empty:
        frontier_points = frontier[frontier['target'].apply(lambda x: isinstance(x, (int, float)))]
        if not frontier_points.empty:
            fig_front.add_trace(go.Scatter(
                x=frontier_points['risk'],
                y=frontier_points['return'],
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='#3498db', width=2)
            ))
    
    fig_front.add_trace(go.Scatter(
        x=[risk],
        y=[ret * 100],
        mode='markers',
        name='Your Portfolio',
        marker=dict(size=20, color='red', symbol='star')
    ))
    fig_front.update_layout(title=f'{title}', xaxis_title='Risk (%)', yaxis_title='Return (%)', template='plotly_white')
    
    # Weights chart
    weights_df = pd.DataFrame({'Asset': available_tickers, 'Weight (%)': weights * 100})
    weights_df = weights_df.sort_values('Weight (%)', ascending=False)
    fig_weights = go.Figure(go.Bar(
        x=weights_df['Asset'],
        y=weights_df['Weight (%)'],
        text=weights_df['Weight (%)'].round(1),
        textposition='outside',
        marker_color='#2ecc71'
    ))
    fig_weights.update_layout(title='Portfolio Allocation', yaxis_title='Weight (%)', template='plotly_white')
    
    # Rolling Sharpe
    port_returns = rets.dot(weights)
    rolling_sharpe = compute_rolling_sharpe(port_returns, window=63, rf=rf_dec/252)
    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        mode='lines',
        name='3-Month Rolling Sharpe',
        line=dict(color='#f1c40f', width=2),
        fill='tozeroy'
    ))
    fig_rolling.update_layout(title='Rolling Sharpe Ratio (3-Month Window)', 
                              xaxis_title='Date', yaxis_title='Sharpe Ratio',
                              template='plotly_white')
    
    # Sector Allocation
    sector_weights = {}
    for i, ticker in enumerate(available_tickers):
        sector = get_sector_for_ticker(ticker)
        if sector in sector_weights:
            sector_weights[sector] += weights[i] * 100
        else:
            sector_weights[sector] = weights[i] * 100
    
    sector_df = pd.DataFrame(list(sector_weights.items()), columns=['Sector', 'Weight (%)'])
    sector_df = sector_df.sort_values('Weight (%)', ascending=False)
    
    fig_sector = px.pie(
        sector_df, 
        values='Weight (%)', 
        names='Sector',
        title='Portfolio Sector Allocation',
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.3
    )
    fig_sector.update_layout(template='plotly_white')
    
    # Summary
    summary = html.Div([
        html.H4("📈 Portfolio Summary", style={'margin-bottom': '10px'}),
        html.Div([html.Span("📊 Expected Annual Return: ", style={'font-weight': 'bold'}), html.Span(f"{ret*100:.2f}%", style={'color': '#27ae60', 'font-weight': 'bold'})]),
        html.Div([html.Span("⚠️ Expected Annual Risk: ", style={'font-weight': 'bold'}), html.Span(f"{risk:.2f}%", style={'color': '#e74c3c', 'font-weight': 'bold'})]),
        html.Div([html.Span("⭐ Sharpe Ratio: ", style={'font-weight': 'bold'}), html.Span(f"{sharpe:.2f}", style={'color': '#3498db', 'font-weight': 'bold'})]),
        html.Hr(),
        html.H5("🏆 Top Holdings:"),
        html.Ul([html.Li(f"{row['Asset']}: {row['Weight (%)']:.1f}%") for _, row in weights_df.head(5).iterrows()])
    ])
    
    # Metrics
    cum = (1 + port_returns).cumprod()
    
    if _global_sp500 is not None:
        common_idx = port_returns.index.intersection(_global_sp500.index)
        benchmark_aligned = _global_sp500.loc[common_idx]
        metrics = PerformanceEvaluator.get_all_metrics(port_returns.loc[common_idx], cum.loc[common_idx], benchmark_aligned, rf_dec)
    else:
        metrics = PerformanceEvaluator.get_all_metrics(port_returns, cum, risk_free_rate=rf_dec)
    
    var95, cvar95 = compute_var_cvar(port_returns, 0.95)
    var99, cvar99 = compute_var_cvar(port_returns, 0.99)
    skew, kurt = compute_skewness_kurtosis(port_returns)
    
    metrics['VaR (95%)'] = f'{var95*100:.2f}%' if not np.isnan(var95) else 'N/A'
    metrics['CVaR (95%)'] = f'{cvar95*100:.2f}%' if not np.isnan(cvar95) else 'N/A'
    metrics['VaR (99%)'] = f'{var99*100:.2f}%' if not np.isnan(var99) else 'N/A'
    metrics['Skewness'] = f'{skew:.2f}' if not np.isnan(skew) else 'N/A'
    metrics['Kurtosis'] = f'{kurt:.2f}' if not np.isnan(kurt) else 'N/A'
    
    metrics_display = []
    metric_colors = {
        'Annualized Return': '#27ae60', 'Annualized Volatility': '#e74c3c',
        'Sharpe Ratio': '#3498db', 'Sortino Ratio': '#f39c12',
        'Maximum Drawdown': '#e67e22', 'Calmar Ratio': '#1abc9c',
        'Information Ratio': '#9b59b6', 'VaR (95%)': '#e67e22',
        'CVaR (95%)': '#e67e22', 'VaR (99%)': '#e74c3c',
        'Skewness': '#7f8c8d', 'Kurtosis': '#7f8c8d'
    }
    
    for key, val in metrics.items():
        if key in metric_colors and val != 'N/A':
            metrics_display.append(html.Div([
                html.P(key, style={'font-size': '11px', 'color': '#7f8c8d', 'margin': '0'}),
                html.P(val, style={'font-size': '18px', 'font-weight': 'bold', 'margin': '0', 'color': metric_colors[key]})
            ], style={'background': '#f8f9fa', 'padding': '12px', 'border-radius': '8px', 'min-width': '110px', 'text-align': 'center'}))
    
    return fig_front, fig_weights, fig_rolling, fig_sector, summary, metrics_display

# ============================================================
# Tab Content Callback
# ============================================================

@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value'),
     Input('stock-selector', 'value'),
     Input('goal-selector', 'value'),
     Input('rf-slider', 'value'),
     Input('benchmark-selector', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_tabs(tab, selected_tickers, goal, rf, benchmark_symbol, start_date, end_date):
    global _global_returns, _global_sp500
    
    if _global_returns is None:
        return html.Div("Loading data... Please wait")
    
    if tab == 'markowitz':
        return html.Div([
            html.H4("📊 Markowitz Portfolio Analysis", style={'color': '#2c3e50'}),
            html.P("Results shown in the charts above.", style={'color': '#27ae60'}),
            html.P(f"Analysis Period: {start_date} to {end_date}", style={'color': '#7f8c8d'})
        ])
    
    elif tab == 'active':
        if not selected_tickers or len(selected_tickers) < 2:
            return html.Div("Select at least 2 stocks")
        
        rets = _global_returns[selected_tickers].dropna()
        rf_dec = rf / 100
        
        if _global_sp500 is None:
            return html.Div("Benchmark not available")
        
        common_idx = rets.index.intersection(_global_sp500.index)
        rets_aligned = rets.loc[common_idx]
        market_aligned = _global_sp500.loc[common_idx]
        
        tb = TreynorBlackOptimizer(rets_aligned, market_aligned, risk_free_rate=rf_dec)
        all_stats = []
        for ticker in selected_tickers:
            try:
                stats = tb.compute_alpha_beta(ticker)
                alpha_val = stats['alpha'] * 252 * 100
                all_stats.append({
                    'Ticker': ticker,
                    'Alpha (Annual)': f"{alpha_val:.2f}%",
                    'Beta': f"{stats['beta']:.2f}",
                    'p-value': f"{stats['p_value']:.3f}",
                    'Significant': '✅ Yes' if stats['p_value'] < 0.10 else '❌ No'
                })
            except:
                continue
        
        if not all_stats:
            return html.Div("Could not compute alphas")
        
        table = dash_table.DataTable(
            data=all_stats,
            columns=[{'name': c, 'id': c} for c in all_stats[0].keys()],
            style_cell={'textAlign': 'center'},
            style_header={'backgroundColor': '#2c3e50', 'color': 'white'},
            style_data_conditional=[
                {'if': {'filter_query': '{Alpha (Annual)} contains "-"'}, 'backgroundColor': '#f8d7da', 'color': '#721c24'},
                {'if': {'filter_query': '{Alpha (Annual)} > 0'}, 'backgroundColor': '#d4edda', 'color': '#155724'}
            ]
        )
        
        return html.Div([
            html.H3("📈 Treynor-Black Active Portfolio", style={'margin-bottom': '20px'}),
            html.P(f"Analysis Period: {start_date} to {end_date}", style={'color': '#7f8c8d'}),
            html.P("Stocks with significant alpha (p-value < 0.10) indicate potential mispricing."),
            html.H4("🔍 Alpha Statistics:", style={'margin-top': '20px'}),
            table
        ])
    
    elif tab == 'value':
        if not selected_tickers:
            return html.Div("Select stocks to analyze")
        
        stocks_data = []
        for ticker in selected_tickers[:20]:
            metrics = get_valuation_metrics(ticker)
            rets = _global_returns[ticker].dropna()
            annual_return = (1 + rets.mean()) ** 252 - 1
            annual_vol = rets.std() * np.sqrt(252)
            
            score = calculate_value_score(
                metrics['pe'], metrics['pb'], 
                annual_return, annual_vol, 
                metrics['dividend']
            )
            rec = get_recommendation(score)
            
            stocks_data.append({
                'Ticker': ticker,
                'P/E': f"{metrics['pe']:.1f}" if metrics['pe'] else 'N/A',
                'P/B': f"{metrics['pb']:.1f}" if metrics['pb'] else 'N/A',
                'Div Yield': f"{metrics['dividend']*100:.1f}%" if metrics['dividend'] else 'N/A',
                'Ann Return': f"{annual_return*100:.1f}%",
                'Value Score': score,
                'Recommendation': f"{rec['emoji']} {rec['text']}",
                'Sector': metrics['sector']
            })
        
        stocks_data.sort(key=lambda x: x['Value Score'], reverse=True)
        
        table = dash_table.DataTable(
            data=stocks_data,
            columns=[{'name': c, 'id': c} for c in stocks_data[0].keys()],
            style_cell={'textAlign': 'center'},
            style_header={'backgroundColor': '#2c3e50', 'color': 'white'},
            style_data_conditional=[
                {'if': {'filter_query': '{Value Score} >= 7'}, 'backgroundColor': '#d4edda', 'color': '#155724'},
                {'if': {'filter_query': '{Value Score} >= 5 && {Value Score} < 7'}, 'backgroundColor': '#fff3cd', 'color': '#856404'},
                {'if': {'filter_query': '{Value Score} < 5'}, 'backgroundColor': '#f8d7da', 'color': '#721c24'}
            ],
            page_size=15
        )
        
        return html.Div([
            html.H3("💰 Value Stock Screener", style={'margin-bottom': '20px'}),
            html.P(f"Analysis Period: {start_date} to {end_date}", style={'color': '#7f8c8d'}),
            html.Div([
                html.H4("📊 Scoring System:"),
                html.Ul([
                    html.Li("🟢 Strong Buy (7+): Excellent value, strong fundamentals"),
                    html.Li("🟡 Buy (5-6): Good value, positive momentum"),
                    html.Li("🟠 Hold (3-4): Fairly valued, monitor"),
                    html.Li("🔴 Sell (0-2): Overvalued or weak fundamentals")
                ])
            ], style={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '10px', 'margin-bottom': '20px'}),
            html.H4("Stock Rankings:", style={'margin-top': '20px'}),
            table
        ])
    
    elif tab == 'monte':
        if not selected_tickers or len(selected_tickers) < 2:
            return html.Div("Select at least 2 stocks")
        
        rets = _global_returns[selected_tickers].dropna()
        rf_dec = rf / 100
        optimizer = PortfolioOptimizer(rets, risk_free_rate=rf_dec)
        
        if goal == 'sharpe':
            weights = optimizer.optimize_max_sharpe()
        elif goal == 'min_vol':
            weights = optimizer.optimize_min_volatility()
        elif goal in TARGET_RETURNS:
            weights = optimizer.optimize_target_return(TARGET_RETURNS[goal])
        else:
            weights = np.array([1/len(selected_tickers)] * len(selected_tickers))
        
        if weights is None:
            weights = np.array([1/len(selected_tickers)] * len(selected_tickers))
        
        n_sim = 1000
        n_days = 252
        mean = rets.mean()
        cov = rets.cov()
        
        np.random.seed(42)
        sim_returns = np.random.multivariate_normal(mean, cov, (n_days, n_sim))
        port_returns_sim = sim_returns @ weights
        cumulative_wealth = (1 + port_returns_sim).cumprod(axis=0)
        final_wealth = cumulative_wealth[-1, :]
        
        mean_final = np.mean(final_wealth)
        median_final = np.median(final_wealth)
        var95 = np.percentile(final_wealth, 5)
        var99 = np.percentile(final_wealth, 1)
        prob_loss = (final_wealth < 1).mean() * 100
        prob_gain_20 = (final_wealth > 1.2).mean() * 100
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=final_wealth, nbinsx=50, marker_color='#3498db', name='Simulated Outcomes'))
        fig_hist.add_vline(x=mean_final, line_dash="dash", line_color="red", annotation_text=f"Mean: ${mean_final:.2f}")
        fig_hist.add_vline(x=median_final, line_dash="dash", line_color="green", annotation_text=f"Median: ${median_final:.2f}")
        fig_hist.add_vline(x=1, line_dash="dot", line_color="gray", annotation_text="Initial Investment")
        fig_hist.update_layout(title='Distribution of Final Portfolio Value ($1 Initial Investment)',
                               xaxis_title='Final Value', yaxis_title='Frequency', template='plotly_white', bargap=0.05)
        
        return html.Div([
            html.H3("🔄 Monte Carlo Simulation", style={'margin-bottom': '20px'}),
            html.P(f"Analysis Period: {start_date} to {end_date}", style={'color': '#7f8c8d'}),
            html.P(f"Simulating {n_sim:,} scenarios over 1 year (252 trading days)"),
            html.Div([
                html.H4("📊 Simulation Statistics", style={'margin-bottom': '15px'}),
                html.Div([
                    html.Div([html.P("Mean Final Value", style={'font-size': '12px', 'color': '#7f8c8d'}), html.P(f"${mean_final:.2f}", style={'font-size': '24px', 'font-weight': 'bold', 'color': '#27ae60'})], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'background': '#f8f9fa', 'border-radius': '10px', 'min-width': '150px'}),
                    html.Div([html.P("Median Final Value", style={'font-size': '12px', 'color': '#7f8c8d'}), html.P(f"${median_final:.2f}", style={'font-size': '24px', 'font-weight': 'bold', 'color': '#3498db'})], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'background': '#f8f9fa', 'border-radius': '10px', 'min-width': '150px'}),
                    html.Div([html.P("VaR (95%)", style={'font-size': '12px', 'color': '#7f8c8d'}), html.P(f"${var95:.2f}", style={'font-size': '24px', 'font-weight': 'bold', 'color': '#e67e22'})], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'background': '#f8f9fa', 'border-radius': '10px', 'min-width': '150px'}),
                    html.Div([html.P("VaR (99%)", style={'font-size': '12px', 'color': '#7f8c8d'}), html.P(f"${var99:.2f}", style={'font-size': '24px', 'font-weight': 'bold', 'color': '#e74c3c'})], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'background': '#f8f9fa', 'border-radius': '10px', 'min-width': '150px'}),
                    html.Div([html.P("Probability of Loss", style={'font-size': '12px', 'color': '#7f8c8d'}), html.P(f"{prob_loss:.1f}%", style={'font-size': '24px', 'font-weight': 'bold', 'color': '#e74c3c'})], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'background': '#f8f9fa', 'border-radius': '10px', 'min-width': '150px'}),
                    html.Div([html.P("Probability of 20%+ Gain", style={'font-size': '12px', 'color': '#7f8c8d'}), html.P(f"{prob_gain_20:.1f}%", style={'font-size': '24px', 'font-weight': 'bold', 'color': '#27ae60'})], style={'display': 'inline-block', 'margin': '10px', 'padding': '15px', 'background': '#f8f9fa', 'border-radius': '10px', 'min-width': '150px'})
                ], style={'text-align': 'center'})
            ]),
            dcc.Graph(figure=fig_hist)
        ])
    
    elif tab == 'compare':
        if not selected_tickers or len(selected_tickers) < 2:
            return html.Div("Select at least 2 stocks")
        
        rets = _global_returns[selected_tickers].dropna()
        rf_dec = rf / 100
        optimizer = PortfolioOptimizer(rets, risk_free_rate=rf_dec)
        
        w1 = optimizer.optimize_max_sharpe()
        if w1 is not None:
            r1, risk1, s1 = optimizer.portfolio_performance(w1)
            port_returns1 = rets.dot(w1)
            var95_1, cvar95_1 = compute_var_cvar(port_returns1, 0.95)
        else:
            r1, risk1, s1, var95_1, cvar95_1 = 0, 0, 0, 0, 0
        
        w2 = np.array([1/len(selected_tickers)] * len(selected_tickers))
        r2, risk2, s2 = optimizer.portfolio_performance(w2)
        port_returns2 = rets.dot(w2)
        var95_2, cvar95_2 = compute_var_cvar(port_returns2, 0.95)
        
        w3 = optimizer.optimize_min_volatility()
        if w3 is not None:
            r3, risk3, s3 = optimizer.portfolio_performance(w3)
            port_returns3 = rets.dot(w3)
            var95_3, cvar95_3 = compute_var_cvar(port_returns3, 0.95)
        else:
            r3, risk3, s3, var95_3, cvar95_3 = 0, 0, 0, 0, 0
        
        table_data = [
            {'Strategy': 'Max Sharpe', 'Return': f"{r1*100:.1f}%", 'Risk': f"{risk1:.1f}%", 'Sharpe': f"{s1:.2f}", 'VaR (95%)': f"{var95_1*100:.1f}%", 'CVaR (95%)': f"{cvar95_1*100:.1f}%"},
            {'Strategy': 'Equal Weight', 'Return': f"{r2*100:.1f}%", 'Risk': f"{risk2:.1f}%", 'Sharpe': f"{s2:.2f}", 'VaR (95%)': f"{var95_2*100:.1f}%", 'CVaR (95%)': f"{cvar95_2*100:.1f}%"},
            {'Strategy': 'Min Volatility', 'Return': f"{r3*100:.1f}%", 'Risk': f"{risk3:.1f}%", 'Sharpe': f"{s3:.2f}", 'VaR (95%)': f"{var95_3*100:.1f}%", 'CVaR (95%)': f"{cvar95_3*100:.1f}%"}
        ]
        
        table = dash_table.DataTable(
            data=table_data,
            columns=[{'name': c, 'id': c} for c in table_data[0].keys()],
            style_cell={'textAlign': 'center', 'padding': '12px'},
            style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'}
        )
        
        return html.Div([
            html.H3("⚖️ Portfolio Strategy Comparison", style={'margin-bottom': '20px'}),
            html.P(f"Analysis Period: {start_date} to {end_date}", style={'color': '#7f8c8d'}),
            html.H4("Performance & Risk Metrics", style={'margin-top': '20px'}),
            table
        ])
    
    return html.Div()

# ============================================================
# Run the app
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(host='0.0.0.0', port=port, debug=False)
