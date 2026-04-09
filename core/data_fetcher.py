# core/data_fetcher.py
"""
Fetch stock data from Yahoo Finance with fallback to sample data
Includes fundamental metrics (P/E, P/B, dividend yield, etc.)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortfolioDataFetcher:
    """Fetch stock data from Yahoo Finance including prices and fundamentals"""
    
    SECTORS = {
        'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'ADBE', 'ORCL', 'CRM', 'AMD', 'INTC'],
        'Financial': ['JPM', 'V', 'MA', 'BAC', 'GS', 'WFC', 'C', 'AXP', 'BLK', 'SCHW'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'BMY', 'AMGN', 'GILD'],
        'Consumer': ['WMT', 'HD', 'COST', 'MCD', 'PG', 'KO', 'PEP', 'NKE', 'SBUX', 'TGT'],
        'Industrial/Energy': ['XOM', 'CVX', 'CAT', 'BA', 'GE', 'UPS', 'HON', 'LMT', 'RTX', 'COP']
    }
    
    @classmethod
    def get_all_tickers(cls):
        """Get all tickers from all sectors"""
        all_tickers = []
        for sector, tickers in cls.SECTORS.items():
            all_tickers.extend(tickers)
        return all_tickers
    
    @classmethod
    def get_sector_groups(cls):
        """Get sector groupings"""
        return cls.SECTORS
    
    @classmethod
    def fetch_fundamentals(cls, ticker):
        """
        Fetch fundamental data for a single ticker
        Returns comprehensive metrics including P/E, P/B, dividend yield, etc.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get current price
            current_price = info.get('currentPrice', info.get('regularMarketPrice', np.nan))
            
            fundamentals = {
                # Valuation Metrics
                'pe_ratio': info.get('trailingPE', np.nan),
                'forward_pe': info.get('forwardPE', np.nan),
                'pb_ratio': info.get('priceToBook', np.nan),
                'ps_ratio': info.get('priceToSalesTrailing12Months', np.nan),
                'peg_ratio': info.get('pegRatio', np.nan),
                'ev_to_ebitda': info.get('enterpriseToEbitda', np.nan),
                'ev_to_revenue': info.get('enterpriseToRevenue', np.nan),
                
                # Dividend Metrics
                'dividend_yield': info.get('dividendYield', np.nan),  # Already as decimal
                'dividend_rate': info.get('dividendRate', np.nan),
                'payout_ratio': info.get('payoutRatio', np.nan),
                
                # Profitability
                'roe': info.get('returnOnEquity', np.nan),  # Return on Equity
                'roa': info.get('returnOnAssets', np.nan),  # Return on Assets
                'profit_margin': info.get('profitMargins', np.nan),
                'operating_margin': info.get('operatingMargins', np.nan),
                'gross_margin': info.get('grossMargins', np.nan),
                
                # Growth
                'revenue_growth': info.get('revenueGrowth', np.nan),
                'earnings_growth': info.get('earningsGrowth', np.nan),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', np.nan),
                
                # Financial Health
                'current_ratio': info.get('currentRatio', np.nan),
                'debt_to_equity': info.get('debtToEquity', np.nan),
                'quick_ratio': info.get('quickRatio', np.nan),
                
                # Company Info
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', np.nan),
                'enterprise_value': info.get('enterpriseValue', np.nan),
                'shares_outstanding': info.get('sharesOutstanding', np.nan),
                'current_price': current_price,
                'target_price': info.get('targetMeanPrice', np.nan),
                
                # Analyst Recommendations
                'recommendation_key': info.get('recommendationKey', 'hold'),  # buy, hold, sell
                'recommendation_mean': info.get('recommendationMean', 3.0),  # 1=Strong Buy, 5=Strong Sell
                'number_of_analysts': info.get('numberOfAnalystOpinions', 0),
                
                # Price Targets
                'target_low': info.get('targetLowPrice', np.nan),
                'target_high': info.get('targetHighPrice', np.nan),
                'target_mean': info.get('targetMeanPrice', np.nan),
                'target_median': info.get('targetMedianPrice', np.nan),
                
                # Other Metrics
                'beta': info.get('beta', np.nan),
                '52_week_high': info.get('fiftyTwoWeekHigh', np.nan),
                '52_week_low': info.get('fiftyTwoWeekLow', np.nan),
                'volume': info.get('volume', np.nan),
                'avg_volume': info.get('averageVolume', np.nan),
            }
            
            # Convert dividend yield from decimal to percentage for display
            if not np.isnan(fundamentals['dividend_yield']):
                fundamentals['dividend_yield_pct'] = fundamentals['dividend_yield'] * 100
            else:
                fundamentals['dividend_yield_pct'] = np.nan
            
            return fundamentals
            
        except Exception as e:
            print(f"⚠️ Error fetching fundamentals for {ticker}: {e}")
            return cls._get_empty_fundamentals()
    
    @classmethod
    def _get_empty_fundamentals(cls):
        """Return empty fundamentals dict with default values"""
        return {
            'pe_ratio': np.nan, 'forward_pe': np.nan, 'pb_ratio': np.nan,
            'ps_ratio': np.nan, 'peg_ratio': np.nan, 'ev_to_ebitda': np.nan,
            'ev_to_revenue': np.nan, 'dividend_yield': np.nan, 'dividend_yield_pct': np.nan,
            'dividend_rate': np.nan, 'payout_ratio': np.nan, 'roe': np.nan,
            'roa': np.nan, 'profit_margin': np.nan, 'operating_margin': np.nan,
            'gross_margin': np.nan, 'revenue_growth': np.nan, 'earnings_growth': np.nan,
            'earnings_quarterly_growth': np.nan, 'current_ratio': np.nan,
            'debt_to_equity': np.nan, 'quick_ratio': np.nan, 'sector': 'Unknown',
            'industry': 'Unknown', 'market_cap': np.nan, 'enterprise_value': np.nan,
            'shares_outstanding': np.nan, 'current_price': np.nan, 'target_price': np.nan,
            'recommendation_key': 'hold', 'recommendation_mean': 3.0,
            'number_of_analysts': 0, 'target_low': np.nan, 'target_high': np.nan,
            'target_mean': np.nan, 'target_median': np.nan, 'beta': np.nan,
            '52_week_high': np.nan, '52_week_low': np.nan, 'volume': np.nan,
            'avg_volume': np.nan
        }
    
    @classmethod
    def fetch_all_fundamentals(cls, tickers):
        """
        Fetch fundamentals for multiple tickers
        Returns DataFrame with all fundamental data
        """
        results = {}
        total = len(tickers)
        
        print(f"\n📊 Fetching fundamental data for {total} stocks...")
        
        for i, ticker in enumerate(tickers, 1):
            print(f"   [{i}/{total}] Fetching {ticker}...", end=' ')
            fundamentals = cls.fetch_fundamentals(ticker)
            results[ticker] = fundamentals
            print(f"✅" if fundamentals['current_price'] is not np.nan else "⚠️")
        
        df = pd.DataFrame(results).T
        print(f"\n✅ Successfully fetched fundamentals for {len(df[df['current_price'].notna()])} stocks")
        return df
    
    @classmethod
    def fetch_data(cls, tickers=None, start_date=None, end_date=None):
        """Fetch historical price data from Yahoo Finance with fallback"""
        
        if tickers is None:
            tickers = cls.get_all_tickers()
        
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=2*365)
        
        # Limit to first 20 for performance
        tickers_to_fetch = tickers[:20]
        
        print(f"\n📈 Fetching price data for {len(tickers_to_fetch)} stocks...")
        print(f"   From: {start_date.strftime('%Y-%m-%d')}")
        print(f"   To:   {end_date.strftime('%Y-%m-%d')}")
        print("-" * 60)
        
        try:
            # Try yfinance
            data = yf.download(
                tickers=tickers_to_fetch,
                start=start_date,
                end=end_date,
                group_by='ticker',
                auto_adjust=True,
                progress=False,
                threads=False
            )
            
            if data.empty:
                print("⚠️ Yahoo Finance unavailable. Using sample data...")
                return cls.create_sample_data(tickers_to_fetch, start_date, end_date)
            
            # Extract close prices
            if len(tickers_to_fetch) == 1:
                if 'Close' in data.columns:
                    prices = data[['Close']].copy()
                    prices.columns = tickers_to_fetch
                else:
                    prices = data
            else:
                try:
                    if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
                        if 'Close' in data.columns.levels[1]:
                            prices = data.xs('Close', axis=1, level=1)
                        else:
                            prices = data
                    else:
                        prices = data
                except:
                    prices = data
            
            if isinstance(prices, pd.DataFrame) and not prices.empty:
                prices = prices.dropna(axis=1, how='all')
                if not prices.empty:
                    print(f"✅ Loaded {len(prices.columns)} stocks from Yahoo Finance")
                    return prices
            
            print("⚠️ No valid price data. Using sample data...")
            return cls.create_sample_data(tickers_to_fetch, start_date, end_date)
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
            print("   Using sample data for demonstration...")
            return cls.create_sample_data(tickers_to_fetch, start_date, end_date)
    
    @classmethod
    def create_sample_data(cls, tickers, start_date, end_date):
        """Create sample data as fallback when Yahoo Finance fails"""
        print("   📊 Generating sample data for demonstration...")
        dates = pd.date_range(start_date, end_date, freq='D')
        np.random.seed(42)
        
        sample_data = {}
        for ticker in tickers:
            # Generate realistic price movements
            returns = np.random.randn(len(dates)) * 0.02
            price = 100 * np.exp(np.cumsum(returns))
            sample_data[ticker] = price
        
        return pd.DataFrame(sample_data, index=dates)
    
    @classmethod
    def calculate_returns(cls, prices):
        """Calculate daily and monthly returns"""
        if prices.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        daily_returns = prices.pct_change().dropna()
        monthly_returns = prices.resample('M').last().pct_change().dropna()
        
        return daily_returns, monthly_returns
    
    @classmethod
    def calculate_annual_return(cls, prices, ticker):
        """Calculate annual return from price history"""
        try:
            if ticker in prices.columns and len(prices) > 0:
                start_price = prices[ticker].iloc[0]
                end_price = prices[ticker].iloc[-1]
                days = len(prices)
                years = days / 365.25
                if start_price > 0 and years > 0:
                    annual_return = (end_price / start_price) ** (1/years) - 1
                    return annual_return
        except:
            pass
        return np.nan
    
    @classmethod
    def calculate_value_score(cls, fundamentals):
        """
        Calculate value score (1-5, higher is better value)
        Based on P/E, P/B, and dividend yield
        """
        score = 0
        metrics_used = 0
        
        # P/E Ratio Score (lower is better for value)
        pe = fundamentals.get('pe_ratio', np.nan)
        if not np.isnan(pe) and pe > 0:
            metrics_used += 1
            if pe < 10:
                score += 2
            elif pe < 15:
                score += 1
            elif pe > 25:
                score -= 1
            elif pe > 30:
                score -= 2
        
        # P/B Ratio Score (lower is better)
        pb = fundamentals.get('pb_ratio', np.nan)
        if not np.isnan(pb) and pb > 0:
            metrics_used += 1
            if pb < 1:
                score += 2
            elif pb < 2:
                score += 1
            elif pb > 5:
                score -= 1
            elif pb > 8:
                score -= 2
        
        # Dividend Yield Score (higher is better)
        div_yield = fundamentals.get('dividend_yield', np.nan)
        if not np.isnan(div_yield) and div_yield > 0:
            metrics_used += 1
            if div_yield > 0.05:  # >5%
                score += 2
            elif div_yield > 0.03:  # >3%
                score += 1
            elif div_yield < 0.01:  # <1%
                score -= 1
        
        # Normalize to 1-5 scale
        if metrics_used == 0:
            return 3  # Neutral if no metrics available
        
        # Base score is 3, then add/subtract
        normalized = 3 + (score / 3)
        normalized = max(1, min(5, normalized))
        
        return round(normalized, 1)
    
    @classmethod
    def get_recommendation(cls, fundamentals, annual_return=None):
        """
        Generate investment recommendation based on fundamentals
        Returns: (recommendation_text, emoji, color)
        """
        pe = fundamentals.get('pe_ratio', np.nan)
        pb = fundamentals.get('pb_ratio', np.nan)
        recommendation_key = fundamentals.get('recommendation_key', 'hold')
        
        # Use analyst recommendation if available
        if recommendation_key in ['buy', 'strong_buy'] and not np.isnan(pe):
            return "🟢 Strong Buy", "Strong Buy", "#2ecc71"
        elif recommendation_key == 'sell' and not np.isnan(pe):
            return "🔴 Sell", "Sell", "#e74c3c"
        
        # Fallback to valuation-based recommendation
        if not np.isnan(pe) and not np.isnan(pb):
            if pe < 15 and pb < 3:
                return "🟢 Buy", "Buy", "#27ae60"
            elif pe < 20 and pb < 4:
                return "🟠 Hold", "Hold", "#f39c12"
            elif pe > 30 or pb > 6:
                return "🔴 Sell", "Sell", "#e74c3c"
            else:
                return "🟠 Hold", "Hold", "#f39c12"
        
        return "⚪ Hold", "Hold", "#95a5a6"
    
    @classmethod
    def get_risk_free_rate(cls):
        """Get current risk-free rate (using 10-year treasury as proxy)"""
        try:
            # Try to fetch current 10-year treasury yield
            treasury = yf.Ticker("^TNX")
            info = treasury.info
            rate = info.get('regularMarketPrice', 3.0) / 100
            return rate
        except:
            # Fallback to 3%
            return 0.03
    
    @classmethod
    def get_company_summary(cls, ticker):
        """
        Get comprehensive company summary including all metrics
        Returns a dictionary with all relevant data
        """
        # Get fundamentals
        fundamentals = cls.fetch_fundamentals(ticker)
        
        # Get price data for annual return calculation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        prices = cls.fetch_data([ticker], start_date, end_date)
        annual_return = cls.calculate_annual_return(prices, ticker)
        
        # Calculate value score
        value_score = cls.calculate_value_score(fundamentals)
        
        # Get recommendation
        recommendation_emoji, recommendation_text, recommendation_color = cls.get_recommendation(
            fundamentals, annual_return
        )
        
        # Format dividend yield as percentage
        div_yield_pct = fundamentals.get('dividend_yield_pct', np.nan)
        if np.isnan(div_yield_pct):
            div_yield_display = "N/A"
        else:
            div_yield_display = f"{div_yield_pct:.2f}%"
        
        # Format P/E ratio
        pe = fundamentals.get('pe_ratio', np.nan)
        pe_display = f"{pe:.2f}" if not np.isnan(pe) else "N/A"
        
        # Format P/B ratio
        pb = fundamentals.get('pb_ratio', np.nan)
        pb_display = f"{pb:.2f}" if not np.isnan(pb) else "N/A"
        
        # Format annual return
        annual_return_display = f"{annual_return*100:.1f}%" if not np.isnan(annual_return) else "N/A"
        
        return {
            'ticker': ticker,
            'name': fundamentals.get('industry', ticker),  # Fallback to ticker
            'sector': fundamentals.get('sector', 'Unknown'),
            'industry': fundamentals.get('industry', 'Unknown'),
            'pe_ratio': pe_display,
            'pb_ratio': pb_display,
            'dividend_yield': div_yield_display,
            'annual_return': annual_return_display,
            'value_score': value_score,
            'recommendation': f"{recommendation_emoji} {recommendation_text}",
            'recommendation_color': recommendation_color,
            'market_cap': fundamentals.get('market_cap', np.nan),
            'current_price': fundamentals.get('current_price', np.nan),
            'target_price': fundamentals.get('target_price', np.nan),
            'beta': fundamentals.get('beta', np.nan),
            'roe': fundamentals.get('roe', np.nan),
            'roa': fundamentals.get('roa', np.nan),
            'debt_to_equity': fundamentals.get('debt_to_equity', np.nan),
        }
    
    @classmethod
    def get_all_companies_summary(cls, tickers=None):
        """
        Get summary for all companies in portfolio
        Returns DataFrame with all metrics
        """
        if tickers is None:
            tickers = cls.get_all_tickers()
        
        summaries = []
        total = len(tickers)
        
        print(f"\n📊 Generating company summaries for {total} stocks...")
        print("-" * 60)
        
        for i, ticker in enumerate(tickers[:20], 1):  # Limit to 20 for performance
            print(f"   [{i}/{min(20, total)}] Processing {ticker}...")
            summary = cls.get_company_summary(ticker)
            summaries.append(summary)
        
        df = pd.DataFrame(summaries)
        print(f"\n✅ Successfully processed {len(df)} companies")
        
        return df


# Quick test function
def test_fetcher():
    """Test the data fetcher with a few tickers"""
    print("="*60)
    print("Testing PortfolioDataFetcher")
    print("="*60)
    
    # Test single ticker
    print("\n1. Testing single ticker (AAPL):")
    apple_data = PortfolioDataFetcher.get_company_summary('AAPL')
    for key, value in apple_data.items():
        print(f"   {key}: {value}")
    
    # Test multiple tickers
    print("\n2. Testing multiple tickers:")
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    df = PortfolioDataFetcher.get_all_companies_summary(test_tickers)
    print("\n   Results:")
    print(df[['ticker', 'pe_ratio', 'pb_ratio', 'dividend_yield', 'value_score', 'recommendation']].to_string(index=False))
    
    print("\n✅ Test complete!")


if __name__ == '__main__':
    test_fetcher()
