# core/data_fetcher.py
"""
Fetch stock data from Yahoo Finance with fallback to sample data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class PortfolioDataFetcher:
    """Fetch stock data from Yahoo Finance"""
    
    SECTORS = {
        'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META'],
        'Financial': ['JPM', 'V', 'MA', 'BAC', 'GS'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV'],
        'Consumer': ['WMT', 'HD', 'COST', 'MCD', 'PG'],
        'Industrial/Energy': ['XOM', 'CVX', 'CAT', 'BA', 'GE']
    }
    
    @classmethod
    def get_all_tickers(cls):
        all_tickers = []
        for sector, tickers in cls.SECTORS.items():
            all_tickers.extend(tickers)
        return all_tickers
    
    @classmethod
    def get_sector_groups(cls):
        return cls.SECTORS
    
    @classmethod
    def create_sample_data(cls, tickers, start_date, end_date):
        """Create sample data as fallback when Yahoo Finance fails"""
        print("   📊 Generating sample data for demonstration...")
        dates = pd.date_range(start_date, end_date, freq='D')
        np.random.seed(42)
        
        sample_data = {}
        for ticker in tickers[:10]:
            # Generate realistic price movements
            returns = np.random.randn(len(dates)) * 0.02
            price = 100 * np.exp(np.cumsum(returns))
            sample_data[ticker] = price
        
        return pd.DataFrame(sample_data, index=dates)
    
    @classmethod
    def fetch_data(cls, tickers=None, start_date=None, end_date=None):
        """Fetch historical data from Yahoo Finance with fallback"""
        
        if tickers is None:
            tickers = cls.get_all_tickers()
        
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=2*365)
        
        print(f"\n📊 Fetching data for {len(tickers[:10])} stocks...")
        print(f"   From: {start_date.strftime('%Y-%m-%d')}")
        print(f"   To:   {end_date.strftime('%Y-%m-%d')}")
        print("-" * 60)
        
        try:
            # Try yfinance with a smaller set of stocks
            tickers_to_fetch = tickers[:10]  # Limit to 10 for performance
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
                return cls.create_sample_data(tickers[:10], start_date, end_date)
            
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
            return cls.create_sample_data(tickers[:10], start_date, end_date)
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
            print("   Using sample data for demonstration...")
            return cls.create_sample_data(tickers[:10], start_date, end_date)
    
    @classmethod
    def calculate_returns(cls, prices):
        """Calculate daily returns"""
        if prices.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        daily_returns = prices.pct_change().dropna()
        monthly_returns = prices.resample('M').last().pct_change().dropna()
        
        return daily_returns, monthly_returns
    
    @classmethod
    def get_risk_free_rate(cls):
        return 0.03
