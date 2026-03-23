# core/data_fetcher.py - Simplified version
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

class PortfolioDataFetcher:
    """Fetch stock data from Alpha Vantage API"""
    
    API_KEY = "ALPHA_VANTAGE_API_KEY"
    
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
    def fetch_single_stock(cls, ticker):
        """Fetch a single stock from Alpha Vantage"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': ticker,
                'apikey': cls.API_KEY,
                'outputsize': 'compact'  # 'compact' returns last 100 days
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                daily_data = data['Time Series (Daily)']
                df = pd.DataFrame.from_dict(daily_data, orient='index')
                df = df.astype(float)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                return df['4. close']
            else:
                print(f"   {ticker}: No data - {data.get('Note', 'Unknown error')[:50]}")
                return None
                
        except Exception as e:
            print(f"   {ticker}: Error - {str(e)[:50]}")
            return None
    
    @classmethod
    def fetch_data(cls, tickers=None, start_date=None, end_date=None):
        """Fetch historical data for multiple tickers"""
        
        if tickers is None:
            tickers = cls.get_all_tickers()
        
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=2*365)
        
        print(f"\n📊 Fetching data from Alpha Vantage")
        print(f"   From: {start_date.strftime('%Y-%m-%d')}")
        print(f"   To:   {end_date.strftime('%Y-%m-%d')}")
        print("-" * 60)
        
        max_stocks = min(len(tickers), 5)  # Start with 5 for testing
        print(f"   Fetching {max_stocks} stocks")
        
        data = {}
        successful = 0
        
        for i, ticker in enumerate(tickers[:max_stocks]):
            print(f"  [{i+1}/{max_stocks}] {ticker:6s}:", end=" ")
            
            prices = cls.fetch_single_stock(ticker)
            
            if prices is not None and not prices.empty:
                # Filter by date range
                mask = (prices.index >= pd.to_datetime(start_date)) & (prices.index <= pd.to_datetime(end_date))
                prices = prices[mask]
                
                if not prices.empty:
                    data[ticker] = prices
                    successful += 1
                    print(f"✓ ({len(prices)} days)")
                else:
                    print(f"✗ (no data in range)")
            else:
                print(f"✗")
            
            time.sleep(12)  # Rate limiting
        
        print("-" * 60)
        print(f"✅ Loaded {successful}/{max_stocks} stocks")
        
        if successful == 0:
            print("❌ No data retrieved")
            return pd.DataFrame()
        
        return pd.DataFrame(data)
    
    @classmethod
    def calculate_returns(cls, prices):
        if prices.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        daily_returns = prices.pct_change().dropna()
        monthly_returns = prices.resample('M').last().pct_change().dropna()
        
        return daily_returns, monthly_returns
    
    @classmethod
    def get_risk_free_rate(cls):
        return 0.03
