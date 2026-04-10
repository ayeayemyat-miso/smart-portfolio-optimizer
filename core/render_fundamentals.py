# core/render_fundamentals.py
"""
Special version for Render deployment - uses Alpha Vantage API
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class RenderFundamentalsFetcher:
    """Dedicated fetcher for Render deployment - doesn't replace your original"""
    
    API_KEY = os.environ.get("ALPHA_VANTAGE_KEY")
    BASE_URL = "https://www.alphavantage.co/query"
    
    @classmethod
    def is_available(cls):
        """Check if Alpha Vantage API is configured"""
        return cls.API_KEY is not None and cls.API_KEY != ""
    
    @classmethod
    def get_fundamentals(cls, ticker):
        """Get fundamental data from Alpha Vantage"""
        if not cls.is_available():
            return None
            
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': ticker,
                'apikey': cls.API_KEY
            }
            response = requests.get(cls.BASE_URL, params=params)
            data = response.json()
            
            if data and 'Symbol' in data:
                # Parse dividend yield
                div_yield = data.get('DividendYield', '0')
                try:
                    div_yield = float(div_yield) / 100 if div_yield else 0
                except:
                    div_yield = 0
                
                # Parse P/E
                pe = data.get('PERatio', None)
                try:
                    pe = float(pe) if pe and pe != 'None' else None
                except:
                    pe = None
                
                # Parse P/B
                pb = data.get('PriceToBookRatio', None)
                try:
                    pb = float(pb) if pb and pb != 'None' else None
                except:
                    pb = None
                
                return {
                    'pe_ratio': pe,
                    'pb_ratio': pb,
                    'dividend_yield': div_yield,
                    'sector': data.get('Sector', 'Unknown'),
                    'industry': data.get('Industry', 'Unknown'),
                    'roe': data.get('ReturnOnEquityTTM', None),
                    'market_cap': data.get('MarketCapitalization', None),
                    'name': data.get('Name', ticker)
                }
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
        
        return None
    
    @classmethod
    def get_batch_fundamentals(cls, tickers, delay=1):
        """Get fundamentals for multiple tickers with rate limiting"""
        results = {}
        print(f"\n📊 Fetching fundamentals for {len(tickers)} stocks from Alpha Vantage...")
        
        for i, ticker in enumerate(tickers, 1):
            print(f"   [{i}/{len(tickers)}] Fetching {ticker}...", end=' ')
            data = cls.get_fundamentals(ticker)
            if data:
                results[ticker] = data
                print(f"✅ (P/E: {data.get('pe_ratio', 'N/A')})")
            else:
                results[ticker] = None
                print(f"❌")
            time.sleep(delay)  # Rate limiting (free tier: 5 calls/minute)
        
        return results
