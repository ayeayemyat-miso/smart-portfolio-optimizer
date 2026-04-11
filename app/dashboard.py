# core/render_fundamentals.py - FMP Version
import os
import requests
import time

class RenderFundamentalsFetcher:
    """Fetcher using FMP API - Better free tier, no key regeneration issues"""
    
    API_KEY = os.environ.get("FMP_API_KEY")
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    @classmethod
    def is_available(cls):
        """Check if API key is configured"""
        return cls.API_KEY is not None and cls.API_KEY != ""
    
    @classmethod
    def get_fundamentals(cls, ticker):
        """Get fundamental data from FMP"""
        if not cls.is_available():
            print(f"   ❌ No FMP API key for {ticker}")
            return None
        
        try:
            # Get company profile (includes P/E, P/B, sector)
            profile_url = f"{cls.BASE_URL}/profile/{ticker}?apikey={cls.API_KEY}"
            profile_resp = requests.get(profile_url, timeout=15)
            profile_data = profile_resp.json()
            
            if profile_data and len(profile_data) > 0:
                company = profile_data[0]
                
                # Get dividend yield from key metrics
                metrics_url = f"{cls.BASE_URL}/key-metrics-ttm/{ticker}?apikey={cls.API_KEY}"
                metrics_resp = requests.get(metrics_url, timeout=15)
                metrics_data = metrics_resp.json()
                
                div_yield = 0
                if metrics_data and len(metrics_data) > 0:
                    div_yield = metrics_data[0].get('dividendYield', 0) or 0
                
                return {
                    'pe_ratio': company.get('pe', None),
                    'pb_ratio': company.get('priceToBookRatio', None),
                    'dividend_yield': div_yield,
                    'sector': company.get('sector', 'Unknown'),
                    'industry': company.get('industry', 'Unknown'),
                    'roe': company.get('roe', None),
                    'market_cap': company.get('mktCap', None),
                    'name': company.get('companyName', ticker)
                }
            return None
            
        except Exception as e:
            print(f"   ❌ Error for {ticker}: {str(e)[:50]}")
            return None
    
    @classmethod
    def get_batch_fundamentals(cls, tickers, delay=0.2):
        """Get fundamentals for multiple tickers"""
        results = {}
        print(f"\n📊 Fetching fundamentals for {len(tickers)} stocks from FMP...")
        print(f"   API Key: {'✓ Configured' if cls.is_available() else '✗ Missing'}")
        print("-" * 50)
        
        for i, ticker in enumerate(tickers, 1):
            print(f"   [{i}/{len(tickers)}] {ticker}...", end=' ')
            data = cls.get_fundamentals(ticker)
            if data:
                results[ticker] = data
                pe_val = data.get('pe_ratio')
                print(f"✅ (P/E: {pe_val if pe_val else 'N/A'})")
            else:
                results[ticker] = None
                print(f"❌")
            
            # Small delay (FMP allows 300+ requests/minute on free tier)
            if i < len(tickers):
                time.sleep(delay)
        
        success_count = sum(1 for v in results.values() if v is not None)
        print("-" * 50)
        print(f"✅ Successfully fetched {success_count}/{len(tickers)} stocks")
        
        return results
