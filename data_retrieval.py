import pandas as pd
from  config import settings
import requests
import yfinance as yf
class DataRetrieval:
    def __init__(self, ticker):
        self.ticker = ticker
        self.apikey = settings.alpha_api_key
        
    def Historical_prices(self):
        # inintiatiating the API
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": self.ticker,
            "apikey": self.apikey,
            "outputsize": "full"
        }
        response = requests.get(url=url, params=params)
        # Data 
        data = response.json()
        hist_data = data['Time Series (Daily)']
        hist_df = pd.DataFrame.from_dict(hist_data, orient='index', dtype=float)

        # Renaming the columns
        new_cols = {col: col[3:].title() for col in hist_df.columns}
        hist_df.rename(columns=new_cols, inplace=True)  

        # Renaming indexes
        hist_df.index = pd.to_datetime(hist_df.index)
        hist_df.index.name = "Date"
        
        return hist_df
    def Dividends_hist(self):
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": self.ticker,
            "apikey": self.apikey,
            "outputsize": "full"
        }
        response = requests.get(url=url, params=params)
        ticker_dataset = yf.Ticker(self.ticker)
        div_hist = ticker_dataset.dividends
        return div_hist
    def Earnings(self):
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "EARNINGS",
            "symbol": self.ticker,
            "apikey": self.apikey,
            "outputsize": "full"
        }
        response = requests.get(url=url, params=params)
        raw_data = response.json()
        # Formatting 
        quarterly_Earnings = pd.DataFrame(raw_data['quarterlyEarnings'])
        # Annual Earning
        annual_Earnings = pd.DataFrame(raw_data['annualEarnings']).set_index("fiscalDateEnding").astype(float)
        return quarterly_Earnings, annual_Earnings
        
    def sector_industry_data(self):
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "OVERVIEW",
            "symbol": self.ticker,
            "apikey": self.apikey           
        }
        response = requests.get(url=url, params=params)

        data = response.json()
        self.sector = data['Sector']
        self.industry = data['Industry']

        # Retriving inductory and sector data 
        return self.sector, self.industry




        