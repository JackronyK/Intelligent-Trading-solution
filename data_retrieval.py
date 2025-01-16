import pandas as pd
from  config import settings
import requests
import yfinance as yf
import logging
class DataRetrieval:
    """
    A class to handle data retrieval for stock analysis, including historicl prices, dividends, earnings, and secto / Indurty data 
    """
    def __init__(self, ticker):
        """
        Initializes the DataRetrieval class with the given stock ticker.

        Args:
            ticker (str): The stock ticker symbol.

        """
        self.ticker = ticker.strip()
        self.apikey = settings.alpha_api_key
        
    def historical_prices(self):
        """
       Retrieves historical daily stock prices using Alpha Vantage API.

        Returns:
            pd.DataFrame: A DataFrame containing historical prices with open, high, low, close, and volume.
        """
        # initializing the API
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": self.ticker,
            "apikey": self.apikey,
            "outputsize": "full"
        }
        response = requests.get(url=url, params=params)

        # Error handling  for api response
        if response.status_code !=200:
            raise Exception(f"Error: Unable to fetch historical prices. Status Code: {response.status_code}")        
         
        data = response.json()

        # Handling case when API limit is exceeded or invalid response
        if "Time Series (Daily)" not in data:
            raise KeyError(f"Error: Time Series (Daily) data not found in response. Check your API limits or ticker")        
        
        hist_data = data['Time Series (Daily)']

        # Creating DataFrame and formatting columns
        hist_df = pd.DataFrame.from_dict(hist_data, orient='index', dtype=float)
        new_cols = {col: col[3:].title() for col in hist_df.columns}
        hist_df.rename(columns=new_cols, inplace=True)   
        hist_df.index = pd.to_datetime(hist_df.index)
        hist_df.index.name = "Date"

        return hist_df
    
    def dividends_history(self):
        """
        Retrieves dividend payment history using the yFinance library.

        Returns:
            pd.Series: A Series containing dividend payments indexed by date.
        """
        try:
            # Use yFinance library for dividends (Alpha Vantage API does not explicitly return dividends)
            ticker_dataset = yf.Ticker(self.ticker)
            div_hist = ticker_dataset.dividends
            #index formatting
            div_hist.index = pd.to_datetime(div_hist.index)
            div_hist.name = "Date"

            # Check if the data is empty
            if div_hist.empty:
                print(f"Warning: No dividend data found for {self.ticker}. The company may not pay dividends.")
                return pd.Series(dtype=float)  # Return an empty Series if no data is found

            return div_hist

        except Exception as e:
        # Handle any unexpected errors gracefully
            print(f"Error retrieving dividend history for {self.ticker}: {e}")
            return pd.Series(dtype=float)  # Return an empty Series in case of failure

    def earnings(self):
        """
        Retrieves quarterly and annual earnings data using Alpha Vantage API.

        Returns:
            tuple: Two DataFrames containing quarterly and annual earnings, respectively.
        """
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "EARNINGS",
            "symbol": self.ticker,
            "apikey": self.apikey,
            "outputsize": "full"
        }
        response = requests.get(url=url, params=params)

        # Error handling for API response
        if response.status_code !=200:
            raise Exception(f"Error: Unable to fetch earnings data. Status Code: {response.status_code}")
        
        raw_data = response.json()

        # Handle cases where earnings data is missing
        if 'quarterlyEarnings' not in raw_data or 'annualEarnings' not in raw_data:
            raise KeyError(f"Error: Earnings data not found for {self.ticker}. Check API limits or ticker.")
        
        # parse quarterly and annual earnings 
        quarterly_Earnings = pd.DataFrame(raw_data['quarterlyEarnings'])
        annual_Earnings = pd.DataFrame(raw_data['annualEarnings']).set_index("fiscalDateEnding").astype(float)
        return quarterly_Earnings, annual_Earnings
        
    def sector_industry_data(self):
        """
        Fetches sector, industry, and company name for the given ticker.
        
        Returns:
            pd.DataFrame: A DataFrame with the following columns:
                - Company Name
                - Sector
                - Industry
        
        Raises:
            ValueError: If the ticker or API key is invalid.
            requests.exceptions.RequestException: For HTTP-related errors.
            KeyError: If expected keys are missing from the response.
        """
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "OVERVIEW",
            "symbol": self.ticker,
            "apikey": self.apikey
        }

        try:
            response = requests.get(url=url, params=params)
            response.raise_for_status()  # Raise error for bad HTTP responses
            
            data = response.json()
            
            # Extract relevant details with default values
            company_name = data.get('Name', 'N/A')
            sector = data.get('Sector', 'N/A')
            industry = data.get('Industry', 'N/A')

            # Logging fetched data for debugging
            logging.info(f"Fetched data for ticker {self.ticker}: {data}")

            # Creating a DataFrame
            details = pd.DataFrame({
                "Company Name": [company_name],
                "Sector": [sector],
                "Industry": [industry]
            })

            return details

        except requests.exceptions.RequestException as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
            raise
        except KeyError as key_err:
            logging.error(f"Missing data in the response: {key_err}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise Exception(f"Error fetching sector/industry data for ticker '{self.ticker}': {e}")





        