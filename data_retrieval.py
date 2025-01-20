import random
import requests
import pandas as pd
import logging
import yfinance as yf
from config import settings

logging.basicConfig(level=logging.INFO)

class DataRetrieval:
    """
    A class to handle data retrieval for stock analysis, including historical prices, dividends, earnings, and sector/industry data.
    """

    def __init__(self, ticker: str):
        """
        Initializes the DataRetrieval class with the given stock ticker, API keys, and proxy list.

        Args:
            ticker (str): The stock ticker symbol.
        """
        self.ticker = ticker.strip()
        self.api_keys = settings.alpha_api_keys
        self.proxy_url = settings.proxy_url
        self.base_url = "https://www.alphavantage.co/query"
        self.timeout = 10

        # Fetch and clean proxies
        proxy_list_raw = requests.get(self.proxy_url).text.split('\n')
        self.proxies = [
            {"http": f"http://{prx}", "https": f"http://{prx}"}
            for prx in proxy_list_raw if prx.strip()
        ]
        if not self.api_keys:
            raise ValueError("No API keys found in settings.")
        if not self.proxies:
            raise ValueError("No proxies retrieved from the proxy URL.")

    def _make_request(self, params: dict) -> dict:
        """
        Attempt to make a request without a proxy first. If it fails due to rate limits, use proxies.

        Args:
            params (dict): Query parameters for the API.

        Returns:
            dict: The API response.

        Raises:
            RuntimeError: If all attempts fail.
        """
        # First, try without a proxy
        for api_key in self.api_keys:
            params["apikey"] = api_key
            logging.info("Attempting direct request without proxy...")
            try:
                response = requests.get(url=self.base_url, params=params, timeout=self.timeout)
                response.raise_for_status()
                if response.status_code == 429:
                    logging.warning("Rate limit hit. Switching to proxy.")
                else:
                    return response.json()
            except requests.exceptions.RequestException as e:
                logging.warning(f"Direct request failed: {e}")
                continue

        # Try using proxies
        for attempt in range(len(self.api_keys) * len(self.proxies)):
            params["apikey"] = random.choice(self.api_keys)
            proxy = random.choice(self.proxies)
            logging.info(f"Attempt {attempt + 1}: Using proxy {proxy} and API key.")
            try:
                response = requests.get(
                    url=self.base_url,
                    params=params,
                    proxies=proxy,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.ProxyError as e:
                logging.warning(f"ProxyError: {e}. Retrying with another proxy...")
            except requests.exceptions.SSLError as e:
                logging.warning(f"SSLError: {e}. Skipping this proxy...")
            except requests.exceptions.ReadTimeout:
                logging.warning(f"ReadTimeout: Proxy {proxy} timed out. Retrying...")
            except requests.exceptions.RequestException as e:
                logging.warning(f"General error with proxy: {e}. Retrying...")

        raise RuntimeError("All attempts to make the request failed. Check API keys and proxies.")

    def historical_prices(self) -> pd.DataFrame:
        """
        Retrieve historical daily stock prices using Alpha Vantage API.

        Returns:
            pd.DataFrame: A DataFrame containing historical prices with open, high, low, close, and volume.
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": self.ticker,
            "outputsize": "full",
        }

        data = self._make_request(params)

        if "Time Series (Daily)" not in data:
            raise KeyError(f"'Time Series (Daily)' data not found for {self.ticker}. Check your API limits.")

        hist_data = data["Time Series (Daily)"]
        hist_df = pd.DataFrame.from_dict(hist_data, orient="index", dtype=float)

        # Rename columns
        new_cols = {col: col.split(" ")[1].capitalize() for col in hist_df.columns}
        hist_df.rename(columns=new_cols, inplace=True)

        # Format index
        hist_df.index = pd.to_datetime(hist_df.index)
        hist_df.index.name = "Date"

        return hist_df

    def dividends(self) -> pd.DataFrame:
        """
        Retrieve dividend data for the stock using Yahoo Finance.

        Returns:
            pd.DataFrame: A DataFrame containing dividend payout history.
        """
        try:
            logging.info(f"Fetching dividend data for {self.ticker} from Yahoo Finance...")
            stock = yf.Ticker(self.ticker)
            dividends = stock.dividends

            if dividends.empty:
                logging.warning(f"No dividend data found for {self.ticker}.")
                return pd.DataFrame(columns=["Date", "Dividend Amount"])

            # Convert to DataFrame
            dividend_df = dividends.reset_index()
            dividend_df.columns = ["Date", "Dividend Amount"]
            return dividend_df

        except Exception as e:
            logging.error(f"Failed to fetch dividend data for {self.ticker}: {e}")
            return pd.DataFrame(columns=["Date", "Dividend Amount"])

    def earnings(self) -> pd.DataFrame:
        """
        Retrieve earnings data for the stock.

        Returns:
            pd.DataFrame: A DataFrame containing quarterly or annual earnings data.
        """
        params = {
            "function": "EARNINGS",
            "symbol": self.ticker,
        }

        data = self._make_request(params)

        if "quarterlyEarnings" not in data:
            raise KeyError(f"'quarterlyEarnings' data not found for {self.ticker}. Check your API limits.")

        return pd.DataFrame(data["quarterlyEarnings"])

    def sector_and_industry(self) -> dict:
        """
        Retrieve sector and industry data for the stock.

        Returns:
            dict: A dictionary containing sector and industry information.
        """
        params = {
            "function": "OVERVIEW",
            "symbol": self.ticker,
        }

        data = self._make_request(params)

        if not data:
            raise KeyError(f"Sector and industry data not found for {self.ticker}. Check your API limits.")

        return {
            "Sector": data.get("Sector"),
            "Industry": data.get("Industry"),
            "Company Name": data.get("Name")
        }
