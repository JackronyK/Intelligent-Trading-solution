import pandas as pd
import requests
import yfinance as yf
import logging
from requests.exceptions import RequestException, HTTPError
from typing import Optional, Tuple, Dict
from config import settings

logging.basicConfig(level=logging.INFO)


class DataRetrieval:
    """
    A class to handle data retrieval for stock analysis, including historical prices, dividends, earnings, and sector/industry data.
    Supports proxy rotation and API key management.
    """

    def __init__(self, ticker: str):
        """
        Initializes the DataRetrieval class with the given stock ticker, API keys, and proxy list.

        Args:
            ticker (str): The stock ticker symbol.
        """
        self.ticker = ticker.strip()
        self.api_keys = settings.alpha_api_keys
        self.proxies = settings.proxy_list

        if not self.api_keys:
            raise ValueError("No AlphaVantage API keys configured.")
        if not self.proxies:
            raise ValueError("No proxies configured.")

        self.current_key_index = 0
        self.current_proxy_index = 0

    def _get_current_api_key(self) -> str:
        """Return the current API key."""
        return self.api_keys[self.current_key_index]

    def _rotate_api_key(self) -> None:
        """Rotate to the next API key."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logging.info(f"API key rotated. Using key {self.current_key_index + 1}/{len(self.api_keys)}.")

    def _get_current_proxy(self) -> Dict[str, str]:
        """Return the current proxy."""
        proxy = self.proxies[self.current_proxy_index]
        return {"http": proxy, "https": proxy}

    def _rotate_proxy(self) -> None:
        """Rotate to the next proxy."""
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
        logging.info(f"Proxy rotated. Using proxy {self.current_proxy_index + 1}/{len(self.proxies)}.")

    def _make_request(self, url: str, params: Dict[str, str]) -> dict:
        """
        Make an API request, handling key and proxy rotation.

        Args:
            url (str): The API endpoint.
            params (dict): Query parameters.

        Returns:
            dict: API response JSON.

        Raises:
            RuntimeError: If all keys and proxies fail.
        """
        for _ in range(len(self.api_keys) * len(self.proxies)):
            try:
                params["apikey"] = self._get_current_api_key()
                proxies = self._get_current_proxy()

                response = requests.get(url, params=params, proxies=proxies, timeout=10)
                response.raise_for_status()

                # Check if rate limit is exceeded
                if "Note" in response.json() and "exceeded" in response.json()["Note"].lower():
                    raise RuntimeError("API limit reached for current key.")

                return response.json()

            except RuntimeError as e:
                logging.warning(f"{e}. Rotating API key and proxy...")
                self._rotate_api_key()
                self._rotate_proxy()
            except RequestException as e:
                logging.warning(f"HTTP error: {e}. Rotating proxy...")
                self._rotate_proxy()

        raise RuntimeError("All API keys and proxies have failed.")

    def historical_prices(self) -> pd.DataFrame:
        """
        Retrieves historical daily stock prices using the AlphaVantage API.

        Returns:
            pd.DataFrame: A DataFrame containing historical prices with open, high, low, close, and volume.
        """
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": self.ticker,
            "outputsize": "full"
        }

        data = self._make_request(url, params)

        # Handling case when API limit is exceeded or invalid response
        if "Time Series (Daily)" not in data:
            raise KeyError(f"Error: Time Series (Daily) data not found in response. Check your API limits or ticker.")

        hist_data = data["Time Series (Daily)"]
        hist_df = pd.DataFrame.from_dict(hist_data, orient="index", dtype=float)
        hist_df.rename(columns=lambda col: col[3:].title(), inplace=True)
        hist_df.index = pd.to_datetime(hist_df.index)
        hist_df.index.name = "Date"

        return hist_df

    def dividends_history(self) -> pd.Series:
        """
        Retrieves dividend payment history using the yFinance library.

        Returns:
            pd.Series: A Series containing dividend payments indexed by date.
        """
        try:
            ticker_dataset = yf.Ticker(self.ticker)
            div_hist = ticker_dataset.dividends
            div_hist.index = pd.to_datetime(div_hist.index)
            div_hist.name = "Date"

            if div_hist.empty:
                logging.warning(f"No dividend data found for {self.ticker}. The company may not pay dividends.")
                return pd.Series(dtype=float)

            return div_hist

        except Exception as e:
            logging.error(f"Error retrieving dividend history for {self.ticker}: {e}")
            return pd.Series(dtype=float)

    def earnings(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retrieves quarterly and annual earnings data using AlphaVantage API.

        Returns:
            tuple: Two DataFrames containing quarterly and annual earnings, respectively.
        """
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "EARNINGS",
            "symbol": self.ticker
        }

        data = self._make_request(url, params)

        if "quarterlyEarnings" not in data or "annualEarnings" not in data:
            raise KeyError(f"Earnings data not found for {self.ticker}. Check API limits or ticker.")

        quarterly_earnings = pd.DataFrame(data["quarterlyEarnings"])
        annual_earnings = pd.DataFrame(data["annualEarnings"]).set_index("fiscalDateEnding").astype(float)

        return quarterly_earnings, annual_earnings

    def sector_industry_data(self) -> pd.DataFrame:
        """
        Fetches sector, industry, and company name for the given ticker.

        Returns:
            pd.DataFrame: A DataFrame with Company Name, Sector, and Industry.
        """
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "OVERVIEW",
            "symbol": self.ticker
        }

        data = self._make_request(url, params)

        company_name = data.get("Name", "N/A")
        sector = data.get("Sector", "N/A")
        industry = data.get("Industry", "N/A")

        details = pd.DataFrame({
            "Company Name": [company_name],
            "Sector": [sector],
            "Industry": [industry]
        })

        return details
