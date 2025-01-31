import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from data_retrieval import DataRetrieval

class TimeSeriesPreprocessor:
    """
    Prepares time-series data for modeling by extracting key attributes 
    such as stationarity, trend, seasonality, heteroscedasticity, etc.
    """

    def __init__(self, ticker):
        """
        Initialize the class with the stock ticker.

        Parameters:
            ticker (str): Stock ticker symbol.
        """
        self.stock = DataRetrieval(ticker=ticker)
        self.data_attributes = {}

    def _compute_returns(self):
        """
        Retrieves historical prices, calculates normal and log returns.
        """
        hist_df = self.stock.historical_prices()

        # Handle missing data
        if hist_df is None or 'Close' not in hist_df.columns:
            raise ValueError("Historical data is missing or improperly formatted.")

        hist_df.dropna(subset=['Close'], inplace=True)  # Drop rows with missing close prices

        returns_df = hist_df[['Close']].pct_change().dropna()
        returns_df.rename(columns={'Close': 'Returns'}, inplace=True)
        returns_df['Log_Returns'] = np.log(hist_df['Close']) - np.log(hist_df['Close'].shift(1))
        
        self.returns = returns_df.dropna()  # Ensure no NaNs

    def _ts_attributes(self):
        """
        Computes key time-series attributes:
        - Stationarity (ADF & KPSS)
        - Trend (KPSS & differencing)
        - Seasonality (ACF & decomposition)
        - Autocorrelation (ACF & PACF)
        - Heteroscedasticity (ARCH Test)
        - Volatility Clustering (Ljung-Box Test)
        """
        self.data_attributes = {
            "Stationarity": False,
            "Trend": False,
            "Seasonality": False,
            "Autocorrelation": False,
            "Heteroscedasticity": False,
            "VolatilityClustering": False,
            "ARCHOrder": 0
        }

        self._compute_returns()
        log_returns = self.returns['Log_Returns']

        ## --- STATIONARITY TESTS ---
        try:
            adf_res = adfuller(log_returns, autolag='AIC')
            p_val_adf = adf_res[1]
        except Exception as e:
            warnings.warn(f"ADF test failed: {e}")
            p_val_adf = 1  # Assume non-stationary if test fails

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_stats, p_val_kpss, _, _ = kpss(log_returns, regression='c', nlags='auto')
        except Exception as e:
            warnings.warn(f"KPSS test failed: {e}")
            p_val_kpss = 0  # Assume stationary if test fails

        self.data_attributes['Stationarity'] = p_val_adf < 0.05 or p_val_kpss > 0.05

        ## --- TREND CHECK ---
        self.data_attributes['Trend'] = p_val_kpss < 0.05

        ## --- SEASONALITY CHECK ---
        try:
            decomposition = seasonal_decompose(log_returns, period=12, model='additive')
            self.data_attributes['Seasonality'] = decomposition.seasonal.std() > 0.01
        except Exception:
            pass  # Fail silently if decomposition fails

        acf_values = acf(log_returns, nlags=40)
        if any(abs(acf_values) > 0.2):
            self.data_attributes['Seasonality'] = True

        ## --- AUTOCORRELATION CHECK ---
        pacf_values = pacf(log_returns, nlags=40)
        self.data_attributes['Autocorrelation'] = any(abs(pacf_values) > 0.2)

        ## --- HETEROSCEDASTICITY (ARCH TEST) ---
        try:
            arch_test = het_arch(log_returns)
            self.data_attributes['Heteroscedasticity'] = arch_test[1] < 0.05
        except Exception:
            pass

        ## --- VOLATILITY CLUSTERING (LJUNG-BOX TEST) ---
        try:
            lb_test = acorr_ljungbox(log_returns ** 2, lags=5, return_df=True)
            self.data_attributes['VolatilityClustering'] = lb_test['lb_pvalue'].iloc[-1] < 0.05
        except Exception:
            pass

        ## --- ARCH ORDER ESTIMATION ---
        significant_lags = sum(lb_test['lb_pvalue'] < 0.05) if 'lb_pvalue' in lb_test else 0
        self.data_attributes['ARCHOrder'] = min(significant_lags, 5)

    def model_selector(self):
        """
        Determines the best model(s) based on time-series attributes.
        """
        self.models_feat = {"name": [], "params": {}}
        self._ts_attributes()  # Ensure attributes are computed

        is_stationary = self.data_attributes['Stationarity']
        has_trend = self.data_attributes['Trend']
        has_seasonality = self.data_attributes['Seasonality']
        has_autocorrelation = self.data_attributes['Autocorrelation']
        has_heteroscedasticity = self.data_attributes['Heteroscedasticity']

        # Model Selection Logic
        if is_stationary:
            if not has_heteroscedasticity:
                self.models_feat["name"].append("ARIMA")
                self.models_feat["params"]["ARIMA"] = {"order": (1, 1, 1)}
            else:
                self.models_feat["name"].append("GARCH")
                self.models_feat["params"]["GARCH"] = {"p": 1, "q": 1}
        else:
            if has_trend or has_seasonality:
                self.models_feat["name"].append("SARIMA")
                self.models_feat["params"]["SARIMA"] = {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)}

            if has_heteroscedasticity:
                self.models_feat["name"].extend(["ARCH", "EGARCH"])
                self.models_feat["params"]["ARCH"] = {"p": 1}
                self.models_feat["params"]["EGARCH"] = {"p": 1, "q": 1}

        return self.models_feat
