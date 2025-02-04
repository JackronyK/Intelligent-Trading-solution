import pandas as pd
import numpy as np
import warnings
import logging
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from arch import arch_model
from  statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from data_retrieval import DataRetrieval
from typing import Dict, Any, Optional

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

    def compute_returns(self):
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

        self.compute_returns()
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
    
#Configure logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolatilityModeler:
    """
    Autonomous end-to-end volatility modeling pipeline that:
    1. Automatically selects models based on TS attributes
    2. Tunes parameters using grid search
    3. Selects best model by AIC/BIC
    4. Stores forecasts and metrics
    """
    def __init__(self, ticker: str):
        """
        Initialize the modeler with a ticker symbol.
        
        Args:
            ticker (str): The ticker symbol for the stock.
        """
        self.ticker = ticker
        self.processor = TimeSeriesPreprocessor(ticker=ticker)
        self._validate_initialization()

        # Auto-detect model candidates
        self.suggested_models = self.processor.model_selector()['name']
        self.default_param_grids = {
            'ARCH': {'p': [1, 2, 3]},
            'GARCH': {'p': [1, 2], 'q': [1, 2]},
            'EGARCH': {'p': [1, 2], 'q': [1], 'o': [1]},
            'ARIMA': {'order': [(1,0,1), (1,1,1)]},
            'SARIMA': {'order': [(1,1,1)], 'seasonal_order': [(0,1,1,12)]}

        }

        # Result storage
        self.trained_models = {}
        self.best_model = None
        self.forecast = {}

        """        # State tracking
        self.is_trained = False
        self.models: Dict[str, any] = {
            'initial_model': None,
            'tuned_model': None,
            'best_params': None,
            'Metrics': {}
        } 
        """
    
    def _validate_initialization(self):
        """ Ensure prerequisite data exists and is valid"""
        try:
            self.processor.compute_returns()
            self.returns_df = self.processor.returns
            if self.returns_df.empty:
                raise ValueError("Empty returns dataframe")
        except AttributeError as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise
    
    def _auto_fit_tune(self):
        """
        Core automation pipeline
        """
        for model_type in  self.suggested_models:
            try:
                # Fit base model
                base_model = self._fit_base_model(model_type)
                if not base_model:
                    continue

                # Tune parameters
                tuned_model = self._tune_model(
                    model_type,
                    self.default_param_grids.get(model_type, {})
                )

                # Store results
                self.trained_models[model_type] = {
                    'base': base_model,
                    'tuned': tuned_model,
                    'metrics': {
                        'aic': tuned_model.aic,
                        'bic': tuned_model.bic
                    }
                }
            except Exception as e:
                logger.warning(f"Failed processing {model_type}: {str(e)}")
                continue
            # Select best model
        self._select_best_model()

    def _fit_base_model(self, model_type: str):
        """
        initial model fitting with error handling
        """
        try:
            config = self._get_model_config(model_type)
            model = config['constructor'](self.returns_df, **config['params'])
            return model.fit(disp=False)
        except Exception as e:
            logger.error(f"Failed fitting {model_type}: {str(e)}")
            return None
    def _tune_model(self, model_type: str, param_grid: dict):
        """ Automated parameter tuning"""
        best_aic = float('inf')
        best_model = None

        for params in self._param_generator(param_grid):
            try:
                model = self._get_model_config(model_type)['constructor'](self.returns_df, **params)
                fitted = model.fit(disp=False)

                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_model = fitted
            except:
                continue
        return best_model
    
    def _select_best_model(self):
        """ select the best model by AIC"""
        if not self.trained_models:
            raise ValueError("No Models trained successfully")
        
        self.best_model = min(
            self.trained_models.items(),
            key=lambda x: x[1]['metrics']['aic']
        )
        logger.info(f"Selected best model: {self.best_model[0]}")

    def run_pipeline(self, forecast_horizon: int = 5):
        """Execute full autonomous pipeline"""
        self._auto_fit_tune()

        # Generate forecast
        if self.best_model:
            forecast = self.best_model[1]['tuned'].forecast(horizon=forecast_horizon)
            self.forecasts = {
                'mean': forecast.mean,
                'variance': forecast.variance,
                'interval': forecast.conf_int()
            }
        return {
            'best_model': self.best_model[0],
            'parameters': self.best_model[1]['tuned'].params,
            'forecast': self.forecasts,
            'all_models': self.trained_models
        }   


    def _get_model_config(self, model_type: str) -> Dict:
        """
        Centralized model configuration for supported models.
        
        Args:
            model_type (str): The type of model (e.g., 'ARCH', 'GARCH', 'EGARCH', 'ARIMA', 'SARIMA').
        
        Returns:
            Dict: Configuration for the specified model type.
        """
        configs = {
            'ARCH': {
                'constructor': arch_model,
                'params': {
                    'vol': "ARCH",
                    'p': self.processor.models_feat['params']['ARCH']['p']
                }},
            'GARCH': {
                'constructor': arch_model,
                'params': {
                    'vol': "GARCH",
                    'p': self.processor.models_feat['params']['GARCH']['p'],
                    'q': self.processor.models_feat['params']['GARCH']['q']
                }
            },
            'EGARCH': {
                'constructor': arch_model,
                'params': {
                    'vol': "EGARCH",
                    'p': self.processor.models_feat['params']['EGARCH']['p'],
                    'q': self.processor.models_feat['params']['EGARCH']['q']
                }
            },           
            'ARIMA': {
                'constructor': ARIMA,
                'params': {
                    'order': self.processor.models_feat['params']['ARIMA']['order']
                }
            },
            'SARIMA': {
                'constructor': SARIMAX,
                'params': {
                    'order': self.processor.models_feat['params']['SARIMA']['order'],
                    'seasonal_order': self.processor.models_feat['params']['SARIMA']['seasonal_order']
                }
            }
        }
        return configs.get(model_type, {})
    # Helper methods
    @staticmethod

    def _param_generator(grid: Dict):
        """
        Generate parameter combinations for grid search.
        
        Args:
            grid (Dict): A dictionary of parameter ranges.
        
        Yields:
            Dict: A dictionary of parameter combinations.
        """
        from itertools import product
        keys, values = zip(*grid.items())
        for v in product(*values):
            yield dict(zip(keys, v))

    def __repr__(self):
        return f"<VolatilityModeler(ticker={self.ticker}, trained={self.is_trained})>"

        



class modelling:
    def __init__(self, ticker):
        self.prossesor = TimeSeriesPreprocessor(ticker=ticker)
        self.prossesor.compute_returns()
        self.returns_df = self.returns

    def model_fitting(self):
        self.model_res = {
            "name": None,
            "init_trained_model":None,
            "tuned_model": None
        }
        self.prossesor.model_selector()
        if "ARCH" in self.model_feat['name']:
            p = self.models_feat["params"]["ARCH"]['p']
            model = arch_model(self.returns_df, vol='ARCH', p=p)
            model.fit(disp=False)

            # Appending result to the res dict
            self.model_res['name'] = 'arch'
            self.model_res['init_trained_model'] = model
        elif "GARCH" in self.model_feat['name']:
            p = self.models_feat["params"]["GARCH"]['p']
            q = self.models_feat["params"]["GARCH"]['q']
            model = arch_model(self.returns_df, vol='GARCH', p=p, q=q)
            model.fit()

            # Appending result to the res dict
            self.model_res['name'] = 'garch'
            self.model_res['init_trained_model'] = model
        elif "EGARCH" in self.model_feat['name']:
            p = self.models_feat["params"]["EGARCH"]['p']
            q = self.models_feat["params"]["EGARCH"]['q']
            model = arch_model(self.returns_df, vol='EGARCH', p=p, q=q)
            model.fit()

            # Appending result to the res dict
            self.model_res['name'] = 'egarch'
            self.model_res['init_trained_model'] = model
        elif "ARIMA" in self.model_feat['name']:
            order = self.models_feat["params"]["ARIMA"]['order']
            model = ARIMA(self.returns_df, order=order)
            model.fit()

            # Appending result to the res dict
            self.model_res['name'] = 'arima'
            self.model_res['init_trained_model'] = model
        elif "SARIMA" in self.model_feat['name']:
            order = self.models_feat["params"]["SARIMA"]['order']
            seasonal_order = self.models_feat["params"]["SARIMA"]['seasonal_order']

            model = SARIMAX(self.returns_df, order=order, seasonal_order=seasonal_order)
            model.fit()

            # Appending result to the res dict
            self.model_res['name'] = 'sarima'
            self.model_res['init_trained_model'] = model
        return self.model_res

        
    def _arch_tuner(self, model_res):
        best_aic = float('inf')    
        for p in range(1, 5):
           # try:
                model = arch_model(self.returns_df, vol='ARCH', p=p)
                res = model.fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    #best_model = 



    def tuning_params(self):
        model_res = self.model_fitting()


    def fit_model(self, model_type: str) -> Optional[Dict]:
        """
        Fit the specified model type to the returns data.
        
        Args:
            model_type (str): The type of model to fit.
        
        Returns:
            Optional[Dict]: A dictionary containing the fitted model and metrics, or None if fitting fails.
        """
        try:
            config = self._get_model_config(model_type)
            if not config:
                raise ValueError(f"Unsupported model type: {model_type}")
            model = config['constructor'](self.returns_df, **config['params'])
            fitted_model = model.fit(disp=False)

            self.models.update(
                {
                    'initial_model': fitted_model,
                    'model_type': model_type,
                    'metrics': {
                        'aic': fitted_model.aic,
                        'bic': fitted_model.bic
                    }
                }
            )
            self.is_trained = True
            return self.models

        except Exception as e:
            logger.error(f"Failed fitting {model_type}: {str(e)}")
            self.is_trained = False
            return None

    def tune_model(self, param_grid: Dict[str, Any], metric: str = 'aic')->Dict:
        """
        Tune the model using grid search over the specified parameter grid.
        
        Args:
            param_grid (Dict[str, Any]): A dictionary of parameters to search over.
            metric (str): The metric to optimize ('aic' or 'bic').
        
        Returns:
            Dict: A dictionary containing the best model, parameters, and metrics.
        """
        if not self.is_trained:
            raise RuntimeError("Fit initial model before tuning")
        best_metric = float('inf')
        best_params = {}

        for params in self._param_generator(param_grid):
            try:
                current_model = self.models['initial_model'].model.clone(self.returns_df, **params)
                results = current_model.fit(disp=False)

                if results.info_criteria[metric] < best_metric:
                    best_metric = results.info_criteria['metric']
                    best_params = params
                    self.models['tuned_model'] = results
            except Exception as e:
                logger.warning(f"Skipping params {params}: {str(e)}")
                continue

        self.models.update({
            'best_params': best_params,
            'metric': {
                metric: best_metric,
                'best_model': self.models['tuned_model'].summary()
            }
        })
        
        return self.models
    
    def forecast(self, horizon: int=5) -> Dict:
        """
        Generate forecasts using the tuned model.
        
        Args:
            horizon (int): The number of periods to forecast.
        
        Returns:
            Dict: A dictionary containing the forecasted values and confidence intervals.
        """
        if not self.models['tuned_model']:
            raise RuntimeError("No tuned model available for forecasting")
        
        try:
            forecast = self.models['tuned_model'].forecast(horizon=horizon)

            return {
                'ticker': self.ticker,
                'forecast': forecast.mean.values,
                'confidence_intervals': forecast.conf_int().values
            }
        except Exception as e:
            logger.error(f"Forecast failed: {str(e)}")
            return {'error': str(e)}
        
        









    
