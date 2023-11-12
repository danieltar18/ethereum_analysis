import pmdarima as pm
import pandas as pd
import numpy as np

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


# arima model functions

def gridsearch_timeseriessplit_arima(dataframe, n_splits=15, is_logarithmic=False, with_seasonality=False, test_size=1):
    """
    Perform time series split and grid search for ARIMA model parameters.

    Parameters:
    - dataframe (pd.DataFrame): Time series data with a column named 'eth_close'.
    - is_logarithmic (bool, optional): If True, considers 'logclose' column; else, 'eth_close'. Default is False.
    - with_seasonality (bool, optional): If True, includes seasonality in the ARIMA model. Default is False.
    - test_size (int, optional): The size of the test set in each time series split. Default is 1.

    Returns:
    - tuple: Train predictions, actual test values, ARIMA forecast, and confidence intervals.
    """
    tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=0)
    root_mean_squared_errors = []
    mean_squared_errors = []
    mean_absolute_errors = []
    mean_absolute_percentage_errors = []

    if is_logarithmic:
        column = "logclose"
    else:
        column = "eth_close"

    for train_idx, val_idx in tss.split(dataframe):
        print(f"Progress: {train_idx[-1]}", end="\r")
        train = dataframe.iloc[train_idx]
        test = dataframe.iloc[val_idx]

        if with_seasonality:
            model = pm.auto_arima(train[column],
                                  error_action="ignore",
                                  maxiter=10,
                                  suppress_warnings=True,
                                  max_p=12,
                                  max_q=2,
                                  max_order=14,
                                  seasonal=True,
                                  stepwise=True,
                                  # n_jobs=-1,
                                  m=14)
        else:
            model = pm.auto_arima(train[column],
                                  error_action="ignore",
                                  suppress_warnings=True,
                                  maxiter=10,
                                  seasonal=False,
                                  # n_jobs=-1
                                  )

        train_pred = model.predict_in_sample(start=1, end=-1)  ## plottolásnál fontos
        fcast, confint = model.predict(n_periods=len(test), return_conf_int=True)  ## confint plottolásnál fontos

        if is_logarithmic:
            rmse = mean_squared_error(test['eth_close'], np.exp(fcast), squared=False)
            root_mean_squared_errors.append(rmse)
            mse = mean_squared_error(test['eth_close'], np.exp(fcast), squared=True)
            mean_squared_errors.append(mse)
            mae = mean_absolute_error(test['eth_close'], np.exp(fcast))
            mean_absolute_errors.append(mae)
            mape = mean_absolute_percentage_error(test['eth_close'], np.exp(fcast))
            mean_absolute_percentage_errors.append(mape)
        else:
            rmse = mean_squared_error(test['eth_close'], fcast, squared=False)
            root_mean_squared_errors.append(rmse)
            mse = mean_squared_error(test['eth_close'], fcast, squared=True)
            mean_squared_errors.append(mse)
            mae = mean_absolute_error(test['eth_close'], fcast)
            mean_absolute_errors.append(mae)
            mape = mean_absolute_percentage_error(test['eth_close'], fcast)
            mean_absolute_percentage_errors.append(mape)

    # plt.show()
    print(model.summary())
    print(f"Model tested for {n_splits} splits, test size: {test_size}")
    print(f"RMSE: {round(np.mean(root_mean_squared_errors), 4)}")
    print(f"MSE: {round(np.mean(mean_squared_errors), 4)}")
    print(f"MAE: {round(np.mean(mean_absolute_errors), 4)}")
    print(f"MAPE: {str(round(np.mean(mean_absolute_percentage_errors) * 100, 4))}%")
    return train_pred, test[column], fcast, confint


# daily_training_dataset functions
def weighted_avg(df, values, weights):
    """
    Weighted average of 2 columns

    parameters:
    - df (pd.DataFrame): dataframe
    - values (pd.Series): values for avg weight calculation
    - weights (pd.Series): weights for calculation
    """
    d = df[values]
    w = df[weights]
    return (d * w).sum() / w.sum()


## MACD
def calculate_macd(price, slow, fast, smooth):
    """
    Calculates Moving Average Convergence Divergence (MACD) and its components.

    Parameters:
    - price (pd.Series): Time series of asset prices.
    - slow (int): Number of days for the slow Exponential Moving Average (EMA).
    - fast (int): Number of days for the fast Exponential Moving Average (EMA).
    - smooth (int): Number of days for the smoothing of the MACD signal line.

    Returns:
    - tuple: Three DataFrames representing MACD line, Signal line, and MACD Histogram.
    """
    exp1 = price.ewm(span=fast, adjust=False).mean()
    exp2 = price.ewm(span=slow, adjust=False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns={'eth_close': 'macd'})
    signal = pd.DataFrame(macd.ewm(span=smooth, adjust=False).mean()).rename(columns={'macd': 'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns={0: 'hist'})
    return macd, signal, hist


## RSI
def calculate_rsi(df, periods=14, ema=True):
    """
    Calculates the Relative Strength Index (RSI) for a given DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the 'eth_close' column.
    - periods (int, optional): Number of periods for RSI calculation. Default is 14.
    - ema (bool, optional): If True, uses Exponential Moving Averages (EMA); otherwise, uses Simple Moving Averages (SMA).

    Returns:
    - pd.Series: Series representing the RSI values.
    """
    close_delta = df.diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    if ema == True:
        # Use exponential moving average
        ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window=periods, adjust=False).mean()
        ma_down = down.rolling(window=periods, adjust=False).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    return rsi


### Stochastic
def calculate_smi(df, n=14, k=3, d=3):
    """
    Calculates the Stochastic Momentum Indicator (SMI) for a given DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'eth_high', 'eth_low', and 'eth_close' columns.
    - n (int, optional): Number of periods for calculating highest high and lowest low. Default is 14.
    - k (int, optional): Number of periods for %K calculation. Default is 3.
    - d (int, optional): Number of periods for %D calculation and smoothing %K. Default is 3.

    Returns:
    - tuple: Two DataFrames with 'SMI' and 'SMI Signal' columns.
    """
    # Calculate highest high and lowest low over n periods
    high_max = df['eth_high'].rolling(n).max()
    low_min = df['eth_low'].rolling(n).min()

    # Calculate %K and %D using highest high and lowest low
    k_percent = 100 * (df['eth_close'] - low_min) / (high_max - low_min)
    d_percent = k_percent.rolling(k).mean().rolling(d).mean()

    # Calculate the SMI
    smi = k_percent - d_percent

    # Calculate the SMI Signal
    smi_signal = smi.rolling(3).mean()

    return smi, smi_signal


### Moving Average
def calculate_ma(df):
    """
    Calculate 12-day and 26-day moving averages for a given DataFrame.
    :param df: DataFrame containing 'Close' column.
    :return: A DataFrame with 'MA 12' and 'MA 26' columns.
    """
    # Calculate the 12-day and 26-day moving averages using rolling window
    ma_12 = df['eth_close'].rolling(window=12).mean()
    ma_26 = df['eth_close'].rolling(window=26).mean()

    return ma_12, ma_26


### Exponential Moving Average
def calculate_ema(df):
    """
    Calculate 12-day and 26-day exponential moving averages for a given DataFrame.
    :param df: DataFrame containing 'Close' column.
    :return: A DataFrame with 'EMA 12' and 'EMA 26' columns.
    """
    # Calculate the 12-day and 26-day exponential moving averages using ewm method
    ema_12 = df['eth_close'].ewm(span=12, adjust=True).mean()  ## Adjust True to be weighted for the closest data
    ema_26 = df['eth_close'].ewm(span=26, adjust=True).mean()

    return ema_12, ema_26


### On-Balance Indicator
def calculate_obv(df):
    """
    Calculates the On-Balance Volume (OBV) for a given DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'eth_close' and 'eth_volume' columns.

    Returns:
    - pd.Series: Series representing the On-Balance Volume (OBV).
    """
    copy = df.copy()
    obv = (np.sign(copy["eth_close"].diff()) * copy["eth_volume"]).fillna(0).cumsum()
    return obv


### Money-Flow Index
def calculate_mfi(df, period=14):
    """
    Calculates the Money Flow Index (MFI) for a given DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'eth_high', 'eth_low', 'eth_close', and 'eth_volume' columns.
    - period (int, optional): Number of periods for MFI calculation. Default is 14.

    Returns:
    - np.ndarray: Array representing the Money Flow Index (MFI) values.
    """
    high = df["eth_high"]
    low = df["eth_low"]
    close = df["eth_close"]
    volume = df["eth_volume"]

    typical_price = (high + low + close) / 3  ## mean of high, low, close

    raw_money_flow = typical_price * volume

    positive_money_flow = pd.DataFrame(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0))
    negative_money_flow = pd.DataFrame(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0))

    positive_money_flow_sum = positive_money_flow.rolling(window=period).sum()
    negative_money_flow_sum = negative_money_flow.rolling(window=period).sum()

    money_ratio = positive_money_flow_sum / negative_money_flow_sum
    money_flow_index = 100 - (100 / (1 + money_ratio))

    return money_flow_index[0].values


## Boilinger bands
def calculate_bollinger_bands(time_series, window_size=20, num_std_dev=2):
    """
    Calculates the Bollinger Bands for a given time series.

    Parameters:
    - time_series (pd.Series): Time series data.
    - window_size (int, optional): Number of periods for rolling mean and standard deviation. Default is 20.
    - num_std_dev (int, optional): Number of standard deviations for upper and lower bands. Default is 2.

    Returns:
    - tuple: Two Pandas Series representing the upper and lower Bollinger Bands.
    """
    # calculate rolling mean and standard deviation
    rolling_mean = time_series.rolling(window=window_size).mean()
    rolling_std = time_series.rolling(window=window_size).std()

    # calculate upper and lower bands
    upper_band = rolling_mean + (num_std_dev * rolling_std)
    lower_band = rolling_mean - (num_std_dev * rolling_std)

    return upper_band, lower_band


# holt winters functions

def gridsearch_timeseriessplit_holtwinters(dataframe, n_splits, test_size, trend_type, seasonal_type, damped_trend, init_method):
    """
    Grid searches Exponential Smoothing model parameters using TimeSeriesSplit and evaluates performance metrics.

    Parameters:
    - dataframe (pd.DataFrame): Time series data with 'logclose' and 'eth_close' columns.
    - n_splits (int): Number of splits for TimeSeriesSplit.
    - test_size (int): Size of the test set in each split.
    - trend_type (str): Type of trend component in the Exponential Smoothing model.
    - seasonal_type (str or None): Type of seasonal component in the Exponential Smoothing model, or None if no seasonality.
    - damped_trend (bool): Whether to dampen the trend in the Exponential Smoothing model.
    - init_method (str): Method for initializing the model.

    Returns:
    - tuple: Mean values of root mean squared error, mean squared error, mean absolute error, and mean absolute percentage error.

    Description:
    Performs a grid search using TimeSeriesSplit, training Exponential Smoothing models with different parameters,
    and evaluates the models on the test sets. Returns mean values of performance metrics including root mean squared error,
    mean squared error, mean absolute error, and mean absolute percentage error.
    """
    tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=0)
    root_mean_squared_errors = []
    mean_squared_errors = []
    mean_absolute_errors = []
    mean_absolute_percentage_errors = []

    for train_idx, val_idx in tss.split(dataframe):
        train = dataframe.iloc[train_idx]
        test = dataframe.iloc[val_idx]

        if seasonal_type is not None:
            hw = ExponentialSmoothing(
                train['logclose'],
                initialization_method=init_method,
                trend=trend_type,
                damped_trend=damped_trend,
                seasonal=seasonal_type,
                seasonal_periods=14)
        else:
            hw = ExponentialSmoothing(
                train['logclose'],
                initialization_method=init_method,
                trend=trend_type,
                damped_trend=damped_trend,
                seasonal=seasonal_type)

        res_hw = hw.fit()
        fcast = res_hw.forecast(len(test))

        rmse = mean_squared_error(test['eth_close'], np.exp(fcast), squared=False)
        root_mean_squared_errors.append(rmse)
        mse = mean_squared_error(test['eth_close'], np.exp(fcast), squared=True)
        mean_squared_errors.append(mse)
        mae = mean_absolute_error(test['eth_close'], np.exp(fcast))
        mean_absolute_errors.append(mae)
        mape = mean_absolute_percentage_error(test['eth_close'], np.exp(fcast))
        mean_absolute_percentage_errors.append(mape)

    print(f"Model tested for {n_splits} splits, test size: {test_size}")
    print(f"RMSE: {round(np.mean(root_mean_squared_errors), 4)}")
    print(f"MSE: {round(np.mean(mean_squared_errors), 4)}")
    print(f"MAE: {round(np.mean(mean_absolute_errors), 4)}")
    print(f"MAPE: {str(round(np.mean(mean_absolute_percentage_errors) * 100, 4))}%")
    return np.mean(root_mean_squared_errors), np.mean(mean_squared_errors), np.mean(mean_absolute_errors), np.mean(
        mean_absolute_percentage_errors)
