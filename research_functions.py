# --------------Moving Average------------
def calculate_sma(price, window):
    return price.rolling(window).mean()

'''
If SMA has a golden cross, we should go long. 
If SMA has a dead cross, we should go short.
'''

def apply_sma(price, sma):
    return (price - sma) / price

def calculate_ema(price, window):
    alpha = 2 / (window + 1)
    ema = price.ewm(span=window, adjust=False).mean()
    return ema

def apply_ema(price, ema):
    return (price - ema) / price

# -----------------RSI------------------
def calculate_rsi(price, window = 14):
    delta = price.diff()
    up, down = delta.where(delta > 0, 0), -delta.where(delta < 0, 0)
    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi

'''
Since higher RSI means the coin is overbought, we should go short when RSI is high and go long when RSI is low
'''
def apply_rsi(price, window = 14):
    return 100 - calculate_rsi(price, window)

# ----------------KDJ-------------------

def calculate_rsv(close, low, high, window = 14):
    return (close - low.rolling(window).min()) / (high.rolling(window).max() - low.rolling(window).min()) * 100

def calculate_kdj(close, low, high, window=14):
    rsv = calculate_rsv(close, low, high, window)
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    j = 1 * k - 1 * d
    return k, d, j

def calculate_k(close, low, high, window=14):
    k, d, j = calculate_kdj(close, low, high, window)
    return k

def calculate_d(close, low, high, window=14):
    k, d, j = calculate_kdj(close, low, high, window)
    return d

def calculate_j(close, low, high, window=14):
    k, d, j = calculate_kdj(close, low, high, window)
    return j

# ----------------MACD------------------
def calculate_macd(price, short, long, signal):
    
    EMAshort = price.ewm(span=short, adjust=False).mean()
    EMAlong = price.ewm(span=long, adjust=False).mean()
    MACD = EMAshort - EMAlong
    Signal_Line = MACD.ewm(span=signal, adjust=False).mean()
    '''
    SMAshort = price.rolling(short).mean()
    SMAlong = price.rolling(long).mean()
    MACD = SMAshort - SMAlong
    Signal_Line = MACD.rolling(signal).mean()
    '''
    return MACD, Signal_Line

def apply_macd(price, short = 12, long = 26, signal = 9):
    MACD, Signal_Line = calculate_macd(price, short, long, signal)
    return MACD - Signal_Line

# -------------Bollinger Bands-----------
# Default multiplier = 2, window = 20
def calculate_bband(price, multiplier, window):
    SMA = calculate_sma(price, window)
    SD = price.rolling(window).std()
    UB = SMA + multiplier * SD
    LB = SMA - multiplier * SD
    ret = []
    ret.append(SMA)
    ret.append(UB)
    ret.append(LB)
    return ret

def bb_range(price, multiplier = 2, window = 20):
    ret = calculate_bband(price, multiplier, window)
    return ret[1] - ret[2]

'''
def apply_bband(price, multiplier = 2, window = 20):
    SMA, UB, LB = calculate_bband(price, multiplier, window)
'''

# ------------Kalman Filter-----------

import numpy as np
import pandas as pd

class KalmanFilter(object):
    def __init__(self, F=None, H=None, Q=None, R=None, P=None, x0=None):
        if F is None or H is None:
            raise ValueError("Set proper system dynamics.")
        
        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.m) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
                        (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def initialize_KalmanFilter(initial_close, dt = 0.1):
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])
    Q = np.array([[0.001, 0], [0, 0.001]])
    R = np.array([[0.1]])
    P = np.eye(2)
    x0 = np.array([[initial_close], [0]])
    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, P=P, x0=x0)
    return kf

def apply_KalmanFilter(df):
    alpha = pd.DataFrame()
    for column in df.columns:
        newdf = df[[column]].dropna().copy()
        close = newdf.iloc[-1].values[0]
        initial_close = newdf.iloc[0].values[0]
        singalpha = newdf.copy()
        kf = initialize_KalmanFilter(initial_close)
        predictions = []
        for i, z in enumerate(newdf[column]):
            prediction = kf.predict()
            predictions.append(prediction[0, 0])
            kf.update(np.array([[z]]))
            #singalpha.iloc[i] = (newdf.iloc[i] - predictions[-1])
            singalpha.iloc[i] = predictions[-1]
        if alpha.empty:
            alpha = singalpha
        else:
            alpha = pd.merge(alpha, singalpha, left_index=True, right_index=True, how='left')
    return alpha


#------------Alpha 191-----------

import math

def delta(price, n):
    shift_val = price.shift(n)
    diff = price - shift_val
    return diff

def rank(A):
    A.sort()
    return A

def corr(A, B):
    return np.corrcoef(A, B)[0, 1]

def alpha_001(volume, close, open, n=6):
    #(-1 * CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
    deltadf = pd.DataFrame()
    for column in volume.columns:
        newdeltadf = volume[[column]].dropna().copy()
        for i in range (len(newdeltadf[column])):
            newdeltadf.iloc[i] = math.log(newdeltadf.iloc[i])
        singdeltadf = newdeltadf.copy()
        for i,z in enumerate(newdeltadf[column]):
            twodelta = [newdeltadf.iloc[i][0]]
            if i > 0:
                twodelta.append(newdeltadf.iloc[i-1][0])
            else:
                twodelta.append(np.nan)
            twodelta = rank(twodelta)
            print(twodelta)
            singdeltadf.iloc[i] = twodelta
        if deltadf.empty:
            deltadf = singdeltadf
        else:
            deltadf = pd.merge(deltadf, singdeltadf, left_index=True, right_index=True, how='left')
    ret = (close-open)/open
    retdf = pd.DataFrame()
    for column in ret.columns:
        newretdf = ret[[column]].dropna().copy()
        singretdf = newretdf.copy()
        for i,z in enumerate(newretdf[column]):
            nret = []
            for j in range(n):
                if i-j < 0:
                    nret.append(np.nan)
                else:
                    nret.append(newdeltadf.iloc[i-j])
            nret = rank(nret)
            singretdf.iloc[i] = nret
        if retdf.empty:
            retdf = singretdf
        else:
            retdf = pd.merge(retdf, singretdf, left_index=True, right_index=True, how='left')
    correlationdf = pd.DataFrame(index=retdf.index, columns=retdf.columns)
    for idx in retdf.index:
        for col in retdf.columns:
            list1 = deltadf.at[idx, col]
            list2 = retdf.at[idx, col]
            correlation = corr(list1, list2)
            correlationdf.at[idx, col] = correlation
    return -1 * correlationdf


def alpha_002(close, low, high):
    return -1 * delta(((close-low)-(high-close))/((high-low)),1)


#------------Regime Shift-----------

def BullBearTest(df, short = 9, long = 30):
    dfcopy = df.copy()
    '''
    btc = df[df.columns[0]].to_frame()
    shortsma = btc.rolling(short).mean()
    longsma = btc.rolling(long).mean()
    '''
    eth = df[df.columns[1]].to_frame()
    shortsma = eth.rolling(short).mean()
    longsma = eth.rolling(long).mean()
    newdf = pd.DataFrame(index=df.index)
    newdf['BullResult'] = (shortsma > longsma).astype(bool)
    for column in dfcopy.columns:
        dfcopy[column] = newdf['BullResult'].values
    return dfcopy

import statsmodels.api as sm

def VarianceTest(df):
    dfcopy = df.copy()
    '''
    btc = df[df.columns[0]].to_frame()
    mod_kns = sm.tsa.MarkovRegression(btc, k_regimes=2, trend="n", switching_variance=True)
    '''
    
    eth = df[df.columns[1]].to_frame()
    mod_kns = sm.tsa.MarkovRegression(eth, k_regimes=2, trend="n", switching_variance=True)
    res_kns = mod_kns.fit()
    prob = res_kns.smoothed_marginal_probabilities
    prob['VarResult'] = prob.apply(lambda row: 1 if row[1] > row[0] else 0, axis=1)
    prob = prob.drop(columns=[0, 1])
    prob['VarResult'] = prob['VarResult'].astype(bool)
    
    '''
    btc = df[df.columns[0]].to_frame()
    mod_hamilton = sm.tsa.MarkovAutoregression(btc, k_regimes=2, order=4, switching_ar=False)
    '''
    
    eth = df[df.columns[1]].to_frame()
    mod_hamilton = sm.tsa.MarkovAutoregression(eth, k_regimes=2, order=4, switching_ar=False)
    res_hamilton = mod_hamilton.fit()
    durations = res_hamilton.expected_durations
    durations = durations.astype(int)
    '''
    '''
    for column in dfcopy.columns:
        dfcopy[column] = prob['VarResult'].values
    return dfcopy
    '''

'''
def SelectAlpha(value, volume, bearHigh = 'ts_ret(close, 14)', bullHigh = 'cs_mktneut((close-low)/(high-low)) + cs_mktneut(ts_willr((open+close+high+low)/4, 14, 14))', bearLow = 'apply_rsi(close)', bullLow = 'ts_ret(close, 14)'):
    criteria = BullBearTest(value, volume, VarianceTest(value))
    conditions = [(criteria['VarResult'] == 1) & (criteria['BullResult'] == 1), (criteria['VarResult'] == 0) & (criteria['BullResult'] == 1), (criteria['VarResult'] == 1) & (criteria['BullResult'] == 0), (criteria['VarResult'] == 0) & (criteria['BullResult'] == 0)]
    choices = [bullHigh, bullLow, bearHigh, bearLow]

    criteria['Alpha'] = np.select(conditions, choices)
    print(criteria)
    return criteria
'''
'''

import pandas as pd
import pandas_ta as ta

def apply_apo(close, window1=12, window2=26):
    return (calculate_ema(close, window1) - calculate_ema(close, window2)) / calculate_ema(close, window2)


from alpha_func_test import *

def algo_ridge(data, feature_list='', normalize='', save_model_pathname=''):
    x, y = split_feature_target(data, feature_list=feature_list)
    if normalize:
        x = algo_normalize(pd.DataFrame(x, columns=data.columns, index=data.index), normalize)
    model = linear_model.Ridge(alpha=1.0)
    model = algo_fit(x, y, model)
    y_pred = algo_predict(x, data.index, model)
    if save_model_pathname:
        save_model(model, save_model_pathname)
    return y_pred

def algo_ridge_split(data, feature_list='', normalize='', test_size=0.2, save_model_pathname=''):
    x, y = split_feature_target(data, feature_list=feature_list)
    x = algo_normalize(x, normalize)
    x_train, _, y_train, _ = train_test_split(x, y, test_size=test_size, shuffle=False)
    model = linear_model.Ridge(alpha=1.0)
    model = algo_fit(x_train, y_train, model)
    y_pred = algo_predict(x, data.index, model)
    if save_model_pathname:
        save_model(model, save_model_pathname)
    return y_pred

def algo_lasso(data, feature_list='', normalize='', test_size=0.2, save_model_pathname=''):
    x, y = split_feature_target(data, feature_list=feature_list)
    x = algo_normalize(x, normalize)
    x_train, _, y_train, _ = train_test_split(x, y, test_size=test_size, shuffle=False)
    model = linear_model.Lasso(alpha=100.0)
    model = algo_fit(x_train, y_train, model)
    y_pred = algo_predict(x, data.index, model)
    if save_model_pathname:
        save_model(model, save_model_pathname)
    return y_pred

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

def algo_rf(data, feature_list='', normalize='', test_size=0.2, n_estimators=10, save_model_pathname=''):
    x, y = split_feature_target(data, feature_list=feature_list)
    x = algo_normalize(x, normalize)
    x_train, _, y_train, _ = train_test_split(x, y, test_size=test_size, shuffle=False)
    model = RandomForestRegressor(n_estimators)
    model = algo_fit(x_train, y_train, model)
    y_pred = algo_predict(x, data.index, model)
    if save_model_pathname:
        save_model(model, save_model_pathname)
    return y_pred

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd

def algo_svr(data, feature_list='', normalize='', test_size=0.2, save_model_pathname=''):
    x, y = split_feature_target(data, feature_list=feature_list)
    x = algo_normalize(x, normalize)
    x_train, _, y_train, _ = train_test_split(x, y, test_size=test_size, shuffle=False)
    model = make_pipeline(PolynomialFeatures(degree=2), SVR(kernel='rbf', C=10, gamma=0.1))
    model = algo_fit(x_train, y_train, model)
    y_pred = algo_predict(x, data.index, model)
    if save_model_pathname:
        save_model(model, save_model_pathname)
    return y_pred

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

def algo_decision_tree(data, feature_list='', normalize='', test_size=0.2, random_state=0, save_model_pathname=''):
    x, y = split_feature_target(data, feature_list=feature_list)
    x = algo_normalize(x, normalize)
    x_train, _, y_train, _ = train_test_split(x, y, test_size=test_size, shuffle=False)
    model = DecisionTreeRegressor(random_state)
    model = algo_fit(x_train, y_train, model)
    y_pred = algo_predict(x, data.index, model)
    if save_model_pathname:
        save_model(model, save_model_pathname)
    return y_pred

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

def algo_knn(data, feature_list='', normalize='', test_size=0.2, n_neighbors=5, save_model_pathname=''):
    x, y = split_feature_target(data, feature_list=feature_list)
    x = algo_normalize(x, normalize)
    x_train, _, y_train, _ = train_test_split(x, y, test_size=test_size, shuffle=False)
    model = KNeighborsRegressor(n_neighbors)
    model = algo_fit(x_train, y_train, model)
    y_pred = algo_predict(x, data.index, model)
    if save_model_pathname:
        save_model(model, save_model_pathname)
    return y_pred

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

def algo_gradient_boosting(data, feature_list='', normalize='', test_size=0.2, n_estimators=10, save_model_pathname=''):
    x, y = split_feature_target(data, feature_list=feature_list)
    x = algo_normalize(x, normalize)
    x_train, _, y_train, _ = train_test_split(x, y, test_size=test_size, shuffle=False)
    model = GradientBoostingRegressor(n_estimators=n_estimators)
    model = algo_fit(x_train, y_train, model)
    y_pred = algo_predict(x, data.index, model)
    if save_model_pathname:
        save_model(model, save_model_pathname)
    return y_pred

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def algo_lstm_split(data, feature_list='', normalize='', test_size=0.2, save_model_pathname=''):
    x, y = split_feature_target(data, feature_list=feature_list)
    x = algo_normalize(x, normalize)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    model = Sequential()
    model.add(LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32, activation='relu', return_sequences=False)) 
    model.add(Dense(1)) 
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_test, y_test), 
              callbacks=[early_stopping], verbose=1)
    x_scaled = scaler.transform(x).reshape(-1, 1, x.shape[1])
    y_pred = model.predict(x_scaled)
    if save_model_pathname:
        model.save(save_model_pathname.replace('.pkl', '.h5')) 
    return y_pred
