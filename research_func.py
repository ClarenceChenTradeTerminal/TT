# --------------Moving Average------------
def calculate_sma(price, window):
    return price.rolling(window).mean()

'''
If SMA has a golden cross, we should go long. 
If SMA has a dead cross, we should go short.
'''
# Note we should have market neutral

def apply_sma(price, sma):
    return price - sma
    
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

def BullBearTest(df, volume, short = 9, long = 30):
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
    '''
    eth = df[df.columns[1]].to_frame()
    mod_hamilton = sm.tsa.MarkovAutoregression(eth, k_regimes=2, order=4, switching_ar=False)
    res_hamilton = mod_hamilton.fit()
    durations = res_hamilton.expected_durations
    durations = durations.astype(int)
    print(durations)
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