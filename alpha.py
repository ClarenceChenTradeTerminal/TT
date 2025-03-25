import sourcedefender
import sys
sys.path.append("/tt_data/ttshared/")
from alpha_func_test import *
import os
import numpy as np
import pandas as pd
import pandas_ta as ta

rec = 'KalmanFilter(close)'
rec1 = '(cs_booksize(cs_mktneut(calculate_rsi(close)), 2000) + cs_booksize(cs_mktneut(alpha_002(close, low, high)), 8000))/2'

alpharunning = 'ts_willr(close, 18, 16)' 
alpha = 'ts_cci(close, 20, 20)'
alphabest = 'cs_mktneut((close-low)/(high-low)) + cs_mktneut(ts_willr((open+close+high+low)/4, 14, 14))'

alpha001 = '-1 * ts_corr(ts_rank(delta(np.log(volume),1)), ts_rank(((close-open)/open)), 6)'
alpha002 = '-1 * delta(((close-low)-(high-close))/((high-low)),1)'


bearHigh = alpha
bullHigh = alphabest
bearLow = alpha
bullLow = alphabest

highVar = 'cond_exp('+ bullHigh +', BullBearTest(close), '+ bearHigh +')'
lowVar = 'cond_exp('+ bullLow +', BullBearTest(close), '+ bearLow +')'

poscor50 = ['DOGEUSDT', 'XLMUSDT', 'BNBUSDT', 'XRPUSDT', 'BCHUSDT', 'BTCUSDT', 'MANAUSDT', 'ETHUSDT', 'FTMUSDT', 'TRXUSDT', 'ADAUSDT', 'LINKUSDT', 'MKRUSDT', 'CHZUSDT', 'ETCUSDT', 'EOSUSDT', 'AVAXUSDT', 'GRTUSDT', 'SOLUSDT', 'VETUSDT', 'LDOUSDT', 'AXSUSDT', 'FILUSDT']
poscor30 = ['FILUSDT', 'VETUSDT', 'BNBUSDT', 'LDOUSDT', 'SOLUSDT', 'GRTUSDT', 'BCHUSDT', 'DOGEUSDT', 'ETHUSDT', 'BTCUSDT', 'AVAXUSDT', 'TRXUSDT', 'ADAUSDT', 'XRPUSDT', 'XLMUSDT', 'ETCUSDT']
alphatest = 'cond_exp('+ highVar +', VarianceTest(close), ' + lowVar +')'
op_simulate(alpha, slippage=0.0003, universe=poscor30, freq='8H', pos_limit=0.05, sd='2020-01-01', ed='2024-12-01', mktneut= True, live = False, alloc_usdt=10000, save_alpha=False)
#op_simulate(alpha, slippage=0.0003, universe=poscor30, freq='8H', pos_limit=0.05, sd='2023-05-01', ed='2024-12-01', mktneut= True, live = True, alloc_usdt=10000, save_alpha=False)
#op_simulate(alpharunning, slippage=0.0003, universe='top50', freq='8H', pos_limit=0.05, sd='2024-09-23', ed='2024-11-13', mktneut= True, live = True, alloc_usdt=10000, save_alpha=False)
#print(dir(alpha_func))

'''
for i in [3, 5, 10, 20, 30, 40]:
    alphalooping = f'apply_sma(close, calculate_sma(close, {i}))'
    op_simulate(alphalooping, slippage=0.0, universe='top50', freq='8H', pos_limit=0.05, sd='2020-02-01', ed='2023-05-01', mktneut= False, live = False, alloc_usdt=10000, save_alpha=False)
    op_simulate(alphalooping, slippage=0.0, universe='top50', freq='8H', pos_limit=0.05, sd='2023-05-01', ed='2025-01-01', mktneut= False, live = True, alloc_usdt=10000, save_alpha=False)
'''
'''
Functions:

1. cs_rank () : find the ranking of a crypto for an alpha
EX: (cs_rank(calculate_rsi(close)) + cs_rank(apply_macd(close))
'''

