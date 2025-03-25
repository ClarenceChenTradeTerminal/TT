from operators import *
from binance_f import RequestClient
from binance_f.model import *
import numpy as np
import pandas as pd
import json
import sys
import gc
import logging
import os
import datetime
from binance.client import Client
import concurrent.futures
import random
import math
import time

prod = 1
path = os.getcwd()
f = open(sys.argv[1], "r")
configs = json.loads(f.read())
key = configs['key']
secret = configs['secret']
alloc_usdt = configs['allocation']

sub = '/data/market_maker/datafolder2/' # data folder

# Close data path
close_filename = 'close.csv'
close_file_path = sub + close_filename

# Open data path
open_filename = 'open.csv'
open_file_path = sub + open_filename

# High data path  
high_filename = 'high.csv'
high_file_path = sub + high_filename

# Low data path
low_filename = 'low.csv'
low_file_path = sub + low_filename

lookback = 100
trade_interval = 8

if not os.path.exists(sub):
    os.mkdir(sub)
# Read close data
elif os.path.exists(close_file_path):
    close = pd.read_csv(close_file_path, index_col=[0]).iloc[-lookback:, :]
# Read open data
elif os.path.exists(open_file_path):
    open = pd.read_csv(open_file_path, index_col=[0]).iloc[-lookback:, :]
# Read high data
elif os.path.exists(high_file_path):
    high = pd.read_csv(high_file_path, index_col=[0]).iloc[-lookback:, :]
# Read low data
elif os.path.exists(low_file_path):
    low = pd.read_csv(low_file_path, index_col=[0]).iloc[-lookback:, :]


os.environ['NUMEXPR_MAX_THREADS'] = '32'

logging.basicConfig(filename="logfile2.log", format='%(asctime)s %(message)s', filemode='w', level=logging.INFO)
request_client = RequestClient(api_key=key, secret_key=secret)
binance_client = Client(key, secret)

q_map = {'BTCUSDT': 3, 'ETHUSDT': 3, 'BCHUSDT': 3, 'XRPUSDT': 1, 'EOSUSDT': 1, 'LTCUSDT': 3, 'TRXUSDT': 0, 'ETCUSDT': 2, 'LINKUSDT': 2, 'XLMUSDT': 0, 'ADAUSDT': 0, 'XMRUSDT': 3, 'DASHUSDT': 3, 'ZECUSDT': 3, 'XTZUSDT': 1, 'BNBUSDT': 2, 'ATOMUSDT': 2, 'ONTUSDT': 1, 'IOTAUSDT': 1, 'BATUSDT': 1, 'VETUSDT': 0, 'NEOUSDT': 2, 'QTUMUSDT': 1, 'IOSTUSDT': 0, 'THETAUSDT': 1, 'ALGOUSDT': 1, 'ZILUSDT': 0, 'KNCUSDT': 0, 'ZRXUSDT': 1, 'COMPUSDT': 3, 'OMGUSDT': 1, 'DOGEUSDT': 0, 'SXPUSDT': 1, 'KAVAUSDT': 1, 'BANDUSDT': 1, 'RLCUSDT': 1, 'WAVESUSDT': 1, 'MKRUSDT': 3, 'SNXUSDT': 1, 'DOTUSDT': 1, 'DEFIUSDT': 3, 'YFIUSDT': 3, 'BALUSDT': 1, 'CRVUSDT': 1, 'TRBUSDT': 1, 'YFIIUSDT': 3, 'RUNEUSDT': 0, 'SUSHIUSDT': 0, 'SRMUSDT': 0, 'BZRXUSDT': 0, 'EGLDUSDT': 1, 'SOLUSDT': 0, 'ICXUSDT': 0, 'STORJUSDT': 0, 'BLZUSDT': 0, 'UNIUSDT': 0, 'AVAXUSDT': 0, 'FTMUSDT': 0, 'HNTUSDT': 0, 'ENJUSDT': 0, 'FLMUSDT': 0, 'TOMOUSDT': 0, 'RENUSDT': 0, 'KSMUSDT': 1, 'NEARUSDT': 0, 'AAVEUSDT': 1, 'FILUSDT': 1, 'RSRUSDT': 0, 'LRCUSDT': 0, 'MATICUSDT': 0, 'OCEANUSDT': 0, 'CVCUSDT': 0, 'BELUSDT': 0, 'CTKUSDT': 0, 'AXSUSDT': 0, 'ALPHAUSDT': 0, 'ZENUSDT': 1, 'SKLUSDT': 0, 'GRTUSDT': 0, '1INCHUSDT': 0, 'BTCBUSD': 3, 'AKROUSDT': 0, 'CHZUSDT': 0, 'SANDUSDT': 0, 'ANKRUSDT': 0, 'LUNAUSDT': 0, 'BTSUSDT': 0, 'LITUSDT': 1, 'UNFIUSDT': 1, 'DODOUSDT': 1, 'REEFUSDT': 0, 'RVNUSDT': 0, 'SFPUSDT': 0, 'XEMUSDT': 0, 'BTCSTUSDT': 1, 'COTIUSDT': 0, 'CHRUSDT': 0, 'MANAUSDT': 0, 'ALICEUSDT': 1, 'BTCUSDT_210625': 3, 'ETHUSDT_210625': 3, 'HBARUSDT': 0, 'ONEUSDT': 0, 'LINAUSDT': 0, 'STMXUSDT': 0, 'DENTUSDT': 0, 'CELRUSDT': 0, 'HOTUSDT': 0, 'MTLUSDT': 0, 'OGNUSDT': 0, 'BTTUSDT': 0, 'NKNUSDT': 0, 'SCUSDT': 0, 'DGBUSDT': 0, '1000SHIBUSDT': 0, 'ICPUSDT': 0, 'BAKEUSDT': 0, 'GTCUSDT': 1, 'ETHBUSD': 3, 'BTCUSDT_210924': 3, 'ETHUSDT_210924': 3, 'LDOUSDT': 0, 'APTUSDT': 1, 'ARBUSDT': 1, 'QNTUSDT': 1, 'APEUSDT': 0, 'RNDRUSDT': 1, 'OPUSDT': 1, 'STXUSDT': 0, 'CFXUSDT': 0, 'FLOWUSDT': 1, 'IMXUSDT': 0, 'INJUSDT': 1, 'ASTRUSDT': 0, 'WOOUSDT':0, 'KLAYUSDT':1, 'MASKUSDT':0, 'CVXUSDT':0, 'DYDXUSDT':1 ,'ENSUSDT':1 , 'CELOUSDT':0 ,'C98USDT': 0, 'XVSUSDT': 1, 'FXSUSDT': 1, 'ARUSDT':1 ,'LQTYUSDT':1 , 'ROSEUSDT':0 , 'FETUSDT':0 ,  'AUDIOUSDT':0 ,'GMTUSDT':0 ,'EDUUSDT':0 , 'IDUSDT': 0, 'IDEXUSDT': 0, 'IOTXUSDT':0 ,'GALAUSDT':0 , 'MAGICUSDT': 1, 'RADUSDT': 0, 'USDCUSDT': 0, 'JASMYUSDT':0 ,'AGIXUSDT': 0, 'JOEUSDT': 0, 'TRUUSDT': 0, 'HFTUSDT': 0, 'TLMUSDT':0 , 'BNXUSDT': 1, 'DUSKUSDT':0 , 'PERPUSDT': 1, 'PHBUSDT': 0, 'ARPAUSDT': 0, 'ACHUSDT': 0, 'MINAUSDT':0 , 'API3USDT':1 , 'CKBUSDT': 0, 'RDNTUSDT': 0,'HIGHUSDT':1 , 'ANTUSDT':1 , 'CTSIUSDT': 0, 'HOOKUSDT': 1, 'SUIUSDT': 1, 'SSVUSDT': 2, 'LEVERUSDT': 0, 'TUSDT': 0, 'PEOPLEUSDT': 0, 'SPELLUSDT': 0, 'STGUSDT': 0 , 'AMBUSDT':0 , 'ATAUSDT': 0, 'LPTUSDT': 1, 'GALUSDT': 0, 'UMAUSDT': 0, 'DARUSDT': 1, 'GMXUSDT': 1}
p_map = {'BTCUSDT': 2, 'ETHUSDT': 2, 'BCHUSDT': 2, 'XRPUSDT': 4, 'EOSUSDT': 3, 'LTCUSDT': 2, 'TRXUSDT': 5, 'ETCUSDT': 3, 'LINKUSDT': 3, 'XLMUSDT': 5, 'ADAUSDT': 5, 'XMRUSDT': 2, 'DASHUSDT': 2, 'ZECUSDT': 2, 'XTZUSDT': 3, 'BNBUSDT': 3, 'ATOMUSDT': 3, 'ONTUSDT': 4, 'IOTAUSDT': 4, 'BATUSDT': 4, 'VETUSDT': 6, 'NEOUSDT': 3, 'QTUMUSDT': 3, 'IOSTUSDT': 6, 'THETAUSDT': 4, 'ALGOUSDT': 4, 'ZILUSDT': 5, 'KNCUSDT': 5, 'ZRXUSDT': 4, 'COMPUSDT': 2, 'OMGUSDT': 4, 'DOGEUSDT': 5, 'SXPUSDT': 4, 'KAVAUSDT': 4, 'BANDUSDT': 4, 'RLCUSDT': 4, 'WAVESUSDT': 4, 'MKRUSDT': 2, 'SNXUSDT': 3, 'DOTUSDT': 3, 'DEFIUSDT': 1, 'YFIUSDT': 1, 'BALUSDT': 3, 'CRVUSDT': 3, 'TRBUSDT': 3, 'YFIIUSDT': 1, 'RUNEUSDT': 4, 'SUSHIUSDT': 4, 'SRMUSDT': 4, 'BZRXUSDT': 4, 'EGLDUSDT': 3, 'SOLUSDT': 4, 'ICXUSDT': 4, 'STORJUSDT': 4, 'BLZUSDT': 5, 'UNIUSDT': 4, 'AVAXUSDT': 4, 'FTMUSDT': 6, 'HNTUSDT': 4, 'ENJUSDT': 5, 'FLMUSDT': 4, 'TOMOUSDT': 4, 'RENUSDT': 5, 'KSMUSDT': 2, 'NEARUSDT': 4, 'AAVEUSDT': 3, 'FILUSDT': 3, 'RSRUSDT': 6, 'LRCUSDT': 5, 'MATICUSDT': 5, 'OCEANUSDT': 5, 'CVCUSDT': 5, 'BELUSDT': 5, 'CTKUSDT': 5, 'AXSUSDT': 5, 'ALPHAUSDT': 5, 'ZENUSDT': 3, 'SKLUSDT': 5, 'GRTUSDT': 5, '1INCHUSDT': 4, 'BTCBUSD': 1, 'AKROUSDT': 5, 'CHZUSDT': 5, 'SANDUSDT': 5, 'ANKRUSDT': 6, 'LUNAUSDT': 4, 'BTSUSDT': 5, 'LITUSDT': 3, 'UNFIUSDT': 3, 'DODOUSDT': 3, 'REEFUSDT': 6, 'RVNUSDT': 5, 'SFPUSDT': 4, 'XEMUSDT': 4, 'BTCSTUSDT': 3, 'COTIUSDT': 5, 'CHRUSDT': 4, 'MANAUSDT': 4, 'ALICEUSDT': 3, 'BTCUSDT_210625': 1, 'ETHUSDT_210625': 2, 'HBARUSDT': 5, 'ONEUSDT': 5, 'LINAUSDT': 5, 'STMXUSDT': 5, 'DENTUSDT': 6, 'CELRUSDT': 5, 'HOTUSDT': 6, 'MTLUSDT': 4, 'OGNUSDT': 4, 'BTTUSDT': 6, 'NKNUSDT': 5, 'SCUSDT': 6, 'DGBUSDT': 5, '1000SHIBUSDT': 6, 'ICPUSDT': 0, 'BAKEUSDT': 4, 'GTCUSDT': 3, 'ETHBUSD': 2, 'BTCUSDT_210924': 1, 'ETHUSDT_210924': 2, 'LDOUSDT': 4, 'APTUSDT': 3, 'ARBUSDT': 4, 'QNTUSDT': 2, 'APEUSDT': 3, 'RNDRUSDT': 4, 'OPUSDT': 4, 'STXUSDT': 4, 'CFXUSDT': 4, 'FLOWUSDT': 3, 'IMXUSDT': 4, 'INJUSDT': 3, 'ASTRUSDT': 5, 'WOOUSDT':5, 'KLAYUSDT':4, 'MASKUSDT':3, 'CVXUSDT':3, 'DYDXUSDT':3 , 'ENSUSDT':3 , 'CELOUSDT':3 ,'C98USDT': 4, 'XVSUSDT': 3, 'FXSUSDT': 3, 'ARUSDT':3 ,'LQTYUSDT': 4, 'ROSEUSDT':5 , 'FETUSDT': 4, 'AUDIOUSDT':4 ,'GMTUSDT':4 ,'EDUUSDT': 4, 'IDUSDT': 4, 'IDEXUSDT':5 , 'IOTXUSDT':5 , 'GALAUSDT': 5 ,'MAGICUSDT': 4, 'RADUSDT': 3, 'USDCUSDT':5 , 'JASMYUSDT':6 , 'AGIXUSDT': 4, 'JOEUSDT': 4, 'TRUUSDT': 5, 'HFTUSDT': 4, 'TLMUSDT': 5, 'BNXUSDT': 4, 'DUSKUSDT': 5, 'PERPUSDT': 4, 'PHBUSDT': 4, 'ARPAUSDT': 5, 'ACHUSDT': 5, 'MINAUSDT':4 , 'API3USDT': 3, 'CKBUSDT': 6, 'RDNTUSDT': 4,'HIGHUSDT': 3, 'ANTUSDT': 3, 'CTSIUSDT': 4, 'HOOKUSDT': 3, 'SUIUSDT': 4, 'SSVUSDT': 2, 'LEVERUSDT': 6, 'TUSDT': 5, 'PEOPLEUSDT': 5, 'SPELLUSDT': 7, 'STGUSDT': 4, 'AMBUSDT': 5, 'ATAUSDT': 4, 'LPTUSDT': 3, 'GALUSDT': 4, 'UMAUSDT': 4, 'DARUSDT': 4, 'GMXUSDT': 2}
#all_tickers = ['BTCUSDT', 'ETHUSDT',  'BNBUSDT',  'XRPUSDT',  'ADAUSDT',  'DOGEUSDT', 'SOLUSDT',  'MATICUSDT',    'TRXUSDT',  'LTCUSDT',  'DOTUSDT',  'AVAXUSDT', 'ATOMUSDT', 'UNIUSDT',  'XMRUSDT',  'ETCUSDT',  'XLMUSDT',  'BCHUSDT',  'ICPUSDT',  'LDOUSDT',  'FILUSDT',  'APTUSDT',  'HBARUSDT', 'VETUSDT',  'ARBUSDT',  'NEARUSDT', 'QNTUSDT',  'GRTUSDT',  'APEUSDT',  'ALGOUSDT', 'SANDUSDT', 'EOSUSDT',  'EGLDUSDT', 'AAVEUSDT', 'RNDRUSDT', 'OPUSDT',   'FTMUSDT',  'MANAUSDT', 'XTZUSDT',  'THETAUSDT',    'STXUSDT',  'CFXUSDT',  'AXSUSDT',  'FLOWUSDT', 'NEOUSDT',  'CHZUSDT',  'CRVUSDT',  'IMXUSDT',  'MKRUSDT',  'SNXUSDT',  'INJUSDT', 'LINKUSDT']
all_tickers = ['FILUSDT', 'VETUSDT', 'BNBUSDT', 'LDOUSDT', 'SOLUSDT', 'GRTUSDT', 'BCHUSDT', 'DOGEUSDT', 'ETHUSDT', 'BTCUSDT', 'AVAXUSDT', 'TRXUSDT', 'MATICUSDT', 'ADAUSDT', 'XRPUSDT', 'XLMUSDT', 'ETCUSDT']

def load_hist_data_all(all_tickers):
    print('!!!!')
    def load_hist_data(ticker):
        t = random.randint(1, 10)
        time.sleep(t)
        while True:
            try:
                logging.info("start loading " + ticker)
                klines = binance_client.get_historical_klines(ticker, Client.KLINE_INTERVAL_8HOUR, start_str= None, end_str= None, limit=400)
                break
            except Exception as e:
                logging.error(e)
        data = pd.DataFrame(klines, columns=['Starttime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Timestamp', 'Volume_qa', 'Numtrades', 'Takerbuyvol', 'Takerbuyvol_qa', 'Ignore'])
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='ms').dt.round('s')
        # Close
        data_close = data[['Timestamp', 'Close']]
        data_close.columns = ['Timestamp', ticker]
        data_close.set_index('Timestamp', inplace=True)
        data_close = data_close.astype('float')
        # Open
        data_open = data[['Timestamp', 'Open']]
        data_open.columns = ['Timestamp', ticker]
        data_open.set_index('Timestamp', inplace=True)
        data_open = data_open.astype('float')
        # High
        data_high = data[['Timestamp', 'High']]
        data_high.columns = ['Timestamp', ticker]
        data_high.set_index('Timestamp', inplace=True)
        data_high = data_high.astype('float')
        # Low
        data_low = data[['Timestamp', 'Low']]
        data_low.columns = ['Timestamp', ticker]
        data_low.set_index('Timestamp', inplace=True)
        data_low = data_low.astype('float')


        return data_close, data_open, data_high, data_low

    logging.info("no hist data, loading it now")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(load_hist_data, all_tickers))
    
    # All close
    results_close = [result[0] for result in results]
    data_all_close = pd.concat(results_close, axis=1, sort=True)
    data_all_close = data_all_close[all_tickers]
    data_all_close = data_all_close.iloc[:-1, ]
    data_all_close.to_csv(close_file_path)
    # All open 
    results_open = [result[1] for result in results]
    data_all_open = pd.concat(results_open, axis=1, sort=True)
    data_all_open = data_all_open[all_tickers]
    data_all_open = data_all_open.iloc[:-1, ]
    data_all_open.to_csv(open_file_path)
    # All high
    results_high = [result[2] for result in results]
    data_all_high = pd.concat(results_high, axis=1, sort=True)
    data_all_high = data_all_high[all_tickers]
    data_all_high = data_all_high.iloc[:-1,]
    data_all_high.to_csv(high_file_path)
    # All low
    results_low = [result[3] for result in results]
    data_all_low = pd.concat(results_low, axis=1, sort=True)
    data_all_low = data_all_low[all_tickers]
    data_all_low = data_all_low.iloc[:-1,]
    data_all_low.to_csv(low_file_path)
    
    logging.info('\n')
    # logging.info(data.tail())
    # data.to_csv(file_path)
    # logging.info('saved data to ' + file_path)
    gc.collect()
    return data_all_close, data_all_open, data_all_high, data_all_low


def get_price(ticker):
    result = request_client.get_candlestick_data(symbol=ticker, interval=CandlestickInterval.HOUR8, limit=1)
    return [ticker, float(result[0].close), float(result[0].open), float(result[0].high), float(result[0].low)]
    


def append_price(close, open, high, low, all_tickers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_tickers)) as executor:
        data_new = list(executor.map(get_price, all_tickers))

    data_close_new = {i[0]: i[1] for i in data_new}
    print(data_close_new)
    close_last_update_time = close.index[-1]
    data_close_new = pd.DataFrame.from_dict(data_close_new, orient='index').transpose()[all_tickers]
    data_close_new.index = [datetime.datetime.utcnow().replace(second=0, microsecond=0)]
    data_close_new = data_close_new[data_close_new.index > close_last_update_time]
    close = pd.concat([close, data_close_new]).iloc[-lookback:, :]
    data_open_new = {i[0]: i[2] for i in data_new}
    open_last_update_time = open.index[-1]
    data_open_new = pd.DataFrame.from_dict(data_open_new, orient='index').transpose()[all_tickers]
    data_open_new.index = [datetime.datetime.utcnow().replace(second=0, microsecond=0)]
    data_open_new = data_open_new[data_open_new.index > open_last_update_time]
    open = pd.concat([open, data_open_new]).iloc[-lookback:, :]
    data_high_new = {i[0]: i[3] for i in data_new}
    high_last_update_time = high.index[-1]
    data_high_new = pd.DataFrame.from_dict(data_high_new, orient='index').transpose()[all_tickers]
    data_high_new.index = [datetime.datetime.utcnow().replace(second=0, microsecond=0)]
    data_high_new = data_high_new[data_high_new.index > high_last_update_time]
    high = pd.concat([high, data_high_new]).iloc[-lookback:, :]
    data_low_new = {i[0]: i[4] for i in data_new}
    low_last_update_time = low.index[-1]
    data_low_new = pd.DataFrame.from_dict(data_low_new, orient='index').transpose()[all_tickers]
    data_low_new.index = [datetime.datetime.utcnow().replace(second=0, microsecond=0)]
    data_low_new = data_low_new[data_low_new.index > low_last_update_time]
    low = pd.concat([low, data_low_new]).iloc[-lookback:, :]

        
    logging.info('\n')
    # logging.info(close.tail())

    return data_close_new, data_open_new, data_high_new, data_low_new, close, open, high, low


def run_strategy(close, open, high, low):
    data_close_new, data_open_new, data_high_new, data_low_new, close, open, high, low = append_price(close, open, high, low, all_tickers)
    logging.info('finished loading new data')
    alpha1 = ts_cci(close, 20, 20)
    #alpha1 = cs_mktneut((close+open-2*low)/(high-low)) + cs_mktneut(ts_willr(close, 14, 14))
    alpha2 = cs_booksize(cs_mktneut(alpha1), alloc_usdt)
    alpha = cs_expand_balance(cs_poslimit(alpha2, 0.1 * alloc_usdt), alloc_usdt)
    rec = alpha.iloc[-1]
    print(rec)
    a1rec = alpha1.iloc[-1]
    a2rec = alpha2.iloc[-1]

    ticker_target = rec.to_dict()

    raw_weight = a1rec.to_dict()
    bk_neu = a2rec.to_dict()

    open_pos = request_client.get_position_v2()
    open_pos = {i.symbol: float(i.positionAmt) for i in open_pos}
    ticker_target_open = []
    for i in ticker_target.keys():
        if i in open_pos.keys():
            if math.isnan(ticker_target[i]):
                ticker_target_open.append([i, 0, open_pos[i]])
            else:
                ticker_target_open.append([i, ticker_target[i], open_pos[i]])
        else:
            if math.isnan(ticker_target[i]):
                ticker_target_open.append([i, 0, 0])
            else:
                ticker_target_open.append([i, ticker_target[i], 0])
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ticker_target_open)) as executor:
        executor.map(execute_orders, ticker_target_open)
    logging.info('finished execution')
    check_open_pos()
    data_close_new.to_csv(close_file_path, mode='a', header=False)
    data_open_new.to_csv(open_file_path, mode='a', header=False)
    data_high_new.to_csv(high_file_path, mode='a', header=False)
    data_low_new.to_csv(low_file_path, mode='a', header=False)
    gc.collect()

    open_pos = request_client.get_position_v2()
    for i in open_pos:
        if abs(float(i.positionAmt)) > 0:
            logging.info('ticker: ' + i.symbol + ' target:' + str(ticker_target[i.symbol]) + ' open: ' + str(float(i.positionAmt) * float(i.markPrice)) + ' Diff:' + str(ticker_target[i.symbol]-float(i.positionAmt) * float(i.markPrice)) + ' Fill Rate: ' + str(ticker_target[i.symbol] / (float(i.positionAmt) * float(i.markPrice)))) 
            logging.info("raw wgt: " + str(raw_weight[i.symbol]))
            logging.info("bk neu: "  + str(bk_neu[i.symbol]))

    return close, open, high, low


def execute_orders(ticker_target_open):
    print(ticker_target_open)
    ticker = ticker_target_open[0]
    target_usdt = ticker_target_open[1]
    open_pos = ticker_target_open[2]
    data = request_client.get_symbol_orderbook_ticker(ticker)[0]
    ask = data.askPrice
    bid = data.bidPrice
    mp = 0.5 * (ask + bid)
    p = int(mp) if p_map[ticker] == 0 else round(float(mp), p_map[ticker])
    diffq = int(target_usdt / float(mp) - float(open_pos)) if q_map[ticker] == 0 else round(target_usdt / float(mp) - float(open_pos), q_map[ticker])
    ask = int(ask) if p_map[ticker] == 0 else round(float(ask), p_map[ticker])
    bid = int(bid) if p_map[ticker] == 0 else round(float(bid), p_map[ticker])
    side = 'BUY' if diffq > 0 else 'SELL'
    request_client.cancel_all_orders(ticker)
    if diffq != 0:
        if abs(diffq * p) >= 5:
            if prod == 1:
                if diffq > 0:
                    try:
                        print(request_client.post_order(symbol=ticker, side=side, ordertype='MARKET', quantity=str(diffq)))
                    except Exception as e:
                        logging.error(ticker, e)
                else:
                    try:
                        request_client.post_order(symbol=ticker, side=side, ordertype='MARKET', quantity=str(-diffq))
                    except Exception as e:
                        logging.error(ticker, e)
        else:
            if prod == 1:
                if diffq > 0 and open_pos < 0:
                    if p * diffq < -open_pos:
                        try:
                            request_client.post_order(symbol=ticker, side=side, ordertype='MARKET', quantity=str(diffq), reduceOnly=True)
                        except Exception as e:
                            logging.error(ticker, e)
                elif diffq < 0 and open_pos > 0:
                    if -p * diffq < open_pos:
                        try:
                            request_client.post_order(symbol=ticker, side=side, ordertype='MARKET', quantity=str(-diffq), reduceOnly=True)
                        except Exception as e:
                            logging.error(ticker, e)
    
def ts_max(df, window=10):
    return df.rolling(window).max()

def ts_min(df, window=10):
    return df.rolling(window).min()

def ts_mean(data, window):
    return data.rolling(window).mean()

def ts_willr(data, window, window1):
    outputs = - (ts_max(ts_mean(data, window), window1) - data) / (ts_max(ts_mean(data, window), window1) - ts_min(ts_mean(data, window), window1))
    return outputs

def ts_cci(inputs, window1, window2):
    pp = (ts_max(inputs, window1) + ts_min(inputs, window1) + inputs) / 3
    return (pp - pp.rolling(window2, min_periods=int(window2/3)).mean()) /pp.rolling(window2, min_periods=int(window2/3)).std()

def cs_booksize(preA, booksize):
    return booksize * preA.divide(preA.abs().sum(axis=1), axis=0)

def cs_mktneut(alpha):
    return alpha.sub(alpha.mean(axis=1), axis=0)

def cs_expand_balance(preA, alloc_usdt):
    alloc_usdt /= 2
    psum = pd.DataFrame(np.matlib.repmat(preA[preA > 0].sum(axis=1), preA.shape[1], 1).T, index=preA.index, columns=preA.columns)
    nsum = pd.DataFrame(np.matlib.repmat((preA[preA < 0]).abs().sum(axis=1), preA.shape[1], 1).T, index=preA.index, columns=preA.columns)
    preA[(preA > 0) & (psum < alloc_usdt)] *= alloc_usdt / psum[preA > 0][psum < alloc_usdt]
    preA[(preA < 0) & (nsum < alloc_usdt)] *= alloc_usdt / nsum[preA < 0][nsum < alloc_usdt]
    return preA

def cs_poslimit(preA, pos_limit=0.05):
    preA[preA.abs() > pos_limit] = np.sign(preA[preA.abs() > pos_limit]) * pos_limit
    return preA

def check_open_pos():
    open_pos = request_client.get_position_v2()
    long_pos = 0
    long_pos_pnl = 0
    short_pos = 0
    short_pos_pnl = 0
    for i in open_pos:
        if float(i.positionAmt) > 0:
            long_pos += round(float(i.positionAmt) * float(i.markPrice), 2)
            long_pos_pnl += i.unrealizedProfit
        else:
            short_pos += round(float(i.positionAmt) * float(i.markPrice), 2)
            short_pos_pnl += i.unrealizedProfit

    logging.info('open positions: long_pos ' + str(round(long_pos, 2)) + ' long_pos_pnl ' + str(round(long_pos_pnl, 2)))
    logging.info('open positions: short_pos ' + str(round(short_pos, 2)) + ' short_pos_pnl ' + str(round(short_pos_pnl, 2)))
    return


if __name__ == '__main__':
    logging.info('Trading started with interval of ' + str(trade_interval) + 'hour')
    check_open_pos()

    if (not os.path.exists(close_file_path) 
    or not os.path.exists(open_file_path) 
    or not os.path.exists(high_file_path) 
    or not os.path.exists(low_file_path)):
        load_hist_data_all(all_tickers)

    close = pd.read_csv(close_file_path, index_col=[0]).iloc[-lookback:, :]
    open = pd.read_csv(open_file_path, index_col=[0]).iloc[-lookback:, :]
    high = pd.read_csv(high_file_path, index_col=[0]).iloc[-lookback:, :]
    low = pd.read_csv(low_file_path, index_col=[0]).iloc[-lookback:, :]
    while True:
        current_time = datetime.datetime.utcnow()
        if current_time.hour % trade_interval == 0 and current_time.minute == 0 and current_time.second < 1:
            logging.info('utc time: ' + str(current_time))
            try: 
                close, open, high, low = run_strategy(close, open, high, low)
            except Exception as e:
                logging.error(e)
        time.sleep(0.5)
