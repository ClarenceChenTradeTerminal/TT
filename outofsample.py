import sourcedefender
import sys
sys.path.append("/tt_data/ttshared/")
from tqdm import tqdm

import multiprocessing
from functools import partial
from research_functions import *

from sklearn.linear_model import (
    LinearRegression, Ridge, RidgeCV, Lasso, LassoCV,
    ElasticNet, ElasticNetCV, BayesianRidge, ARDRegression,
    SGDRegressor, PassiveAggressiveRegressor, HuberRegressor,
    TheilSenRegressor, RANSACRegressor
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor

import lightgbm as lgb


# configs = {
#             # 'btc':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/BTCUSDT/btcusdt_lightgbm.pkl', 'threshold': 0.001},
#             # 'bnb':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/BNBUSDT/bnbusdt_lightgbm.pkl', 'threshold': 0.001},

#             'celo':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/LIGHTGBM_HL/CELOUSDT.pkl', 'threshold': 0.001}, #  相对百分比：0.1%
#             # 'crv':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/CRVUSDT/crvusdt_lightgbm.pkl', 'threshold': 0.001},

#             # 'dgb':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/DGBUSDT/dgbusdt_lightgbm.pkl', 'threshold': 0.001},
#             # 'eth':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/ETHUSDT/ethusdt_lightgbm.pkl', 'threshold': 0.001},
#             # 'edu':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/EDUUSDT/eduusdt_lightgbm.pkl', 'threshold': 0.001},

#             # 'flm':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/FLMUSDT/flmusdt_lightgbm.pkl', 'threshold': 0.001},
#             # 'kas':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/KASUSDT/kasusdt_lightgbm.pkl', 'threshold': 0.001},
#             'lina':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/LIGHTGBM_HL/LINAUSDT.pkl', 'threshold': 0.001},
#             # 'magic':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/MAGICUSDT/magicusdt_lightgbm.pkl', 'threshold': 0.001},
#             'movr':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/LIGHTGBM_HL/MOVRUSDT.pkl', 'threshold': 0.001},
#             # 'ntrn':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/NTRNUSDT/ntrnusdt_lightgbm.pkl', 'threshold': 0.001},
            
#             # 'ogn':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/OGNUSDT/ognusdt_lightgbm.pkl', 'threshold': 0.001},
#             # 'ong':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/ONGUSDT/ongusdt_lightgbm.pkl', 'threshold': 0.001},

#             'stmx':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/LIGHTGBM_HL/STMXUSDT.pkl', 'threshold': 0.001},
#             # 'tao':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/TAOUSDT/taousdt_lightgbm.pkl', 'threshold': 0.001},
#             # 'xem':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/XEMUSDT/xemusdt_lightgbm.pkl', 'threshold': 0.001},
#             # 'xtz':{ 'normalize_function': 'normalization(x, 5*24*60)', 'model_path': f'/data/market_maker/models/XTZUSDT/xtzusdt_lightgbm.pkl', 'threshold': 0.001}
            
#             }

model_file = f'/data/market_maker/models'



# model_file = f'/tt_data/zchen/models'

configs = {
            # 'KAS':{ 'normalize_function': 'normalization(x, 3*24*60)', 'model_path': '/tt_data/zchen/models/KAS.pkl', 'threshold': 0.000125},
            'ORDI':{ 'normalize_function': 'normalization(x, 1*24*60)', 'model_path': '/tt_data/zchen/models/ORDI.pkl', 'threshold': 0.0005},
            # 'TIA':{ 'normalize_function': 'normalization(x, 1*24*60)', 'model_path': '/tt_data/zchen/models/TIA.pkl', 'threshold': 0.001},
            # 'WIF':{ 'normalize_function': 'normalization(x, 1*24*60)', 'model_path': '/tt_data/zchen/models/WIF.pkl', 'threshold': 0.0005},
            # 'WLD':{ 'normalize_function': 'normalization(x, 1*24*60)', 'model_path': '/tt_data/zchen/models/WLD.pkl', 'threshold': 0.00025}
          }
            


# op_porteval(configs, freq='1M', slippage=0.0003)

op_porteval(configs, freq='1M', slippage=0.0005, out_sample = True)