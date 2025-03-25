import sourcedefender
import sys
sys.path.append("/tt_data/ttshared/")
from alpha_func_test import *
import multiprocessing

'''
"feature_list": ["open", "high", "low", "close", "volume", "return", "RSI_14", "RSI_7", "STOCHk_14_3_3", "STOCHd_14_3_3", "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3", "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9", "PPO_12_26_9", "PPOh_12_26_9", "PPOs_12_26_9", "PVO_12_26_9", "PVOh_12_26_9", "PVOs_12_26_9", "AO_5_34", "MFI_14", "CMF_20", "OBV", "AD", "EOM_14_100000000", "ROC_10", "MOM_10", "CCI_20_0.015", "WILLR_14", "AROOND_14", "AROONU_14", "AROONOSC_14", "CHOP_14_1_100", "ADX_14", "DMP_14", "DMN_14", "TRIX_14_9", "TRIXs_14_9", "KST_10_15_20_30_10_10_10_15", "KSTs_9", "TSI_13_25_13", "TSIs_13_25_13", "FISHERT_9_1", "FISHERTs_9_1", "UO_7_14_28", "RVI_14", "ATRr_14", "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0", "KCLe_20_2", "KCBe_20_2", "KCUe_20_2", "DCL_20_20", "DCM_20_20", "DCU_20_20", "PSARl_0.02_0.2", "PSARs_0.02_0.2", "PSARaf_0.02_0.2", "PSARr_0.02_0.2", "SUPERT_10_3.0", "SUPERTd_10_3.0", "SUPERTl_10_3.0", "SUPERTs_10_3.0", "ISA_9", "ISB_26", "ITS_9", "IKS_26", "LR_14", "EMA_80", "EMA_50", "EMA_21", "SMA_200", "SMA_50", "SMA_21", "DEMA_14", "TEMA_14", "WMA_14", "HMA_14", "T3_14_0.7"]
'''

#coins = ['CHESS', 'FLM', '1000CAT', 'STG', 'ICP', 'ZIL', 'COS', 'ETH', 'IDEX', 'ICX', 'RENDER', 'COW', 'SUPER', 'DENT', 'FTT', 'UNFI', 'KEY', 'KNC', 'BOME', 'XMR', 'BLUEBIRD', 'FET', 'RAY', 'BAKE', 'ENJ', 'MANA', 'ADA', 
#'MOVE', 'DAR', 'MDT', 'SCR', 'LEVER', 'NTRN', 'MAVIA', 'OCEAN', 'VOXEL', 'SPELL', 'GLM', 'WAVES', 'HIFI', 'STORJ', 'LOKA', 'KAS', 'SUSHI', 'TIA', 'AR', 'HIGH', 'POL', 'RDNT', 'CVC', 'TRB', 'REEF', 'ALPHA', 'ACE', 'ZRO', 
#'SOL', 'W', 'ZETA', 'ALICE', 'UNI', 'ID', 'CELO', 'EDU', 'BEL', 'GHST', 'XTZ', 'MASK', 'ZEN', 'JASMY', '1000SATS', 'RNDR', 'CETUS', 'SANTOS', 'G', 'BNB', 'OMNI', 'GMT', 'OGN', 'ARKM', 'MBOX', 'KSM', 'IOTA', 'MAGIC', 'SUN', 
#'NEAR', 'GMX', '1000LUNC', 'CTSI', 'EOS', 'BTCDOM', 'ONG', 'DEGEN', 'MANTA', 'JUP', 'BAN', 'CGPT', 'POPCAT', '1MBABYDOGE', 'ATA', 'EGLD', 'MINA', 'BCH', 'AGIX', 'HIVE', 'IOTX', 'SSV', 'SWELL', 'PYTH', 'VELODROME', 'ACH', 'ZEC', 
#'ACX', 'RSR', 'AAVE', 'MAV', 'CHR', 'IMX', 'SFP', 'CELR', 'UXLINK', 'GALA', 'MTL', 'BRETT', 'ME', 'SXP', 'DOGE', 'FXS', 'ALT', 'ACT', 'THE', 'ORDI', 'CHZ', 'KDA', 'METIS', 'DEFI', 'OP', 'SCRT', 'STPT', 'DEXE', 'STRK', 'SC', 'XAI', 
#'FARTCOIN', 'IO', 'FIO', 'VIDT', 'WAXP', 'KOMA', '1000CHEEMS', 'BADGER', 'API3', 'ORCA', '1000000MOG', 'ENS', 'RPL', 'SNT', 'LINK', 'TROY', 'GRASS', 'SPX', 'GRT', 'AKT', 'INJ', 'YFI', 'FTM', 'SNX', 'DGB', 'RAD', 'PHA', 'REN', 'ANKR', 
#'RAYSOL', 'AERO', 'RLC', 'AXS', 'DYDX', 'IOST', 'QNT', 'GAS', 'T', 'SLERF', 'COMBO', 'LUMIA', 'KMNO', 'ONDO', 'TOKEN', 'NULS', 'TWT', 'POLYX', 'REZ', 'CYBER', '1000FLOKI', 'BIO', 'APE', 'BTS', 'TRU', 'SUI', 'HOOK', 'CTK', 'CVX', 'BLZ', 
#'XEM', 'ZRX', 'JOE', 'RIF', 'HFT', 'DIA', 'TLM', 'ARB', 'DOT', '1000BONK', 'LPT', 'CRV', 'USDC', 'BTC', 'XRP', 'ONT', 'LRC', 'COMP', 'VET', 'AIXBT', 'USUAL', 'FLUX', 'AUCTION', 'DODOX', 'BEAMX', 'THETA', 'PEOPLE', 'BOND', 'TNSR', 'DF', 
#'AI16Z', 'DOGS', 'LIT', 'BICO', 'LDO', 'COCOS', 'MORPHO', 'ARPA', 'GTC', 'SAFE', '1000WHY', 'NMR', 'STMX', 'STX', 'ENA', 'KLAY', 'WIF', 'SRM', 'KAVA', 'CHILLGUY', 'BSW', 'DRIFT', 'LINA', 'MOVR', 'VIRTUAL', 'MATIC', 'WOO', 'ETC', 'CKB', 
#'1INCH', 'LQTY', 'LISTA', 'HNT', 'AUDIO', '1000XEC', 'NFP', 'PIXEL', 'MYRO', 'CATI', '1000PEPE', 'AEVO', 'SYS', 'PNUT', 'OXT', 'AI', 'ETHW', 'RARE', 'EIGEN', 'ATOM', 'JTO', '1000RATS', 'PERP', '1000X', 'PENDLE', 'PHB', 'ASTR', 'MKR', 
#'SLP', 'OMG', 'C98', 'AMB', 'ALPACA', 'GLMR', 'MOODENG', 'HMSTR', 'NKN', 'USTC', 'ILV', 'UMA', 'NOT', 'DEGO', 'RUNE', 'BLUR', 'AERGO', 'AXL', 'LUNA2', 'GAL', 'COTI', 'ETHFI', 'CFX', 'BIGTIME', 'DASH', 'RVN', 'HBAR', 'TON', 'BSV', 'LOOM', 
#'DUSK', 'VANRY', 'FOOTBALL', 'FIDA', 'SEI', 'DYM', 'MOCA', 'AGLD', 'RONIN', 'NEIRO', 'SYN', 'BANANA', 'PORTAL', 'REI', 'LSK', 'MEME', 'POWR', 'GRIFFAIN', 'TRX', 'BAND', 'NEO', 'HOT', 'XVS', 'XLM', 'BB', 'ONE', 'FLOW', 'QUICK', 'BAL', 'ALGO', 
#'YGG', 'SKL', 'CAKE', 'AVAX', 'TOMO', 'APT', 'ROSE', 'LTC', 'ARK', 'BNX', 'ZEREBRO', 'ANT', 'FIL', 'STRAX', 'OM', 'GOAT', 'XVG', 'QTUM', 'WLD', 'BAT', 'SAND', 'MEW', 'VANA', 'BNT', 'PONKE', 'SAGA', '1000SHIB', 'TURBO', 'HIPPO', 'NEIROETH', 
#'ORBS', 'TAO', 'AVA', 'STEEM', 'KAIA', 'PENGU', 'ZK']

'''
coin = 'GLM'
normalize_function = 'normalization(x, 2*24*60)'
model_path = '/data/market_maker/models/' + coin + '.pkl'
#feature_str = f"algo_load(data, normalize='{normalize_function}', load_model_pathname='{model_path}')"
feature_str = f"algo_ridge_split(data, normalize='{normalize_function}', test_size=0.3, save_model_pathname='{model_path}')"
op_feval(feature_str, freq='1M', symbol=coin, slippage=0.0002)
'''


configs = {
            'OM':{ 'normalize_function': 'normalization(x, 1*24*60)', 'model_path': '/tt_data/zchen/models/OM.pkl', 'threshold': 0.000125},
            'ORDI':{ 'normalize_function': 'normalization(x, 1*24*60)', 'model_path': '/tt_data/zchen/models/ORDI.pkl', 'threshold': 0.000125},
            'TIA':{ 'normalize_function': 'normalization(x, 1*24*60)', 'model_path': '/tt_data/zchen/models/TIA.pkl', 'threshold': 0.000125},
            'WIF':{ 'normalize_function': 'normalization(x, 1*24*60)', 'model_path': '/tt_data/zchen/models/WIF.pkl', 'threshold': 0.000125},
            'SUI':{ 'normalize_function': 'normalization(x, 1*24*60)', 'model_path': '/tt_data/zchen/models/SUI.pkl', 'threshold': 0.000125}
            }
op_porteval(configs, freq='5M', slippage=0.0002)


'''
import multiprocessing
from functools import partial

#coin_list = ['CHESS', 'FLM', '1000CAT', 'STG', 'ICP', 'ZIL', 'COS', 'ETH', 'IDEX', 'ICX', 'RENDER', 'COW', 'SUPER', 'DENT', 'FTT', 'UNFI', 'KEY', 'KNC', 'BOME', 'XMR', 'BLUEBIRD', 'FET', 'RAY', 'BAKE', 'ENJ', 'MANA', 'ADA', 'MOVE', 'DAR', 'MDT', 'SCR', 'LEVER', 'NTRN', 'MAVIA', 'OCEAN', 'VOXEL', 'SPELL', 'GLM', 'WAVES', 'HIFI', 'STORJ', 'LOKA', 'KAS', 'SUSHI', 'TIA', 'AR', 'HIGH', 'POL', 'RDNT', 'CVC', 'TRB', 'REEF', 'ALPHA', 'ACE', 'ZRO', 'SOL', 'W', 'ZETA', 'ALICE', 'UNI', 'ID', 'CELO', 'EDU', 'BEL', 'GHST', 'XTZ', 'MASK', 'ZEN', 'JASMY', '1000SATS', 'RNDR', 'CETUS', 'SANTOS', 'G', 'BNB', 'OMNI', 'GMT', 'OGN', 'ARKM', 'MBOX', 'KSM', 'IOTA', 'MAGIC', 'SUN', 'NEAR', 'GMX', '1000LUNC', 'CTSI', 'EOS', 'BTCDOM', 'ONG', 'DEGEN', 'MANTA', 'JUP', 'BAN', 'CGPT', 'POPCAT', '1MBABYDOGE', 'ATA', 'EGLD', 'MINA', 'BCH', 'AGIX', 'HIVE', 'IOTX', 'SSV', 'SWELL', 'PYTH', 'VELODROME', 'ACH', 'ZEC', 'ACX', 'RSR', 'AAVE', 'MAV', 'CHR', 'IMX', 'SFP', 'CELR', 'UXLINK', 'GALA', 'MTL', 'BRETT', 'ME', 'SXP', 'DOGE', 'FXS', 'ALT', 'ACT', 'THE', 'ORDI', 'CHZ', 'KDA', 'METIS', 'DEFI', 'OP', 'SCRT', 'STPT', 'DEXE', 'STRK', 'SC', 'XAI', 'FARTCOIN', 'IO', 'FIO', 'VIDT', 'WAXP', 'KOMA', '1000CHEEMS', 'BADGER', 'API3', 'ORCA', '1000000MOG', 'ENS', 'RPL', 'SNT', 'LINK', 'TROY', 'GRASS', 'SPX', 'GRT', 'AKT', 'INJ', 'YFI', 'FTM', 'SNX', 'DGB', 'RAD', 'PHA', 'REN', 'ANKR', 'RAYSOL', 'AERO', 'RLC', 'AXS', 'DYDX', 'IOST', 'QNT', 'GAS', 'T', 'SLERF', 'COMBO', 'LUMIA', 'KMNO', 'ONDO', 'TOKEN', 'NULS', 'TWT', 'POLYX', 'REZ', 'CYBER', '1000FLOKI', 'BIO', 'APE', 'BTS', 'TRU', 'SUI', 'HOOK', 'CTK', 'CVX', 'BLZ', 'XEM', 'ZRX', 'JOE', 'RIF', 'HFT', 'DIA', 'TLM', 'ARB', 'DOT', '1000BONK', 'LPT', 'CRV', 'USDC', 'BTC', 'XRP', 'ONT', 'LRC', 'COMP', 'VET', 'AIXBT', 'USUAL', 'FLUX', 'AUCTION', 'DODOX', 'BEAMX', 'THETA', 'PEOPLE', 'BOND', 'TNSR', 'DF', 'AI16Z', 'DOGS', 'LIT', 'BICO', 'LDO', 'COCOS', 'MORPHO', 'ARPA', 'GTC', 'SAFE', '1000WHY', 'NMR', 'STMX', 'STX', 'ENA', 'KLAY', 'WIF', 'SRM', 'KAVA', 'CHILLGUY', 'BSW', 'DRIFT', 'LINA', 'MOVR', 'VIRTUAL', 'MATIC', 'WOO', 'ETC', 'CKB', '1INCH', 'LQTY', 'LISTA', 'HNT', 'AUDIO', '1000XEC', 'NFP', 'PIXEL', 'MYRO', 'CATI', '1000PEPE', 'AEVO', 'SYS', 'PNUT', 'OXT', 'AI', 'ETHW', 'RARE', 'EIGEN', 'ATOM', 'JTO', '1000RATS', 'PERP', '1000X', 'PENDLE', 'PHB', 'ASTR', 'MKR', 'SLP', 'OMG', 'C98', 'AMB', 'ALPACA', 'GLMR', 'MOODENG', 'HMSTR', 'NKN', 'USTC', 'ILV', 'UMA', 'NOT', 'DEGO', 'RUNE', 'BLUR', 'AERGO', 'AXL', 'LUNA2', 'GAL', 'COTI', 'ETHFI', 'CFX', 'BIGTIME', 'DASH', 'RVN', 'HBAR', 'TON', 'BSV', 'LOOM', 'DUSK', 'VANRY', 'FOOTBALL', 'FIDA', 'SEI', 'DYM', 'MOCA', 'AGLD', 'RONIN', 'NEIRO', 'SYN', 'BANANA', 'PORTAL', 'REI', 'LSK', 'MEME', 'POWR', 'GRIFFAIN', 'TRX', 'BAND', 'NEO', 'HOT', 'XVS', 'XLM', 'BB', 'ONE', 'FLOW', 'QUICK', 'BAL', 'ALGO', 'YGG', 'SKL', 'CAKE', 'AVAX', 'TOMO', 'APT', 'ROSE', 'LTC', 'ARK', 'BNX', 'ZEREBRO', 'ANT', 'FIL', 'STRAX', 'OM', 'GOAT', 'XVG', 'QTUM', 'WLD', 'BAT', 'SAND', 'MEW', 'VANA', 'BNT', 'PONKE', 'SAGA', '1000SHIB', 'TURBO', 'HIPPO', 'NEIROETH', 'ORBS', 'TAO', 'AVA', 'STEEM', 'KAIA', 'PENGU', 'ZK']
coin_list = ["1000BONK", "1000FLOKI", "1000PEPE", "1000SHIB", "1MBABYDOGE", "AAVE", "ACT", "ADA", "AVAX", "BCH", "BNB", "BTC", "CRV", "DOGE", "DOT", "ENA", "ENS", "ETH", "FIL", "FTM", "GOAT", "HBAR", "LINK", "LTC", "MOODENG", "NEIRO", "OM", "ORDI", "PNUT", "SAND", "SOL", "SUI", "TIA", "TRX", "UNI", "WIF", "WLD", "XLM", "XRP"]

def process_coin(coin, feature_str_template):
    model_path = f"/data/market_maker/models/{coin}.pkl"
    for i in range(1, 2):
        feature_str = feature_str_template.format(normalize_function = 'normalization(x, ' + str(i) +'*24*60)', model_path=model_path)
        print(f"Processing {coin} with model path: {model_path}")
        try:
            op_feval(feature_str, freq='5M', symbol=coin.replace('usdt', ''), slippage=0.0002)
        except Exception as e:
            print(f"Error processing {coin}: {e}")

feature_str_template = "algo_gradient_boosting(data, normalize='{normalize_function}', test_size=0.3, n_estimators=5, save_model_pathname='{model_path}')"

process_coin_partial = partial(process_coin, feature_str_template=feature_str_template)
with multiprocessing.Pool(processes=20) as pool:
    pool.map(process_coin_partial, coin_list)
'''

