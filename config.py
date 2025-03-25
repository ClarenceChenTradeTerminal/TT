# Paths
path = "/tt_data/ttshared/"
path_output = '/data/market_maker/p&l_png'
path_alpha = '/data/market_maker/alpha'
path_feat = '/data/market_maker/feature_test/results/'
path_feat_allen = '/data/market_maker/feature_test_allen/results/'
path_corr = '/data/market_maker/corr'
# Industry Map
industry_map = {
    'BTCUSDT': 'Store of Value',
    'ETHUSDT': 'Smart Contract Platform',
    'BNBUSDT': 'Other',
    'XRPUSDT': 'Payments / Cross-border',
    'ADAUSDT': 'Smart Contract Platform',
    'DOGEUSDT': 'Other',
    'SOLUSDT': 'Smart Contract Platform',
    'TRXUSDT': 'Smart Contract Platform',
    'LTCUSDT': 'Store of Value',
    'DOTUSDT': 'Blockchain Interoperability',
    'AVAXUSDT': 'Smart Contract Platform',
    'ATOMUSDT': 'Blockchain Interoperability',
    'UNIUSDT': 'Decentralized Finance (DeFi)',
    'ETCUSDT': 'Smart Contract Platform',
    'XLMUSDT': 'Payments / Cross-border',
    'BCHUSDT': 'Store of Value',
    'ICPUSDT': 'Smart Contract Platform',
    'LDOUSDT': 'Decentralized Finance (DeFi)',
    'FILUSDT': 'Data',
    'APTUSDT': 'Smart Contract Platform',
    'HBARUSDT': 'Other',
    'VETUSDT': 'Other',
    'ARBUSDT': 'Layer 2 Scaling Solution',
    'NEARUSDT': 'Smart Contract Platform',
    'QNTUSDT': 'Blockchain Interoperability',
    'GRTUSDT': 'Data',
    'APEUSDT': 'NFT',
    'ALGOUSDT': 'Smart Contract Platform',
    'SANDUSDT': 'NFT',
    'EOSUSDT': 'Smart Contract Platform',
    'EGLDUSDT': 'Smart Contract Platform',
    'AAVEUSDT': 'Decentralized Finance (DeFi)',
    'OPUSDT': 'Layer 2 Scaling Solution',
    'FTMUSDT': 'Smart Contract Platform',
    'MANAUSDT': 'NFT',
    'XTZUSDT': 'Smart Contract Platform',
    'THETAUSDT': 'Other',
    'STXUSDT': 'Smart Contract Platform',
    'CFXUSDT': 'Smart Contract Platform',
    'AXSUSDT': 'NFT',
    'FLOWUSDT': 'Smart Contract Platform',
    'NEOUSDT': 'Smart Contract Platform',
    'CHZUSDT': 'NFT',
    'CRVUSDT': 'Decentralized Finance (DeFi)',
    'IMXUSDT': 'NFT',
    'MKRUSDT': 'Decentralized Finance (DeFi)',
    'SNXUSDT': 'Decentralized Finance (DeFi)',
    'INJUSDT': 'Decentralized Finance (DeFi)',
    'LINKUSDT': 'Other'
}
# Chain Map
chain_map = {
    'BTCUSDT': 'btc',
    'ETHUSDT': 'eth',
    'BNBUSDT': 'bnb',
    'XRPUSDT': 'xrp',
    'ADAUSDT': 'ada',
    'DOGEUSDT': 'doge',
    'SOLUSDT': 'sol',
    'TRXUSDT': 'trx',
    'LTCUSDT': 'ltc',
    'DOTUSDT': 'dot',
    'AVAXUSDT': 'avax',
    'ATOMUSDT': 'atom',
    'UNIUSDT': 'eth',
    'ETCUSDT': 'eth',
    'XLMUSDT': 'xrp',
    'BCHUSDT': 'btc',
    'ICPUSDT': 'icp',
    'LDOUSDT': 'eth',
    'FILUSDT': 'fil',
    'APTUSDT': 'apt',
    'HBARUSDT': 'hbar',
    'VETUSDT': 'vet',
    'ARBUSDT': 'eth',
    'NEARUSDT': 'near',
    'QNTUSDT': 'eth',
    'GRTUSDT': 'eth',
    'APEUSDT': 'eth',
    'ALGOUSDT': 'aglo',
    'SANDUSDT': 'eth',
    'EOSUSDT': 'eos',
    'EGLDUSDT': 'egld',
    'AAVEUSDT': 'eth',
    'OPUSDT': 'eth',
    'FTMUSDT': 'ftm',
    'MANAUSDT': 'eth',
    'XTZUSDT': 'xtz',
    'THETAUSDT': 'eth',
    'STXUSDT': 'stx',
    'CFXUSDT': 'cfx',
    'AXSUSDT': 'eth',
    'FLOWUSDT': 'flow',
    'NEOUSDT': 'neo',
    'CHZUSDT': 'chz',
    'CRVUSDT': 'eth',
    'IMXUSDT': 'imx',
    'MKRUSDT': 'eth',
    'SNXUSDT': 'eth',
    'INJUSDT': 'inj',
    'LINKUSDT': 'eth'
}
# Universe Lists
universe_top1 = ['BTCUSDT']
universe_top2 = ['BTCUSDT', 'ETHUSDT']
universe_top5 = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT']
universe_top10 = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'TRXUSDT', 'LTCUSDT', 'DOTUSDT']
universe_top20 = ['NEARUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'SOLUSDT', 'TRXUSDT', 'LTCUSDT', 'DOTUSDT', 'AVAXUSDT', 'ATOMUSDT', 'UNIUSDT', 'XMRUSDT', 'ETCUSDT', 'XLMUSDT', 'BCHUSDT', 'FILUSDT', 'HBARUSDT', 'LINKUSDT']
universe_top50 = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'SOLUSDT', 'TRXUSDT', 'LTCUSDT', 'DOTUSDT', 'AVAXUSDT', 'ATOMUSDT', 'UNIUSDT', 'ETCUSDT', 'XLMUSDT', 'BCHUSDT', 'ICPUSDT', 'LDOUSDT', 'FILUSDT', 'APTUSDT', 'HBARUSDT', 'VETUSDT', 'ARBUSDT', 'NEARUSDT', 'QNTUSDT', 'GRTUSDT', 'APEUSDT', 'ALGOUSDT', 'SANDUSDT', 'EOSUSDT', 'EGLDUSDT', 'AAVEUSDT', 'OPUSDT', 'FTMUSDT', 'MANAUSDT', 'XTZUSDT', 'THETAUSDT', 'STXUSDT', 'CFXUSDT', 'AXSUSDT', 'FLOWUSDT', 'NEOUSDT', 'CHZUSDT', 'CRVUSDT', 'IMXUSDT', 'MKRUSDT', 'SNXUSDT', 'INJUSDT', 'LINKUSDT']
# Add other universe lists as needed
factor_map = {
    '1m': 24 * 60
}