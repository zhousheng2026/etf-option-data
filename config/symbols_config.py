"""
品种配置文件
包含监测品种列表和相关信息
"""

# ==================== 监测品种列表 ====================

# 主要监测品种（期货合约）
MONITORING_SYMBOLS = [
    # 上期能源
    'sc2604',  # 原油
    'ec2604',  # 集运指数
    
    # 上期所
    'bu2606',  # 沥青
    'fu2605',  # 燃料油
    'rb2610',  # 螺纹钢
    'hc2610',  # 热卷
    
    # 大商所
    'pg2604',  # 液化石油气
    'pp2605',  # 聚丙烯
    'i2609',   # 铁矿石
    'j2609',   # 焦炭
    'jm2609',  # 焦煤
    
    # 郑商所
    'ta2605',  # PTA
    'sm2605',  # 锰硅
    'sa2605',  # 纯碱
    'ma2605',  # 甲醇
    
    # 广期所
    'px2605',  # 对二甲苯
]

# ==================== 品种信息映射 ====================

SYMBOL_INFO = {
    'sc2604': {
        'name': '原油',
        'exchange': 'INE',
        'underlying': 'SC',
        'multiplier': 1000,
        'min_tick': 0.1,
        'has_option': True,
    },
    'ec2604': {
        'name': '集运指数',
        'exchange': 'INE',
        'underlying': 'EC',
        'multiplier': 50,
        'min_tick': 0.1,
        'has_option': False,
    },
    'bu2606': {
        'name': '沥青',
        'exchange': 'SHFE',
        'underlying': 'BU',
        'multiplier': 10,
        'min_tick': 2,
        'has_option': True,
    },
    'fu2605': {
        'name': '燃料油',
        'exchange': 'SHFE',
        'underlying': 'FU',
        'multiplier': 10,
        'min_tick': 1,
        'has_option': True,
    },
    'rb2610': {
        'name': '螺纹钢',
        'exchange': 'SHFE',
        'underlying': 'RB',
        'multiplier': 10,
        'min_tick': 1,
        'has_option': True,
    },
    'hc2610': {
        'name': '热卷',
        'exchange': 'SHFE',
        'underlying': 'HC',
        'multiplier': 10,
        'min_tick': 1,
        'has_option': True,
    },
    'pg2604': {
        'name': '液化石油气',
        'exchange': 'DCE',
        'underlying': 'PG',
        'multiplier': 20,
        'min_tick': 1,
        'has_option': True,
    },
    'pp2605': {
        'name': '聚丙烯',
        'exchange': 'DCE',
        'underlying': 'PP',
        'multiplier': 5,
        'min_tick': 1,
        'has_option': True,
    },
    'i2609': {
        'name': '铁矿石',
        'exchange': 'DCE',
        'underlying': 'I',
        'multiplier': 100,
        'min_tick': 0.5,
        'has_option': True,
    },
    'j2609': {
        'name': '焦炭',
        'exchange': 'DCE',
        'underlying': 'J',
        'multiplier': 100,
        'min_tick': 0.5,
        'has_option': True,
    },
    'jm2609': {
        'name': '焦煤',
        'exchange': 'DCE',
        'underlying': 'JM',
        'multiplier': 60,
        'min_tick': 0.5,
        'has_option': True,
    },
    'ta2605': {
        'name': 'PTA',
        'exchange': 'CZCE',
        'underlying': 'TA',
        'multiplier': 5,
        'min_tick': 2,
        'has_option': True,
    },
    'sm2605': {
        'name': '锰硅',
        'exchange': 'CZCE',
        'underlying': 'SM',
        'multiplier': 5,
        'min_tick': 2,
        'has_option': True,
    },
    'sa2605': {
        'name': '纯碱',
        'exchange': 'CZCE',
        'underlying': 'SA',
        'multiplier': 20,
        'min_tick': 1,
        'has_option': True,
    },
    'ma2605': {
        'name': '甲醇',
        'exchange': 'CZCE',
        'underlying': 'MA',
        'multiplier': 10,
        'min_tick': 1,
        'has_option': True,
    },
    'px2605': {
        'name': '对二甲苯',
        'exchange': 'GFEX',
        'underlying': 'PX',
        'multiplier': 5,
        'min_tick': 1,
        'has_option': False,
    },
}

# ==================== 交易所映射 ====================

EXCHANGE_MAP = {
    'INE': '上期能源',
    'SHFE': '上期所',
    'DCE': '大商所',
    'CZCE': '郑商所',
    'GFEX': '广期所',
    'CFFEX': '中金所',
}

# ==================== 交易时间配置 ====================

TRADING_HOURS = {
    # 日盘
    'day_start': '09:00:00',
    'day_end': '15:00:00',
    
    # 夜盘
    'night_start': '21:00:00',
    'night_end': '02:30:00',  # 部分品种到次日01:00
    
    # 夜盘品种
    'night_symbols': ['sc', 'fu', 'bu', 'rb', 'hc', 'i', 'j', 'jm', 'ma'],
}
