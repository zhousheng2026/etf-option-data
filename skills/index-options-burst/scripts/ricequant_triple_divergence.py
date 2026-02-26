"""
米筐(RiceQuant)平台 - 三重底背离策略
布林带底背离 + MACD底背离 + 价格底背离
买入：至少2种底背离同时出现
卖出：30分钟通道跌破中轨
只做多，不做空
"""

from rqalpha.api import *
import numpy as np
import pandas as pd
from datetime import datetime, time

# 策略参数（可在米筐平台界面配置）
CHANNEL_BARS = 20      # 通道周期
BB_PERIOD = 20         # 布林带周期
BB_STD = 2.0           # 布林带标准差
MACD_FAST = 12         # MACD快线
MACD_SLOW = 26         # MACD慢线
MACD_SIGNAL = 9        # MACD信号线
DIVERGENCE_LOOKBACK = 30  # 背离检测回溯周期
MIN_DIVERGENCE_BARS = 5   # 最小背离间距
STOP_LOSS_PCT = 0.30      # 止损比例
COOLDOWN_BARS = 5         # 冷却期

# 标的配置 - 7个ETF期权标的（按优先级排序）
SYMBOLS = [
    '588000.XSHG',  # 科创50ETF - 首选
    '588080.XSHG',  # 科创板50ETF
    '510500.XSHG',  # 中证500ETF
    '159845.XSHE',  # 中证1000ETF
    '159915.XSHE',  # 创业板ETF
    '510050.XSHG',  # 50ETF
    '510300.XSHG',  # 300ETF
]


def init(context):
    """初始化函数"""
    logger.info("三重底背离策略初始化")
    
    # 订阅标的
    for symbol in SYMBOLS:
        subscribe(symbol)
    
    # 状态变量
    context.position_symbol = None  # 当前持仓标的
    context.entry_price = 0         # 入场价格
    context.cooldown_counter = 0    # 冷却计数器
    context.option_contract = None  # 期权合约
    
    # 设置运行频率（30分钟）
    scheduler.run_daily(trade, time_rule=market_open(minute=0))
    scheduler.run_daily(trade, time_rule=market_open(minute=30))
    scheduler.run_daily(trade, time_rule=market_open(minute=60))
    scheduler.run_daily(trade, time_rule=market_open(minute=90))
    scheduler.run_daily(trade, time_rule=market_open(minute=120))
    scheduler.run_daily(trade, time_rule=market_open(minute=150))
    scheduler.run_daily(trade, time_rule=market_open(minute=180))
    scheduler.run_daily(trade, time_rule=market_open(minute=210))


def get_minute_data(context, symbol, count=50):
    """获取分钟数据"""
    try:
        # 获取历史数据
        bars = history_bars(symbol, count, '30m', 
                           fields=['open', 'high', 'low', 'close', 'volume'])
        if bars is None or len(bars) < count // 2:
            return None
        
        df = pd.DataFrame(bars)
        return df
    except Exception as e:
        logger.error(f"获取数据失败 {symbol}: {e}")
        return None


def calculate_bollinger_bands(df, period=20, std=2.0):
    """计算布林带"""
    df = df.copy()
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    bb_std = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_middle'] + std * bb_std
    df['bb_lower'] = df['bb_middle'] - std * bb_std
    return df


def calculate_macd(df, fast=12, slow=26, signal=9):
    """计算MACD"""
    df = df.copy()
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df


def find_local_lows(prices, lookback=3):
    """寻找局部低点"""
    lows = []
    for i in range(lookback, len(prices) - lookback):
        window = prices[i-lookback:i+lookback+1]
        if prices[i] == np.min(window):
            lows.append(i)
    return lows


def detect_bollinger_divergence(df, idx, lookback=30, min_bars=5):
    """检测布林带底背离"""
    if idx < lookback + 20:
        return False
    
    prices = df['close'].iloc[idx-lookback:idx+1].values
    bb_lower = df['bb_lower'].iloc[idx-lookback:idx+1].values
    
    local_lows = find_local_lows(prices, lookback=3)
    
    if len(local_lows) < 2:
        return False
    
    recent_low_idx = local_lows[-1]
    prev_low_idx = local_lows[-2]
    
    if recent_low_idx - prev_low_idx < min_bars:
        return False
    
    price_change = (prices[recent_low_idx] - prices[prev_low_idx]) / prices[prev_low_idx]
    bb_change = (bb_lower[recent_low_idx] - bb_lower[prev_low_idx]) / bb_lower[prev_low_idx]
    
    # 底背离：价格跌，布林带相对强
    if price_change < -0.002 and bb_change > price_change * 0.5:
        return True
    
    return False


def detect_macd_divergence(df, idx, lookback=30, min_bars=5):
    """检测MACD底背离"""
    if idx < lookback + 26:
        return False
    
    prices = df['close'].iloc[idx-lookback:idx+1].values
    macd = df['macd'].iloc[idx-lookback:idx+1].values
    
    local_lows = find_local_lows(prices, lookback=3)
    
    if len(local_lows) < 2:
        return False
    
    recent_low_idx = local_lows[-1]
    prev_low_idx = local_lows[-2]
    
    if recent_low_idx - prev_low_idx < min_bars:
        return False
    
    price_change = (prices[recent_low_idx] - prices[prev_low_idx]) / prices[prev_low_idx]
    macd_change = macd[recent_low_idx] - macd[prev_low_idx]
    
    # 底背离：价格跌，MACD上升
    if price_change < -0.002 and macd_change > 0:
        return True
    
    return False


def detect_price_divergence(df, idx, lookback=30, min_bars=5):
    """检测价格底背离（成交量）"""
    if idx < lookback + 5:
        return False
    
    prices = df['close'].iloc[idx-lookback:idx+1].values
    volumes = df['volume'].iloc[idx-lookback:idx+1].values
    
    local_lows = find_local_lows(prices, lookback=3)
    
    if len(local_lows) < 2:
        return False
    
    recent_low_idx = local_lows[-1]
    prev_low_idx = local_lows[-2]
    
    if recent_low_idx - prev_low_idx < min_bars:
        return False
    
    price_change = (prices[recent_low_idx] - prices[prev_low_idx]) / prices[prev_low_idx]
    volume_change = (volumes[recent_low_idx] - volumes[prev_low_idx]) / volumes[prev_low_idx]
    
    # 底背离：价格跌，成交量萎缩
    if price_change < -0.002 and volume_change < -0.05:
        return True
    
    return False


def check_entry_signal(context, df, idx):
    """检查买入信号"""
    if context.position_symbol is not None:
        return False, 0, []
    
    if context.cooldown_counter > 0:
        return False, 0, []
    
    # 时间过滤
    current_time = context.now.time() if hasattr(context, 'now') else datetime.now().time()
    if time(9, 30) <= current_time <= time(10, 0):
        return False, 0, []
    if time(14, 30) <= current_time <= time(15, 0):
        return False, 0, []
    
    # 检测三种背离
    bb_div = detect_bollinger_divergence(df, idx, DIVERGENCE_LOOKBACK, MIN_DIVERGENCE_BARS)
    macd_div = detect_macd_divergence(df, idx, DIVERGENCE_LOOKBACK, MIN_DIVERGENCE_BARS)
    price_div = detect_price_divergence(df, idx, DIVERGENCE_LOOKBACK, MIN_DIVERGENCE_BARS)
    
    divergences = []
    if bb_div:
        divergences.append('Bollinger')
    if macd_div:
        divergences.append('MACD')
    if price_div:
        divergences.append('Price')
    
    if len(divergences) >= 2:
        strength = 2 if len(divergences) == 3 else 1
        return True, strength, divergences
    
    return False, 0, []


def check_exit_signal(context, df, idx):
    """检查卖出信号"""
    if context.position_symbol is None:
        return False, ""
    
    current_price = df['close'].iloc[idx]
    
    # 止损检查
    pnl_pct = (current_price - context.entry_price) / context.entry_price
    
    if pnl_pct <= -STOP_LOSS_PCT:
        return True, f"Stop Loss: {pnl_pct:.2%}"
    
    # 通道跌破检查
    if idx >= CHANNEL_BARS:
        highs = df['high'].iloc[idx-CHANNEL_BARS:idx].values
        lows = df['low'].iloc[idx-CHANNEL_BARS:idx].values
        
        middle = (np.max(highs) + np.min(lows)) / 2
        
        if current_price < middle:
            return True, f"Channel Breakdown: {current_price:.2f} < {middle:.2f}"
    
    return False, ""


def trade(context, bar_dict):
    """主交易函数"""
    # 更新冷却计数器
    if context.cooldown_counter > 0:
        context.cooldown_counter -= 1
    
    # 检查当前持仓的离场条件
    if context.position_symbol is not None:
        df = get_minute_data(context, context.position_symbol, count=50)
        if df is not None and len(df) > CHANNEL_BARS:
            df = calculate_bollinger_bands(df, BB_PERIOD, BB_STD)
            df = calculate_macd(df, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            
            should_exit, exit_reason = check_exit_signal(context, df, len(df)-1)
            
            if should_exit:
                logger.info(f'卖出信号: {context.position_symbol}, 原因: {exit_reason}')
                
                # 平仓期权
                if context.option_contract:
                    order_target(context.option_contract, 0)
                    context.option_contract = None
                
                context.position_symbol = None
                context.cooldown_counter = COOLDOWN_BARS
    
    # 检查入场条件
    if context.position_symbol is None:
        for symbol in SYMBOLS:
            df = get_minute_data(context, symbol, count=50)
            if df is None or len(df) < 40:
                continue
            
            # 计算指标
            df = calculate_bollinger_bands(df, BB_PERIOD, BB_STD)
            df = calculate_macd(df, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            
            should_buy, strength, divergences = check_entry_signal(context, df, len(df)-1)
            
            if should_buy:
                div_str = '+'.join(divergences)
                current_price = df['close'].iloc[-1]
                
                logger.info(f'买入信号: {symbol}, 背离: {div_str}, 强度: {strength}, 价格: {current_price:.2f}')
                
                # 记录持仓
                context.position_symbol = symbol
                context.entry_price = current_price
                
                # TODO: 在实际交易中，需要：
                # 1. 查询对应的期权合约列表
                # 2. 选择虚值2-3档的认购期权
                # 3. 买入期权
                # context.option_contract = get_option_contract(symbol, current_price)
                # order_value(context.option_contract, context.portfolio.available_cash * (0.3 if strength == 1 else 0.5))
                
                break


def after_trading(context):
    """盘后处理"""
    if context.position_symbol:
        logger.info(f'盘后持仓: {context.position_symbol}')
