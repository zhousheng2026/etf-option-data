"""
聚宽平台 - 三重底背离策略
布林带底背离 + MACD底背离 + 价格底背离
买入：至少2种底背离同时出现
卖出：30分钟通道跌破中轨
只做多，不做空
"""

# 导入函数库
from jqdata import *
import numpy as np
import pandas as pd
from datetime import datetime, time

# 初始化函数，设定基准等等
def initialize(context):
    # 设定基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式
    set_option('use_real_price', True)
    # 输出内容到日志 log.info()
    log.info('初始函数开始运行且全局只运行一次')
    
    # 策略参数
    context.channel_bars = 20      # 通道周期
    context.bb_period = 20         # 布林带周期
    context.bb_std = 2.0           # 布林带标准差
    context.macd_fast = 12         # MACD快线
    context.macd_slow = 26         # MACD慢线
    context.macd_signal = 9        # MACD信号线
    context.divergence_lookback = 30  # 背离检测回溯周期
    context.min_divergence_bars = 5   # 最小背离间距
    context.stop_loss_pct = 0.30      # 止损比例
    context.cooldown_bars = 5         # 冷却期
    
    # 标的配置 - 7个ETF期权标的
    context.symbols = [
        '588000.XSHG',  # 科创50ETF - 首选
        '588080.XSHG',  # 科创板50ETF
        '510500.XSHG',  # 中证500ETF
        '159845.XSHE',  # 中证1000ETF
        '159915.XSHE',  # 创业板ETF
        '510050.XSHG',  # 50ETF
        '510300.XSHG',  # 300ETF
    ]
    
    # 当前持仓状态
    context.position = None  # None 或 symbol
    context.entry_price = 0
    context.cooldown_counter = 0
    
    # 运行频率：每30分钟运行一次
    run_interval(context, interval_mins=30, reference_security='000300.XSHG')


def run_interval(context, interval_mins=30, reference_security='000300.XSHG'):
    """设置定时运行"""
    run_daily(trade, time='09:30')
    run_daily(trade, time='10:00')
    run_daily(trade, time='10:30')
    run_daily(trade, time='11:00')
    run_daily(trade, time='13:00')
    run_daily(trade, time='13:30')
    run_daily(trade, time='14:00')
    run_daily(trade, time='14:30')


def get_minute_data(context, symbol, count=100):
    """获取分钟数据"""
    try:
        df = get_price(symbol, count=count, frequency='30m', 
                       fields=['open', 'high', 'low', 'close', 'volume'])
        return df
    except:
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
    if context.position is not None:
        return False, 0, []
    
    if context.cooldown_counter > 0:
        return False, 0, []
    
    # 时间过滤
    current_time = datetime.now().time()
    if time(9, 30) <= current_time <= time(10, 0):
        return False, 0, []
    if time(14, 30) <= current_time <= time(15, 0):
        return False, 0, []
    
    # 检测三种背离
    bb_div = detect_bollinger_divergence(df, idx, context.divergence_lookback, context.min_divergence_bars)
    macd_div = detect_macd_divergence(df, idx, context.divergence_lookback, context.min_divergence_bars)
    price_div = detect_price_divergence(df, idx, context.divergence_lookback, context.min_divergence_bars)
    
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
    if context.position is None:
        return False, ""
    
    current_price = df['close'].iloc[idx]
    
    # 止损检查
    pnl_pct = (current_price - context.entry_price) / context.entry_price
    
    if pnl_pct <= -context.stop_loss_pct:
        return True, f"Stop Loss: {pnl_pct:.2%}"
    
    # 通道跌破检查 - 使用20根K线通道
    if idx >= context.channel_bars:
        highs = df['high'].iloc[idx-context.channel_bars:idx].values
        lows = df['low'].iloc[idx-context.channel_bars:idx].values
        closes = df['close'].iloc[idx-context.channel_bars:idx].values
        
        # 简化的通道中轨计算
        middle = (np.max(highs) + np.min(lows)) / 2
        
        if current_price < middle:
            return True, f"Channel Breakdown: Below middle ({current_price:.2f} < {middle:.2f})"
    
    return False, ""


def trade(context):
    """主交易函数"""
    # 更新冷却计数器
    if context.cooldown_counter > 0:
        context.cooldown_counter -= 1
    
    # 检查当前持仓的离场条件
    if context.position is not None:
        df = get_minute_data(context, context.position, count=50)
        if df is not None and len(df) > context.channel_bars:
            df = calculate_bollinger_bands(df, context.bb_period, context.bb_std)
            df = calculate_macd(df, context.macd_fast, context.macd_slow, context.macd_signal)
            
            should_exit, exit_reason = check_exit_signal(context, df, len(df)-1)
            
            if should_exit:
                # 卖出期权（这里简化处理，实际应该找到对应的期权合约）
                log.info(f'卖出信号: {exit_reason}')
                context.position = None
                context.cooldown_counter = context.cooldown_bars
    
    # 检查入场条件
    if context.position is None:
        for symbol in context.symbols:
            df = get_minute_data(context, symbol, count=50)
            if df is None or len(df) < 40:
                continue
            
            # 计算指标
            df = calculate_bollinger_bands(df, context.bb_period, context.bb_std)
            df = calculate_macd(df, context.macd_fast, context.macd_slow, context.macd_signal)
            
            should_buy, strength, divergences = check_entry_signal(context, df, len(df)-1)
            
            if should_buy:
                div_str = '+'.join(divergences)
                log.info(f'买入信号: {symbol}, 背离: {div_str}, 强度: {strength}')
                
                # 记录持仓
                context.position = symbol
                context.entry_price = df['close'].iloc[-1]
                
                # 实际交易中这里需要：
                # 1. 获取对应的期权合约（认购期权，虚值2-3档）
                # 2. 买入期权
                break


def after_trading_end(context):
    """盘后处理"""
    # 打印当日持仓信息
    if context.position:
        log.info(f'盘后持仓: {context.position}')
