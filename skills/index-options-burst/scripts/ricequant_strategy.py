"""
股指期权趋势爆发策略 - 米筐(RiceQuant)版本
双向版本（做多+做空）
"""
import numpy as np
import pandas as pd
from rqalpha.api import *
from rqalpha import run_func

# 策略参数
CHANNEL_BARS = 20      # 通道K线数
VOLUME_THRESHOLD = 1.3 # 成交量倍数
BODY_THRESHOLD = 1.5   # K线实体倍数
IV_RANK_MAX_LONG = 55  # 做多IV上限
IV_RANK_MAX_SHORT = 65 # 做空IV上限
STOP_LOSS_1 = 0.30     # 第一次止损
STOP_LOSS_2 = 0.50     # 第二次止损
COOLDOWN_BARS = 3      # 冷却期

def init(context):
    """初始化策略"""
    context.symbol = '510300.XSHG'  # 沪深300ETF
    context.channel_bars = CHANNEL_BARS
    context.volume_threshold = VOLUME_THRESHOLD
    context.body_threshold = BODY_THRESHOLD
    context.cooldown = 0
    context.position = 0  # 0=空仓, 1=多头, -1=空头
    context.entry_price = 0
    context.trades = []  # 记录交易
    
    # 设置基准和滑点
    set_benchmark(context.symbol)
    set_slippage(FixedSlippage(0.002))  # 固定滑点0.2%
    
    # 订阅30分钟K线
    subscribe(context.symbol, '30m')
    
    logger.info("策略初始化完成")

def handle_bar(context, bar_dict):
    """主策略逻辑，每30分钟运行一次"""
    symbol = context.symbol
    
    # 获取历史数据
    try:
        hist = history_bars(symbol, context.channel_bars + 10, '30m', 
                           fields=['open', 'high', 'low', 'close', 'volume'])
    except:
        return
    
    if len(hist) < context.channel_bars:
        return
    
    # 更新冷却期
    if context.cooldown > 0:
        context.cooldown -= 1
    
    # 获取当前bar
    current_bar = bar_dict[symbol]
    current_price = current_bar.close
    current_time = context.now
    
    # 时间过滤
    hour = current_time.hour
    minute = current_time.minute
    time_val = hour * 100 + minute
    
    # 9:30-10:00 不入场
    if 930 <= time_val <= 1000:
        return
    
    # 14:30-15:00 不开新仓
    if 1430 <= time_val <= 1500 and context.position == 0:
        return
    
    # 绘制通道
    channel = draw_channel(hist)
    
    # 检查离场
    if context.position != 0:
        check_exit(context, current_price, channel, hist)
    
    # 检查入场
    if context.position == 0 and context.cooldown == 0:
        check_entry(context, current_price, channel, hist, current_bar)

def draw_channel(hist):
    """绘制趋势通道"""
    highs = hist['high'][-CHANNEL_BARS:]
    lows = hist['low'][-CHANNEL_BARS:]
    closes = hist['close'][-CHANNEL_BARS:]
    
    # 简单通道：最高点和最低点
    upper = np.max(highs)
    lower = np.min(lows)
    middle = (upper + lower) / 2
    
    return {
        'upper': upper,
        'lower': lower,
        'middle': middle,
        'type': 'flat' if (upper - lower) / closes[-1] < 0.02 else 'trend'
    }

def check_entry(context, price, channel, hist, current_bar):
    """检查入场信号"""
    # 计算指标
    avg_volume = np.mean(hist['volume'][-20:])
    volume_ratio = current_bar.volume / avg_volume if avg_volume > 0 else 0
    
    avg_body = np.mean([abs(hist['close'][i] - hist['open'][i]) for i in range(-5, 0)])
    body = abs(current_bar.close - current_bar.open)
    body_ratio = body / avg_body if avg_body > 0 else 0
    
    # 做多信号：突破上轨
    if price > channel['upper']:
        conditions = 1
        if volume_ratio >= context.volume_threshold:
            conditions += 1
        if body_ratio >= context.body_threshold:
            conditions += 1
        
        if conditions >= 2:
            # 计算仓位
            weights = {2: 0.30, 3: 0.50, 4: 0.70}
            weight = weights.get(conditions, 0.30)
            
            # 买入ETF（模拟期权，用ETF代替）
            cash = context.portfolio.available_cash
            amount = int(cash * weight / price / 100) * 100  # 调整为100的倍数
            
            if amount >= 100:
                order_shares(context.symbol, amount)
                context.position = 1
                context.entry_price = price
                logger.info(f"做多入场: 价格{price:.3f}, 数量{amount}")
                return
    
    # 做空信号：跌破下轨
    if price < channel['lower']:
        conditions = 1
        if volume_ratio >= context.volume_threshold:
            conditions += 1
        if body_ratio >= context.body_threshold:
            conditions += 1
        
        if conditions >= 2:
            # 融券卖出（如果支持）
            # 这里用反向ETF或空仓代替
            logger.info(f"做空信号(未执行): 价格{price:.3f}")

def check_exit(context, price, channel, hist):
    """检查离场信号"""
    if context.position == 1:  # 多头
        # 计算盈亏
        pnl_pct = (price - context.entry_price) / context.entry_price
        
        # 止损
        if pnl_pct <= -STOP_LOSS_2:
            order_target_value(context.symbol, 0)
            context.position = 0
            context.cooldown = COOLDOWN_BARS
            logger.info(f"止损离场: 价格{price:.3f}, 盈亏{pnl_pct:.2%}")
            return
        
        # 通道跌破
        if price < channel['upper']:
            if price < channel['middle']:
                order_target_value(context.symbol, 0)
                context.position = 0
                context.cooldown = COOLDOWN_BARS
                logger.info(f"跌破中轨离场: 价格{price:.3f}")
            else:
                # 减仓50%
                current_value = context.portfolio.positions[context.symbol].value
                order_target_value(context.symbol, current_value * 0.5)
                logger.info(f"跌破上轨减仓: 价格{price:.3f}")

def after_trading(context):
    """盘后处理"""
    # 记录每日净值
    record(total_value=context.portfolio.total_value)

# 回测配置
config = {
    "base": {
        "start_date": "2023-01-01",
        "end_date": "2025-02-25",
        "frequency": "1d",
        "accounts": {
            "stock": 1000000
        }
    },
    "extra": {
        "log_level": "verbose",
    },
    "mod": {
        "sys_analyzer": {
            "enabled": True,
            "plot": True
        }
    }
}

if __name__ == "__main__":
    run_func(init=init, handle_bar=handle_bar, config=config)
