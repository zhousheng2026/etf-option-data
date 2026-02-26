"""
股指期权趋势爆发策略 - 米筐(RiceQuant)版本
纯做多版本（只买入看涨期权）
"""
import numpy as np
from rqalpha.api import *
from rqalpha import run_func

# 策略参数
CHANNEL_BARS = 20
VOLUME_THRESHOLD = 1.3
BODY_THRESHOLD = 1.5
STOP_LOSS_1 = 0.30
STOP_LOSS_2 = 0.50
COOLDOWN_BARS = 3

def init(context):
    """初始化"""
    context.symbol = '510300.XSHG'  # 沪深300ETF
    context.channel_bars = CHANNEL_BARS
    context.volume_threshold = VOLUME_THRESHOLD
    context.body_threshold = BODY_THRESHOLD
    context.cooldown = 0
    context.position = False
    context.entry_price = 0
    
    set_benchmark(context.symbol)
    set_slippage(FixedSlippage(0.002))
    subscribe(context.symbol, '30m')
    
    logger.info("纯做多策略初始化完成")

def handle_bar(context, bar_dict):
    """主逻辑"""
    symbol = context.symbol
    
    try:
        hist = history_bars(symbol, context.channel_bars + 10, '30m',
                           fields=['open', 'high', 'low', 'close', 'volume'])
    except:
        return
    
    if len(hist) < context.channel_bars:
        return
    
    if context.cooldown > 0:
        context.cooldown -= 1
    
    current_bar = bar_dict[symbol]
    current_price = current_bar.close
    current_time = context.now
    
    # 时间过滤
    hour, minute = current_time.hour, current_time.minute
    time_val = hour * 100 + minute
    
    if 930 <= time_val <= 1000:
        return
    if 1430 <= time_val <= 1500 and not context.position:
        return
    
    # 绘制通道
    channel = draw_channel(hist)
    
    # 检查离场
    if context.position:
        check_exit(context, current_price, channel)
    else:
        check_entry(context, current_price, channel, hist, current_bar)

def draw_channel(hist):
    """绘制通道"""
    highs = hist['high'][-CHANNEL_BARS:]
    lows = hist['low'][-CHANNEL_BARS:]
    closes = hist['close'][-CHANNEL_BARS:]
    
    upper = np.max(highs)
    lower = np.min(lows)
    middle = (upper + lower) / 2
    
    return {'upper': upper, 'lower': lower, 'middle': middle}

def check_entry(context, price, channel, hist, current_bar):
    """只做多：突破上轨"""
    if price <= channel['upper']:
        return
    
    # 计算条件
    avg_volume = np.mean(hist['volume'][-20:])
    volume_ratio = current_bar.volume / avg_volume if avg_volume > 0 else 0
    
    avg_body = np.mean([abs(hist['close'][i] - hist['open'][i]) for i in range(-5, 0)])
    body = abs(current_bar.close - current_bar.open)
    body_ratio = body / avg_body if avg_body > 0 else 0
    
    conditions = 1  # 突破
    if volume_ratio >= context.volume_threshold:
        conditions += 1
    if body_ratio >= context.body_threshold:
        conditions += 1
    
    if conditions >= 2:
        weights = {2: 0.30, 3: 0.50, 4: 0.70}
        weight = weights.get(conditions, 0.30)
        
        cash = context.portfolio.available_cash
        amount = int(cash * weight / price / 100) * 100
        
        if amount >= 100:
            order_shares(context.symbol, amount)
            context.position = True
            context.entry_price = price
            logger.info(f"做多入场: 价格{price:.3f}, 数量{amount}")

def check_exit(context, price, channel):
    """多头离场"""
    pnl_pct = (price - context.entry_price) / context.entry_price
    
    # 止损
    if pnl_pct <= -STOP_LOSS_2:
        order_target_value(context.symbol, 0)
        context.position = False
        context.cooldown = COOLDOWN_BARS
        logger.info(f"止损离场: 价格{price:.3f}, 盈亏{pnl_pct:.2%}")
        return
    
    # 跌破通道
    if price < channel['upper']:
        if price < channel['middle']:
            order_target_value(context.symbol, 0)
            context.position = False
            context.cooldown = COOLDOWN_BARS
            logger.info(f"跌破中轨离场: 价格{price:.3f}")
        else:
            # 减仓50%
            current_value = context.portfolio.positions[context.symbol].value
            order_target_value(context.symbol, current_value * 0.5)
            logger.info(f"跌破上轨减仓: 价格{price:.3f}")

def after_trading(context):
    record(total_value=context.portfolio.total_value)

config = {
    "base": {
        "start_date": "2023-01-01",
        "end_date": "2025-02-25",
        "frequency": "1d",
        "accounts": {"stock": 1000000}
    },
    "extra": {"log_level": "verbose"},
    "mod": {"sys_analyzer": {"enabled": True, "plot": True}}
}

if __name__ == "__main__":
    run_func(init=init, handle_bar=handle_bar, config=config)
