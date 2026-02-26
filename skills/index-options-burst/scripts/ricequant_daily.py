"""
股指期权趋势爆发策略 - 米筐(RiceQuant)版本
日线20日高低点突破版本
"""
import numpy as np
from rqalpha.api import *
from rqalpha import run_func

# 策略参数
LOOKBACK = 20      # 20日高低点
EXIT_MA = 10       # 10日均线离场
VOLUME_THRESHOLD = 1.2
IV_RANK_MAX = 50
STOP_LOSS_1 = 0.30
STOP_LOSS_2 = 0.50

def init(context):
    """初始化"""
    context.symbol = '510300.XSHG'
    context.lookback = LOOKBACK
    context.exit_ma = EXIT_MA
    context.position = False
    context.entry_price = 0
    
    set_benchmark(context.symbol)
    set_slippage(FixedSlippage(0.002))
    
    logger.info("日线版本策略初始化完成")

def handle_bar(context, bar_dict):
    """主逻辑 - 日线"""
    symbol = context.symbol
    
    try:
        hist = history_bars(symbol, context.lookback + 20, '1d',
                           fields=['open', 'high', 'low', 'close', 'volume'])
    except:
        return
    
    if len(hist) < context.lookback:
        return
    
    current_bar = bar_dict[symbol]
    current_price = current_bar.close
    
    # 检查离场
    if context.position:
        check_exit(context, current_price, hist)
    else:
        check_entry(context, current_price, hist, current_bar)

def check_entry(context, price, hist, current_bar):
    """日线入场：突破20日高点"""
    high_20 = np.max(hist['high'][-context.lookback:])
    
    if price <= high_20:
        return
    
    # 成交量条件
    avg_volume = np.mean(hist['volume'][-20:])
    volume_ratio = current_bar.volume / avg_volume if avg_volume > 0 else 0
    
    # MACD简化判断
    if len(hist) >= 26:
        ema12 = np.mean(hist['close'][-12:])
        ema26 = np.mean(hist['close'][-26:])
        macd_positive = ema12 > ema26
    else:
        macd_positive = price > np.mean(hist['close'][-10:])
    
    conditions = 1  # 突破
    if volume_ratio >= VOLUME_THRESHOLD:
        conditions += 1
    if macd_positive:
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
            logger.info(f"日线突破入场: 价格{price:.3f}, 数量{amount}")

def check_exit(context, price, hist):
    """日线离场：跌破10日均线"""
    pnl_pct = (price - context.entry_price) / context.entry_price
    
    # 止损
    if pnl_pct <= -STOP_LOSS_2:
        order_target_value(context.symbol, 0)
        context.position = False
        logger.info(f"止损离场: 价格{price:.3f}, 盈亏{pnl_pct:.2%}")
        return
    
    if pnl_pct <= -STOP_LOSS_1:
        order_target_value(context.symbol, 0)
        context.position = False
        logger.info(f"止损离场(30%): 价格{price:.3f}, 盈亏{pnl_pct:.2%}")
        return
    
    # 跌破10日均线
    if len(hist) >= context.exit_ma:
        ma10 = np.mean(hist['close'][-context.exit_ma:])
        if price < ma10:
            order_target_value(context.symbol, 0)
            context.position = False
            logger.info(f"跌破MA10离场: 价格{price:.3f}, MA10={ma10:.3f}")

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
