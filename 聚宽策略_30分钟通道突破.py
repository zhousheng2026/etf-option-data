# 聚宽版本 - 股指期权30分钟通道突破策略
# 策略名称：OptionTrendBurst_JQ
# 作者：AI Assistant
# 日期：2026-02-25

from jqdata import *
import pandas as pd
import numpy as np
from datetime import datetime, time

# 初始化函数
def initialize(context):
    # 设置参数
    g.channel_period = 20      # 通道周期（20根K线）
    g.volume_threshold = 1.3   # 成交量阈值
    g.body_threshold = 1.5     # K线实体阈值
    
    # 时间过滤
    g.no_entry_start = time(9, 30)   # 9:30不入场
    g.no_entry_end = time(10, 0)     # 10:00恢复
    g.no_new_position_start = time(14, 30)  # 14:30不开新仓
    g.no_new_position_end = time(15, 0)     # 15:00收盘
    
    # 标的：50ETF期权（用ETF代替，实际交易期权合约）
    g.underlying = '510050.XSHG'  # 50ETF
    
    # 状态
    g.position = None  # 当前持仓
    g.klines = []      # K线缓存
    
    # 定时运行（每30分钟）
    run_interval(context, interval=30, unit='m')
    
    log.info('策略初始化完成')
    log.info('标的：50ETF期权')
    log.info('通道周期：20根30分钟K线')

# 主循环
def handle_data(context, data):
    # 获取当前时间
    current_time = context.current_dt.time()
    current_dt = context.current_dt
    
    # 获取K线数据
    hist = get_bars(g.underlying, count=50, unit='30m', 
                    fields=['open', 'high', 'low', 'close', 'volume'])
    
    if len(hist) < g.channel_period + 1:
        return
    
    # 计算通道
    upper, lower, middle = calculate_channel(hist[:-1])
    
    # 当前K线
    current = hist.iloc[-1]
    current_price = current['close']
    
    log.info(f'时间：{current_dt}, 价格：{current_price:.4f}, 上轨：{upper:.4f}, 下轨：{lower:.4f}')
    
    # 检查持仓
    if g.position:
        # 检查离场信号
        if check_exit_signal(g.position, current_price, middle, lower, upper):
            # 平仓
            close_position(context)
            g.position = None
    else:
        # 时间过滤：14:30-15:00不开新仓
        if g.no_new_position_start <= current_time <= g.no_new_position_end:
            return
        
        # 时间过滤：9:30-10:00不入场
        if g.no_entry_start <= current_time <= g.no_entry_end:
            return
        
        # 检查入场信号
        signal = check_entry_signal(hist, upper, lower)
        if signal:
            # 开仓
            open_position(context, signal)
            g.position = {
                'direction': signal,
                'entry_price': current_price,
                'entry_time': current_dt
            }

# 计算通道
def calculate_channel(df):
    recent = df.tail(g.channel_period)
    upper = recent['high'].max()
    lower = recent['low'].min()
    middle = (upper + lower) / 2
    return upper, lower, middle

# 检查入场信号
def check_entry_signal(df, upper, lower):
    current = df.iloc[-1]
    prev = df.iloc[:-1]
    
    # 计算平均成交量和实体
    avg_volume = prev.tail(g.channel_period)['volume'].mean()
    avg_body = (prev.tail(g.channel_period)['high'] - 
                prev.tail(g.channel_period)['low']).mean()
    
    current_volume = current['volume']
    current_body = current['high'] - current['low']
    
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
    body_ratio = current_body / avg_body if avg_body > 0 else 0
    
    close_price = current['close']
    
    # 做多信号
    if (close_price > upper and 
        volume_ratio > g.volume_threshold and 
        body_ratio > g.body_threshold):
        log.info(f'做多信号：突破{upper:.4f}，成交量{volume_ratio:.2f}倍，实体{body_ratio:.2f}倍')
        return 'LONG'
    
    # 做空信号
    if (close_price < lower and 
        volume_ratio > g.volume_threshold and 
        body_ratio > g.body_threshold):
        log.info(f'做空信号：跌破{lower:.4f}，成交量{volume_ratio:.2f}倍，实体{body_ratio:.2f}倍')
        return 'SHORT'
    
    return None

# 检查离场信号
def check_exit_signal(position, current_price, middle, lower, upper):
    if position['direction'] == 'LONG':
        if current_price < middle:
            log.info(f'做多平仓：价格{current_price:.4f}跌破中轨{middle:.4f}')
            return True
    
    if position['direction'] == 'SHORT':
        if current_price > middle:
            log.info(f'做空平仓：价格{current_price:.4f}突破中轨{middle:.4f}')
            return True
    
    return False

# 开仓
def open_position(context, direction):
    # 这里用ETF代替期权，实际应该交易期权合约
    if direction == 'LONG':
        order_target(g.underlying, 10000)  # 买入10000股
        log.info('买入开仓')
    else:
        order_target(g.underlying, -10000)  # 融券卖出（如果支持）
        log.info('卖出开仓')

# 平仓
def close_position(context):
    order_target(g.underlying, 0)  # 平仓
    log.info('平仓')

# 盘后处理
def after_trading_end(context):
    log.info('盘后处理')
    # 可以在这里保存数据或发送报告
