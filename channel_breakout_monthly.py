#!/usr/bin/env python3
"""
通道突破策略 - 多周期回测（月化收益率）
"""

import pandas as pd
import numpy as np
from datetime import datetime

# 读取数据
data_lines = open('/root/openclaw/kimi/downloads/19c8f5bd-03a2-8f41-8000-0000977f629a_159922.txt', 'r', encoding='gbk').readlines()

records = []
for line in data_lines[2:]:
    line = line.strip()
    if not line or '时间' in line:
        continue
    parts = line.split('\t')
    if len(parts) >= 6:
        try:
            dt = pd.to_datetime(parts[0].strip(), format='%Y/%m/%d-%H:%M')
            records.append({
                'datetime': dt,
                'open': float(parts[1]),
                'high': float(parts[2]),
                'low': float(parts[3]),
                'close': float(parts[4]),
            })
        except:
            continue

df_1min = pd.DataFrame(records)
df_1min = df_1min.sort_values('datetime').reset_index(drop=True)
df_1min.set_index('datetime', inplace=True)

print("="*70)
print("通道突破策略 - 月化收益率")
print("="*70)

def resample_data(df, period):
    """重采样"""
    return df.resample(period).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }).dropna()

def calculate_trend_lines(df, lookback=20):
    """计算趋势线"""
    df = df.copy()
    df['trend_high'] = df['high'].rolling(window=lookback).max()
    df['trend_low'] = df['low'].rolling(window=lookback).min()
    df['desc_trend'] = df['high'].shift(1) > df['high'].shift(2)
    df['asc_trend'] = df['low'].shift(1) > df['low'].shift(2)
    return df

def backtest_channel_breakout(df, capital=10000, commission=4):
    """通道突破策略回测"""
    # 计算MACD
    fast, slow, signal = 12, 26, 9
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_golden'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    
    df = calculate_trend_lines(df)
    df = df.dropna()
    
    position = 0
    equity = capital
    trades = []
    entry_price = 0
    max_price = 0
    
    for i in range(2, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        prev2_row = df.iloc[i-2]
        
        desc_channel = (prev2_row['high'] > prev_row['high'])
        break_desc = row['close'] > prev_row['high']
        long_signal = desc_channel and break_desc and row['macd_golden']
        
        if position == 1:
            asc_trend_line = prev_row['low'] + (prev_row['low'] - prev2_row['low'])
            break_asc = row['close'] < min(asc_trend_line, prev_row['low'] * 0.995)
            max_price = max(max_price, row['close'])
            trailing_stop = row['close'] < max_price * 0.97
            close_signal = break_asc or trailing_stop
        
        if position == 0 and long_signal:
            position = 1
            entry_price = row['close']
            entry_time = row.name
            max_price = entry_price
            
        elif position == 1 and close_signal:
            exit_price = row['close']
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl = capital * pnl_pct * 0.4 - commission
            equity += pnl
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': row.name,
                'entry': entry_price,
                'exit': exit_price,
                'pnl': pnl,
            })
            position = 0
    
    return trades, equity

# 测试不同周期
periods = {
    '5min': '5min',
    '10min': '10min', 
    '15min': '15min',
    '30min': '30min',
    '60min': '60min'
}

# 数据时间范围
data_start = df_1min.index.min()
data_end = df_1min.index.max()
total_days = (data_end - data_start).days
total_months = total_days / 30.44  # 平均每月天数

print(f"\n数据时间范围: {data_start.strftime('%Y-%m-%d')} 至 {data_end.strftime('%Y-%m-%d')}")
print(f"总天数: {total_days}天, 约{total_months:.1f}个月")
print(f"\n资金: 10000元, 手续费: 4元/笔")
print("="*70)

results = []
for name, period in periods.items():
    df_period = resample_data(df_1min, period)
    trades, final_equity = backtest_channel_breakout(df_period)
    
    if not trades:
        continue
    
    trades_df = pd.DataFrame(trades)
    total = len(trades)
    wins = len(trades_df[trades_df['pnl'] > 0])
    win_rate = wins / total * 100
    
    # 总收益率
    total_return = (final_equity - 10000) / 10000
    
    # 月化收益率（复利）
    # (1 + 总收益率)^(1/月数) - 1
    monthly_return = (1 + total_return) ** (1 / total_months) - 1
    monthly_return_pct = monthly_return * 100
    
    # 年化收益率
    annual_return = (1 + monthly_return) ** 12 - 1
    annual_return_pct = annual_return * 100
    
    results.append({
        'period': name,
        'trades': total,
        'win_rate': win_rate,
        'total_return': total_return * 100,
        'monthly_return': monthly_return_pct,
        'annual_return': annual_return_pct
    })

# 打印结果
print(f"\n{'周期':<8} {'交易次数':<10} {'胜率':<10} {'总收益率':<12} {'月化收益率':<12} {'年化收益率':<12}")
print("-"*70)
for r in results:
    print(f"{r['period']:<8} {r['trades']:<10} {r['win_rate']:.1f}%      {r['total_return']:.2f}%       {r['monthly_return']:.2f}%       {r['annual_return']:.2f}%")

print(f"\n{'='*70}")
print("说明：月化收益率 = (1 + 总收益率)^(1/月数) - 1")
print("      年化收益率 = (1 + 月化收益率)^12 - 1")
print(f"{'='*70}")
