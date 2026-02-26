#!/usr/bin/env python3
"""
通道突破策略 - 多周期回测
突破下降通道入场，跌破上升趋势线平仓
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
print("通道突破策略 - 多周期回测")
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
    """
    计算趋势线
    下降趋势线：连接近期高点
    上升趋势线：连接近期低点
    """
    df = df.copy()
    df['trend_high'] = df['high'].rolling(window=lookback).max()
    df['trend_low'] = df['low'].rolling(window=lookback).min()
    
    # 下降趋势线斜率（简化：用最近2个高点）
    df['desc_trend'] = df['high'].shift(1) > df['high'].shift(2)
    
    # 上升趋势线斜率
    df['asc_trend'] = df['low'].shift(1) > df['low'].shift(2)
    
    return df

def backtest_channel_breakout(df, capital=10000, commission=4):
    """
    通道突破策略回测
    入场：突破下降趋势线 + MACD金叉
    平仓：跌破上升趋势线
    """
    # 计算MACD
    fast, slow, signal = 12, 26, 9
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_golden'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    
    # 计算趋势线
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
        
        # 下降趋势线突破（简化：前两个高点下降，当前突破）
        desc_channel = (prev2_row['high'] > prev_row['high'])  # 下降
        break_desc = row['close'] > prev_row['high']  # 突破前高
        
        # 入场信号：突破下降通道 + MACD金叉
        long_signal = desc_channel and break_desc and row['macd_golden']
        
        # 上升趋势线（连接低点）
        if position == 1:
            # 简化：用最近两个低点连线
            asc_trend_line = prev_row['low'] + (prev_row['low'] - prev2_row['low'])
            break_asc = row['close'] < min(asc_trend_line, prev_row['low'] * 0.995)
            
            max_price = max(max_price, row['close'])
            trailing_stop = row['close'] < max_price * 0.97  # 移动止损3%
            
            close_signal = break_asc or trailing_stop
        
        # 执行交易
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

print(f"\n资金: 10000元, 手续费: 4元/笔")
print("-"*70)

for name, period in periods.items():
    df_period = resample_data(df_1min, period)
    trades, final_equity = backtest_channel_breakout(df_period)
    
    if not trades:
        print(f"{name}: 无交易")
        continue
    
    trades_df = pd.DataFrame(trades)
    total = len(trades)
    wins = len(trades_df[trades_df['pnl'] > 0])
    win_rate = wins / total * 100
    total_return = (final_equity - 10000) / 10000 * 100
    
    print(f"\n{name}:")
    print(f"  交易{total}次, 胜率{win_rate:.1f}%, 收益率{total_return:.2f}%")
    print(f"  最终资金: {final_equity:.2f}元")

print(f"\n{'='*70}")
