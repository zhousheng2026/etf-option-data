#!/usr/bin/env python3
"""
中证500ETF期权策略回测 - 多周期对比
使用1分钟数据合成5/15/30/60分钟
"""

import pandas as pd
import numpy as np
from datetime import datetime

# 读取1分钟数据
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
                'open': float(parts[1].strip()),
                'high': float(parts[2].strip()),
                'low': float(parts[3].strip()),
                'close': float(parts[4].strip()),
                'volume': float(parts[5].strip()),
            })
        except:
            continue

df_1min = pd.DataFrame(records)
df_1min = df_1min.sort_values('datetime').reset_index(drop=True)
df_1min.set_index('datetime', inplace=True)

print("="*70)
print("中证500ETF (159922) 多周期策略回测")
print("="*70)
print(f"\n1分钟数据时间范围: {df_1min.index.min()} 至 {df_1min.index.max()}")
print(f"1分钟数据条数: {len(df_1min)} 条")

def resample_data(df, period):
    """重采样到指定周期"""
    return df.resample(period).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

def calculate_indicators(df):
    """计算技术指标"""
    df = df.copy()
    
    # MACD
    fast, slow, signal = 12, 26, 9
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    
    df['macd_golden'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_dead'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    
    # BOLL
    period, std_dev = 20, 2.0
    df['boll_mid'] = df['close'].rolling(window=period).mean()
    df['boll_std'] = df['close'].rolling(window=period).std()
    df['boll_upper'] = df['boll_mid'] + std_dev * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - std_dev * df['boll_std']
    
    df['boll_range'] = df['boll_upper'] - df['boll_lower']
    df['price_to_lower'] = (df['close'] - df['boll_lower']) / df['boll_range']
    df['near_lower'] = df['price_to_lower'] <= 0.40
    df['near_upper'] = df['price_to_lower'] >= 0.60
    
    # 底背离检测
    df['divergence_bull'] = False
    df['divergence_bear'] = False
    lookback = 20
    
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i+1]
        
        price_low = window['low'].min()
        macd_low = window['macd'].min()
        if (df.iloc[i]['low'] <= price_low * 1.001 and 
            df.iloc[i]['macd'] > macd_low * 0.95 and
            df.iloc[i]['macd'] < 0):
            df.loc[df.index[i], 'divergence_bull'] = True
        
        price_high = window['high'].max()
        macd_high = window['macd'].max()
        if (df.iloc[i]['high'] >= price_high * 0.999 and 
            df.iloc[i]['macd'] < macd_high * 1.05 and
            df.iloc[i]['macd'] > 0):
            df.loc[df.index[i], 'divergence_bear'] = True
    
    return df

def backtest(df, capital=10000, commission=4):
    """回测函数"""
    df = calculate_indicators(df)
    df = df.dropna()
    
    position = 0
    equity = capital
    trades = []
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        long_signal = row['macd_golden'] and row['near_lower'] and row['divergence_bull']
        short_signal = row['macd_dead'] and row['near_upper'] and row['divergence_bear']
        
        close_long = row['macd_dead'] or (row['close'] > row['boll_mid'] and df.iloc[i-1]['close'] <= df.iloc[i-1]['boll_mid'])
        close_short = row['macd_golden'] or (row['close'] < row['boll_mid'] and df.iloc[i-1]['close'] >= df.iloc[i-1]['boll_mid'])
        
        if position == 0:
            if long_signal:
                position = 1
                entry_price = row['close']
                entry_time = row.name
            elif short_signal:
                position = -1
                entry_price = row['close']
                entry_time = row.name
        
        elif position == 1 and close_long:
            exit_price = row['close']
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl = capital * pnl_pct * 0.4 - commission
            equity += pnl
            
            trades.append({
                'time': entry_time,
                'direction': 'LONG',
                'entry': entry_price,
                'exit': exit_price,
                'pnl': pnl,
            })
            position = 0
            
        elif position == -1 and close_short:
            exit_price = row['close']
            pnl_pct = (entry_price - exit_price) / entry_price
            pnl = capital * pnl_pct * 0.4 - commission
            equity += pnl
            
            trades.append({
                'time': entry_time,
                'direction': 'SHORT',
                'entry': entry_price,
                'exit': exit_price,
                'pnl': pnl,
            })
            position = 0
    
    return trades, equity

# 测试不同周期
periods = {
    '5min': '5min',
    '15min': '15min',
    '30min': '30min',
    '60min': '60min'
}

results = {}

for name, period in periods.items():
    print(f"\n{'='*70}")
    print(f"【{name} 周期回测】")
    print(f"{'='*70}")
    
    df_period = resample_data(df_1min, period)
    print(f"数据条数: {len(df_period)} 条")
    
    trades, final_equity = backtest(df_period, capital=10000)
    
    if not trades:
        print(f"无交易信号")
        continue
    
    trades_df = pd.DataFrame(trades)
    total = len(trades)
    wins = len(trades_df[trades_df['pnl'] > 0])
    win_rate = wins / total * 100
    total_return = (final_equity - 10000) / 10000 * 100
    
    results[name] = {
        'trades': total,
        'win_rate': win_rate,
        'return': total_return,
        'final': final_equity
    }
    
    print(f"\n资金 10000元:")
    print(f"  交易次数: {total}")
    print(f"  盈利次数: {wins}")
    print(f"  胜率: {win_rate:.2f}%")
    print(f"  总收益率: {total_return:.2f}%")
    print(f"  最终资金: {final_equity:.2f}元")

# 汇总
print(f"\n{'='*70}")
print("【多周期对比汇总】")
print(f"{'='*70}")
print(f"{'周期':<10} {'交易次数':<10} {'胜率':<10} {'收益率':<10}")
print("-"*70)
for name, r in results.items():
    print(f"{name:<10} {r['trades']:<10} {r['win_rate']:.2f}%    {r['return']:.2f}%")

print(f"\n{'='*70}")
print("回测完成!")
print(f"{'='*70}")
