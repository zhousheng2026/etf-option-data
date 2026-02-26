#!/usr/bin/env python3
"""
中证500ETF期权策略 - 15分钟周期参数优化
"""

import pandas as pd
import numpy as np
from datetime import datetime
import itertools

# 读取1分钟数据并合成15分钟
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

# 合成15分钟数据
df = df_1min.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

print("="*70)
print("中证500ETF (159922) - 15分钟周期参数优化")
print("="*70)
print(f"数据时间范围: {df.index.min()} 至 {df.index.max()}")
print(f"15分钟数据条数: {len(df)} 条")

def calculate_indicators(df, macd_fast, macd_slow, macd_signal, boll_period, boll_std):
    """计算技术指标"""
    df = df.copy()
    
    # MACD
    df['ema_fast'] = df['close'].ewm(span=macd_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=macd_slow, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()
    
    df['macd_golden'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_dead'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    
    # BOLL
    df['boll_mid'] = df['close'].rolling(window=boll_period).mean()
    df['boll_std'] = df['close'].rolling(window=boll_period).std()
    df['boll_upper'] = df['boll_mid'] + boll_std * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - boll_std * df['boll_std']
    
    df['boll_range'] = df['boll_upper'] - df['boll_lower']
    df['price_to_lower'] = (df['close'] - df['boll_lower']) / df['boll_range']
    
    return df

def detect_divergence(df, lookback):
    """检测背离"""
    df = df.copy()
    df['divergence_bull'] = False
    df['divergence_bear'] = False
    
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i+1]
        
        # 底背离
        price_low = window['low'].min()
        macd_low = window['macd'].min()
        if (df.iloc[i]['low'] <= price_low * 1.001 and 
            df.iloc[i]['macd'] > macd_low * 0.95 and
            df.iloc[i]['macd'] < 0):
            df.loc[df.index[i], 'divergence_bull'] = True
        
        # 顶背离
        price_high = window['high'].max()
        macd_high = window['macd'].max()
        if (df.iloc[i]['high'] >= price_high * 0.999 and 
            df.iloc[i]['macd'] < macd_high * 1.05 and
            df.iloc[i]['macd'] > 0):
            df.loc[df.index[i], 'divergence_bear'] = True
    
    return df

def backtest(df, capital, commission, entry_range):
    """回测"""
    df = df.dropna()
    
    df['near_lower'] = df['price_to_lower'] <= entry_range
    df['near_upper'] = df['price_to_lower'] >= (1 - entry_range)
    
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
            trades.append({'pnl': pnl})
            position = 0
            
        elif position == -1 and close_short:
            exit_price = row['close']
            pnl_pct = (entry_price - exit_price) / entry_price
            pnl = capital * pnl_pct * 0.4 - commission
            equity += pnl
            trades.append({'pnl': pnl})
            position = 0
    
    return trades, equity

# 简化参数网格
param_grid = {
    'macd_fast': [8, 12],
    'macd_slow': [26, 30],
    'macd_signal': [9, 12],
    'boll_period': [15, 20],
    'boll_std': [1.5, 2.0],
    'divergence_lookback': [15, 20],
    'entry_range': [0.30, 0.40, 0.50]  # 入场范围30%/40%/50%
}

# 生成参数组合
combinations = []
for fast in param_grid['macd_fast']:
    for slow in param_grid['macd_slow']:
        if fast >= slow:
            continue
        for signal in param_grid['macd_signal']:
            for boll_p in param_grid['boll_period']:
                for boll_s in param_grid['boll_std']:
                    for div_lb in param_grid['divergence_lookback']:
                        for entry_r in param_grid['entry_range']:
                            combinations.append({
                                'macd_fast': fast,
                                'macd_slow': slow,
                                'macd_signal': signal,
                                'boll_period': boll_p,
                                'boll_std': boll_s,
                                'divergence_lookback': div_lb,
                                'entry_range': entry_r
                            })

print(f"\n参数组合总数: {len(combinations)}")
print("\n开始参数优化...")

results = []
for i, params in enumerate(combinations):
    try:
        df_test = calculate_indicators(
            df, 
            params['macd_fast'], 
            params['macd_slow'], 
            params['macd_signal'],
            params['boll_period'],
            params['boll_std']
        )
        df_test = detect_divergence(df_test, params['divergence_lookback'])
        
        trades, final_equity = backtest(df_test, capital=10000, commission=4, entry_range=params['entry_range'])
        
        if len(trades) >= 3:  # 至少3次交易
            wins = sum(1 for t in trades if t['pnl'] > 0)
            win_rate = wins / len(trades)
            total_return = (final_equity - 10000) / 10000
            
            results.append({
                'params': params,
                'trades': len(trades),
                'wins': wins,
                'win_rate': win_rate,
                'return': total_return,
                'final': final_equity
            })
        
        if (i + 1) % 50 == 0:
            print(f"  已完成 {i+1}/{len(combinations)}")
            
    except Exception as e:
        continue

print(f"\n✓ 优化完成，有效结果: {len(results)} 组")

# 排序并显示Top 10
results_sorted = sorted(results, key=lambda x: x['win_rate'], reverse=True)

print(f"\n{'='*70}")
print("【胜率最高的Top 10参数组合】")
print(f"{'='*70}")

for i, r in enumerate(results_sorted[:10], 1):
    p = r['params']
    print(f"\n排名 {i}:")
    print(f"  MACD({p['macd_fast']},{p['macd_slow']},{p['macd_signal']}), "
          f"BOLL({p['boll_period']},{p['boll_std']}), "
          f"背离回看{p['divergence_lookback']}, 入场范围{p['entry_range']*100:.0f}%")
    print(f"  交易{r['trades']}次, 胜率{r['win_rate']*100:.2f}%, 收益率{r['return']*100:.2f}%")

# 按交易次数排序（胜率>50%且交易次数>=5）
filtered = [r for r in results if r['win_rate'] > 0.5 and r['trades'] >= 5]
if filtered:
    filtered_sorted = sorted(filtered, key=lambda x: x['win_rate'], reverse=True)
    print(f"\n{'='*70}")
    print("【胜率>50%且交易次数>=5的参数组合】")
    print(f"{'='*70}")
    for i, r in enumerate(filtered_sorted[:5], 1):
        p = r['params']
        print(f"\n排名 {i}:")
        print(f"  MACD({p['macd_fast']},{p['macd_slow']},{p['macd_signal']}), "
              f"BOLL({p['boll_period']},{p['boll_std']}), "
              f"背离回看{p['divergence_lookback']}, 入场范围{p['entry_range']*100:.0f}%")
        print(f"  交易{r['trades']}次, 胜率{r['win_rate']*100:.2f}%, 收益率{r['return']*100:.2f}%")

print(f"\n{'='*70}")
print("参数优化完成!")
print(f"{'='*70}")
