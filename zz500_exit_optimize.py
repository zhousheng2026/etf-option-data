#!/usr/bin/env python3
"""
中证500ETF期权策略 - 平仓条件优化
使用最优参数：MACD(8,26,9)+BOLL(15,1.5)+背离15+入场50%
"""

import pandas as pd
import numpy as np

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
                'open': float(parts[1].strip()),
                'high': float(parts[2].strip()),
                'low': float(parts[3].strip()),
                'close': float(parts[4].strip()),
            })
        except:
            continue

df_1min = pd.DataFrame(records)
df_1min = df_1min.sort_values('datetime').reset_index(drop=True)
df_1min.set_index('datetime', inplace=True)

# 合成15分钟
df = df_1min.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
}).dropna()

print("="*70)
print("平仓条件优化 - 让盈利奔跑")
print("="*70)

# 计算指标（最优参数）
macd_fast, macd_slow, macd_signal = 8, 26, 9
boll_period, boll_std = 15, 1.5
divergence_lookback = 15
entry_range = 0.50

df['ema_fast'] = df['close'].ewm(span=macd_fast, adjust=False).mean()
df['ema_slow'] = df['close'].ewm(span=macd_slow, adjust=False).mean()
df['macd'] = df['ema_fast'] - df['ema_slow']
df['macd_signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()
df['macd_golden'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
df['macd_dead'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))

df['boll_mid'] = df['close'].rolling(window=boll_period).mean()
df['boll_std'] = df['close'].rolling(window=boll_period).std()
df['boll_upper'] = df['boll_mid'] + boll_std * df['boll_std']
df['boll_lower'] = df['boll_mid'] - boll_std * df['boll_std']
df['boll_range'] = df['boll_upper'] - df['boll_lower']
df['price_to_lower'] = (df['close'] - df['boll_lower']) / df['boll_range']

# 背离检测
df['divergence_bull'] = False
for i in range(divergence_lookback, len(df)):
    window = df.iloc[i-divergence_lookback:i+1]
    price_low = window['low'].min()
    macd_low = window['macd'].min()
    if (df.iloc[i]['low'] <= price_low * 1.001 and 
        df.iloc[i]['macd'] > macd_low * 0.95 and
        df.iloc[i]['macd'] < 0):
        df.loc[df.index[i], 'divergence_bull'] = True

df['near_lower'] = df['price_to_lower'] <= entry_range

def backtest_with_exit(df, capital, commission, exit_type, exit_param=None):
    """
    回测不同平仓条件
    
    exit_type:
    - 'macd_reverse': MACD反向交叉平仓
    - 'middle_cross': 中轨反向平仓
    - 'boll_upper': 触及上轨平仓
    - 'trailing_stop': 移动止损（exit_param=止损比例）
    - 'fixed_profit': 固定止盈（exit_param=止盈比例）
    """
    df = df.dropna()
    position = 0
    equity = capital
    trades = []
    entry_price = 0
    max_price = 0  # 用于移动止损
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 入场信号
        long_signal = row['macd_golden'] and row['near_lower'] and row['divergence_bull']
        
        # 平仓信号判断
        close_signal = False
        
        if position == 1:
            max_price = max(max_price, row['close'])
            
            if exit_type == 'macd_reverse':
                close_signal = row['macd_dead']
            elif exit_type == 'middle_cross':
                close_signal = (row['close'] > row['boll_mid'] and prev_row['close'] <= prev_row['boll_mid'])
            elif exit_type == 'boll_upper':
                close_signal = row['high'] >= row['boll_upper']
            elif exit_type == 'trailing_stop' and exit_param:
                # 移动止损：从最高点回撤exit_param比例
                drawdown = (max_price - row['close']) / max_price
                close_signal = drawdown >= exit_param
            elif exit_type == 'fixed_profit' and exit_param:
                # 固定止盈
                profit_pct = (row['close'] - entry_price) / entry_price
                close_signal = profit_pct >= exit_param
        
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
                'time': entry_time,
                'entry': entry_price,
                'exit': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct * 100,
                'hold_bars': i
            })
            position = 0
    
    return trades, equity

# 测试不同平仓条件
exit_strategies = [
    ('MACD反向交叉', 'macd_reverse', None),
    ('中轨突破', 'middle_cross', None),
    ('触及上轨', 'boll_upper', None),
    ('移动止损2%', 'trailing_stop', 0.02),
    ('移动止损3%', 'trailing_stop', 0.03),
    ('移动止损5%', 'trailing_stop', 0.05),
    ('固定止盈1%', 'fixed_profit', 0.01),
    ('固定止盈2%', 'fixed_profit', 0.02),
    ('固定止盈3%', 'fixed_profit', 0.03),
]

print(f"\n资金: 10000元, 手续费: 4元/笔")
print("-"*70)

results = []
for name, exit_type, param in exit_strategies:
    trades, final_equity = backtest_with_exit(df, 10000, 4, exit_type, param)
    
    if not trades:
        print(f"{name}: 无交易")
        continue
    
    trades_df = pd.DataFrame(trades)
    total = len(trades)
    wins = len(trades_df[trades_df['pnl'] > 0])
    win_rate = wins / total * 100
    total_return = (final_equity - 10000) / 10000 * 100
    avg_pnl = trades_df['pnl'].mean()
    max_pnl = trades_df['pnl'].max()
    
    results.append({
        'name': name,
        'trades': total,
        'win_rate': win_rate,
        'return': total_return,
        'avg_pnl': avg_pnl,
        'max_pnl': max_pnl
    })
    
    print(f"{name}:")
    print(f"  交易{total}次, 胜率{win_rate:.1f}%, 收益率{total_return:.2f}%")
    print(f"  平均盈亏{avg_pnl:.2f}元, 最大盈利{max_pnl:.2f}元")

# 排序
results_sorted = sorted(results, key=lambda x: x['return'], reverse=True)

print(f"\n{'='*70}")
print("【按收益率排序】")
print(f"{'='*70}")
for i, r in enumerate(results_sorted, 1):
    print(f"{i}. {r['name']}: 交易{r['trades']}次, 胜率{r['win_rate']:.1f}%, 收益率{r['return']:.2f}%")

print(f"\n{'='*70}")
