#!/usr/bin/env python3
"""
科创50ETF期权策略回测 - 使用用户提供的通达信数据
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

# 读取数据 - 手动处理
data_lines = open('/root/openclaw/kimi/downloads/19c8f50c-4e32-84a5-8000-0000f5c860a5_588000.txt', 'r', encoding='gbk').readlines()

# 跳过前两行，解析数据
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
                'boll_mid': float(parts[14].strip()) if len(parts) > 14 and parts[14].strip() else np.nan,
                'boll_upper': float(parts[15].strip()) if len(parts) > 15 and parts[15].strip() else np.nan,
                'boll_lower': float(parts[16].strip()) if len(parts) > 16 and parts[16].strip() else np.nan,
                'macd': float(parts[17].strip()) if len(parts) > 17 and parts[17].strip() else np.nan,
                'macd_signal': float(parts[18].strip()) if len(parts) > 18 and parts[18].strip() else np.nan,
            })
        except:
            continue

df = pd.DataFrame(records)
df = df.sort_values('datetime').reset_index(drop=True)

print("="*70)
print("科创50ETF (588000) 期权策略回测")
print("="*70)
print(f"\n数据时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
print(f"数据条数: {len(df)} 条 (30分钟周期)")
print(f"约 {len(df)/8:.1f} 个交易日")

# 删除MACD为0的行（无数据）
df = df[df['macd'] != 0].reset_index(drop=True)
print(f"有效数据: {len(df)} 条")

# 生成信号
print("\n【信号生成】")
print("-"*70)

# MACD金叉/死叉
df['macd_golden'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
df['macd_dead'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))

# 触及布林带极值
df['touch_upper'] = df['high'] >= df['boll_upper']
df['touch_lower'] = df['low'] <= df['boll_lower']

# 底背离检测（简化版：价格新低但MACD抬高）
df['divergence'] = False
lookback = 10
for i in range(lookback, len(df)):
    window = df.iloc[i-lookback:i+1]
    price_low = window['low'].min()
    macd_low = window['macd'].min()
    
    if (df.iloc[i]['low'] <= price_low * 1.001 and 
        df.iloc[i]['macd'] > macd_low * 0.95 and
        df.iloc[i]['macd'] < 0):
        df.loc[df.index[i], 'divergence'] = True

# 信号统计
print(f"MACD金叉次数: {df['macd_golden'].sum()}")
print(f"MACD死叉次数: {df['macd_dead'].sum()}")
print(f"触及上轨次数: {df['touch_upper'].sum()}")
print(f"触及下轨次数: {df['touch_lower'].sum()}")
print(f"底背离信号次数: {df['divergence'].sum()}")

# 策略回测
print("\n【策略回测】")
print("-"*70)

def backtest(df, capital=5000, commission=4):
    """回测函数"""
    position = 0
    equity = capital
    trades = []
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        # 做多信号：MACD金叉 + 触及下轨（简化版，去掉底背离要求）
        long_signal = row['macd_golden'] and row['touch_lower']
        
        # 做空信号：MACD死叉 + 触及上轨
        short_signal = row['macd_dead'] and row['touch_upper']
        
        # 平仓信号
        close_long = row['macd_dead'] or (row['close'] > row['boll_mid'] and df.iloc[i-1]['close'] <= df.iloc[i-1]['boll_mid'])
        close_short = row['macd_golden'] or (row['close'] < row['boll_mid'] and df.iloc[i-1]['close'] >= df.iloc[i-1]['boll_mid'])
        
        if position == 0:
            if long_signal:
                position = 1
                entry_price = row['close']
                entry_time = row['datetime']
            elif short_signal:
                position = -1
                entry_price = row['close']
                entry_time = row['datetime']
        
        elif position == 1 and close_long:
            exit_price = row['close']
            pnl_pct = (exit_price - entry_price) / entry_price
            # 期权Delta约0.4，杠杆效应
            pnl = capital * pnl_pct * 0.4 - commission
            equity += pnl
            
            trades.append({
                'time': entry_time,
                'direction': 'LONG',
                'entry': entry_price,
                'exit': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct * 100
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
                'pnl_pct': pnl_pct * 100
            })
            position = 0
    
    return trades, equity

# 测试不同资金档位
for capital in [1000, 5000, 10000]:
    trades, final_equity = backtest(df, capital=capital)
    
    if not trades:
        print(f"\n资金 {capital}元: 无交易信号")
        continue
    
    trades_df = pd.DataFrame(trades)
    total = len(trades)
    wins = len(trades_df[trades_df['pnl'] > 0])
    win_rate = wins / total * 100
    total_return = (final_equity - capital) / capital * 100
    
    print(f"\n资金 {capital}元:")
    print(f"  交易次数: {total}")
    print(f"  盈利次数: {wins}")
    print(f"  胜率: {win_rate:.2f}%")
    print(f"  总收益率: {total_return:.2f}%")
    print(f"  最终资金: {final_equity:.2f}元")
    
    if total > 0:
        print(f"\n  最近3笔交易:")
        for _, t in trades_df.tail(3).iterrows():
            print(f"    {t['time']} {t['direction']} 入{t['entry']:.3f} 出{t['exit']:.3f} 盈亏{t['pnl']:.2f}元")

print("\n" + "="*70)
print("回测完成!")
print("="*70)
