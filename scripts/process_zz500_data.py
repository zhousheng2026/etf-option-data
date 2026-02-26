#!/usr/bin/env python3
"""
基于现有的中证500ETF 5分钟数据，生成策略回测所需数据
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

def load_5min_data():
    """加载现有的5分钟数据"""
    df = pd.read_csv('/root/.openclaw/workspace/zz500_etf_5min.csv')
    df.columns = ['datetime', 'open', 'close', 'high', 'low', 'pct_change', 'change', 'volume', 'amount', 'amplitude', 'turnover']
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def resample_to_daily(df):
    """将5分钟数据重采样为日线"""
    df = df.copy()
    df['date'] = df['datetime'].dt.date
    
    daily = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'amount': 'sum'
    }).reset_index()
    
    daily['date'] = pd.to_datetime(daily['date'])
    return daily

def resample_to_30min(df):
    """将5分钟数据重采样为30分钟"""
    df = df.copy()
    df.set_index('datetime', inplace=True)
    
    # 30分钟重采样
    resampled = df.resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'amount': 'sum'
    }).dropna()
    
    resampled.reset_index(inplace=True)
    return resampled

def calculate_indicators(df):
    """计算技术指标"""
    # 布林带
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['std20'] = df['close'].rolling(window=20).std()
    df['upper'] = df['ma20'] + 2 * df['std20']
    df['lower'] = df['ma20'] - 2 * df['std20']
    df['z_score'] = (df['close'] - df['ma20']) / df['std20']
    
    # MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = 2 * (df['dif'] - df['dea'])
    
    # 成交量均线
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    
    # ATR
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr20'] = df['tr'].rolling(window=20).mean()
    
    # 通道突破（20周期）
    df['channel_high'] = df['high'].rolling(window=20).max()
    df['channel_low'] = df['low'].rolling(window=20).min()
    df['channel_mid'] = (df['channel_high'] + df['channel_low']) / 2
    
    # 突破信号
    df['breakout_up'] = df['close'] > df['channel_high'].shift(1)
    df['breakout_down'] = df['close'] < df['channel_low'].shift(1)
    
    # K线实体
    df['body'] = abs(df['close'] - df['open'])
    df['body_ma20'] = df['body'].rolling(window=20).mean()
    
    # 删除临时列
    df = df.drop(['tr1', 'tr2', 'tr3'], axis=1)
    
    return df

def save_data(df, name, freq):
    """保存数据"""
    os.makedirs('data', exist_ok=True)
    
    # CSV
    csv_path = f'data/zz500_{freq}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ {csv_path}")
    
    # Pickle
    pkl_path = f'data/zz500_{freq}.pkl'
    df.to_pickle(pkl_path)
    print(f"✓ {pkl_path}")
    
    return csv_path, pkl_path

def analyze_signals(df, freq_name):
    """分析信号"""
    print(f"\n{'='*60}")
    print(f"{freq_name}数据统计")
    print("="*60)
    
    print(f"\n总记录数: {len(df)}")
    print(f"起始时间: {df['datetime'].iloc[0] if 'datetime' in df.columns else df['date'].iloc[0]}")
    print(f"结束时间: {df['datetime'].iloc[-1] if 'datetime' in df.columns else df['date'].iloc[-1]}")
    
    # 突破统计
    up_signals = df['breakout_up'].sum()
    down_signals = df['breakout_down'].sum()
    
    print(f"\n通道突破信号:")
    print(f"  向上突破: {up_signals}次")
    print(f"  向下突破: {down_signals}次")
    
    # 最新信号
    latest = df.iloc[-1]
    print(f"\n最新数据:")
    print(f"  收盘价: {latest['close']:.3f}")
    print(f"  Z分数: {latest['z_score']:.3f}")
    
    if latest['breakout_up']:
        print(f"  信号: ⚠️ 向上突破")
    elif latest['breakout_down']:
        print(f"  信号: ⚠️ 向下突破")
    else:
        print(f"  信号: 无突破")

def main():
    print("="*60)
    print("中证500ETF数据处理")
    print("="*60)
    
    # 加载5分钟数据
    print("\n加载5分钟数据...")
    df_5min = load_5min_data()
    print(f"加载了 {len(df_5min)} 条5分钟数据")
    
    # 生成30分钟数据
    print("\n生成30分钟数据...")
    df_30min = resample_to_30min(df_5min)
    df_30min = calculate_indicators(df_30min)
    save_data(df_30min, '510500', '30min')
    analyze_signals(df_30min, "30分钟")
    
    # 生成日线数据
    print("\n生成日线数据...")
    df_daily = resample_to_daily(df_5min)
    df_daily = calculate_indicators(df_daily)
    save_data(df_daily, '510500', 'daily')
    analyze_signals(df_daily, "日线")
    
    # 保存原始5分钟数据（带指标）
    print("\n处理5分钟数据...")
    df_5min = calculate_indicators(df_5min)
    save_data(df_5min, '510500', '5min')
    analyze_signals(df_5min, "5分钟")
    
    # 保存元数据
    meta = {
        'symbol': '510500',
        'name': '中证500ETF',
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'files': {
            '5min': {'records': len(df_5min), 'columns': list(df_5min.columns)},
            '30min': {'records': len(df_30min), 'columns': list(df_30min.columns)},
            'daily': {'records': len(df_daily), 'columns': list(df_daily.columns)},
        }
    }
    
    with open('data/zz500_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("数据处理完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  - data/zz500_5min.csv/pkl (5分钟数据)")
    print("  - data/zz500_30min.csv/pkl (30分钟数据 - 策略主用)")
    print("  - data/zz500_daily.csv/pkl (日线数据)")
    print("  - data/zz500_meta.json (元数据)")

if __name__ == "__main__":
    main()
