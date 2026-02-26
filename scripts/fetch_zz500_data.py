#!/usr/bin/env python3
"""
抓取中证500ETF（510500）完整历史数据
使用新浪接口（之前测试可用）
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import time

def fetch_etf_daily_sina(symbol="510500"):
    """使用新浪接口获取ETF日线数据"""
    print(f"开始抓取中证500ETF（{symbol}）数据...")
    
    try:
        # 使用新浪接口获取日线数据
        df = ak.fund_etf_hist_em(symbol=symbol, period="daily", adjust="qfq")
        
        if df.empty:
            print("获取数据失败")
            return None
        
        print(f"获取到 {len(df)} 条日线数据")
        print(f"数据范围: {df['日期'].iloc[0]} 至 {df['日期'].iloc[-1]}")
        
        # 标准化列名
        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_change', 'change', 'turnover']
        
        # 转换数值类型
        numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'amount']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None

def calculate_indicators(df):
    """计算技术指标"""
    print("\n计算技术指标...")
    
    # 1. 布林带
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['std20'] = df['close'].rolling(window=20).std()
    df['upper'] = df['ma20'] + 2 * df['std20']
    df['lower'] = df['ma20'] - 2 * df['std20']
    df['z_score'] = (df['close'] - df['ma20']) / df['std20']
    
    # 2. MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = 2 * (df['dif'] - df['dea'])
    
    # 3. 成交量均线
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    
    # 4. ATR（真实波幅）
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr20'] = df['tr'].rolling(window=20).mean()
    
    # 5. 通道突破信号（20日高低点）
    df['channel_high'] = df['high'].rolling(window=20).max()
    df['channel_low'] = df['low'].rolling(window=20).min()
    df['channel_mid'] = (df['channel_high'] + df['channel_low']) / 2
    
    # 判断是否突破（用前一天的通道值判断）
    df['breakout_up'] = df['close'] > df['channel_high'].shift(1)
    df['breakout_down'] = df['close'] < df['channel_low'].shift(1)
    
    # 删除临时列
    df = df.drop(['tr1', 'tr2', 'tr3'], axis=1)
    
    print("技术指标计算完成")
    return df

def save_data(df):
    """保存数据到本地"""
    if df is None:
        return
    
    # 创建data目录
    os.makedirs('data', exist_ok=True)
    
    # 保存为CSV（方便Excel查看）
    csv_path = 'data/zz500_510500_daily.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ CSV格式: {csv_path}")
    
    # 保存为Pickle（Python快速读取）
    pkl_path = 'data/zz500_510500_daily.pkl'
    df.to_pickle(pkl_path)
    print(f"✓ Pickle格式: {pkl_path}")
    
    # 保存为JSON（方便其他语言读取）
    json_path = 'data/zz500_510500_daily.json'
    df_json = df.copy()
    df_json['date'] = df_json['date'].astype(str)
    # 处理NaN值
    df_json = df_json.fillna(0)
    df_json.to_json(json_path, orient='records', force_ascii=False, indent=2)
    print(f"✓ JSON格式: {json_path}")
    
    # 保存元数据
    meta = {
        'symbol': '510500',
        'name': '中证500ETF',
        'total_records': len(df),
        'start_date': str(df['date'].iloc[0]),
        'end_date': str(df['date'].iloc[-1]),
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'columns': list(df.columns),
        'indicators': ['ma20', 'std20', 'upper', 'lower', 'z_score', 'dif', 'dea', 'macd', 'atr20', 'channel_high', 'channel_low', 'breakout_up', 'breakout_down']
    }
    
    meta_path = 'data/zz500_510500_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✓ 元数据: {meta_path}")
    
    return csv_path, pkl_path, json_path

def analyze_data(df):
    """数据分析摘要"""
    print("\n" + "="*60)
    print("数据统计摘要")
    print("="*60)
    
    print(f"\n总交易日: {len(df)}")
    print(f"起始日期: {df['date'].iloc[0]}")
    print(f"结束日期: {df['date'].iloc[-1]}")
    
    print(f"\n价格统计:")
    print(f"  最高价: {df['high'].max():.3f}")
    print(f"  最低价: {df['low'].min():.3f}")
    print(f"  最新价: {df['close'].iloc[-1]:.3f}")
    print(f"  平均价: {df['close'].mean():.3f}")
    
    print(f"\n波动率统计:")
    print(f"  20日波动率均值: {df['std20'].mean():.3f}")
    print(f"  ATR均值: {df['atr20'].mean():.3f}")
    
    # 突破信号统计
    up_signals = df['breakout_up'].sum()
    down_signals = df['breakout_down'].sum()
    
    print(f"\n通道突破信号:")
    print(f"  向上突破: {up_signals}次 ({up_signals/len(df)*100:.2f}%)")
    print(f"  向下突破: {down_signals}次 ({down_signals/len(df)*100:.2f}%)")
    
    # 布林带突破统计
    bb_up = (df['close'] > df['upper']).sum()
    bb_down = (df['close'] < df['lower']).sum()
    
    print(f"\n布林带突破:")
    print(f"  突破上轨: {bb_up}次 ({bb_up/len(df)*100:.2f}%)")
    print(f"  突破下轨: {bb_down}次 ({bb_down/len(df)*100:.2f}%)")
    
    # 最新信号
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    print(f"\n最新数据 ({latest['date']}):")
    print(f"  收盘价: {latest['close']:.3f}")
    print(f"  20日均线: {latest['ma20']:.3f}")
    print(f"  布林上轨: {latest['upper']:.3f}")
    print(f"  布林下轨: {latest['lower']:.3f}")
    print(f"  Z分数: {latest['z_score']:.3f}")
    print(f"  通道上轨: {latest['channel_high']:.3f}")
    print(f"  通道下轨: {latest['channel_low']:.3f}")
    
    # 判断信号
    if latest['breakout_up']:
        if prev['breakout_up']:
            print(f"  信号: ✅ 确认向上突破")
        else:
            print(f"  信号: ⚠️ 首次向上突破")
    elif latest['breakout_down']:
        if prev['breakout_down']:
            print(f"  信号: ✅ 确认向下突破")
        else:
            print(f"  信号: ⚠️ 首次向下突破")
    else:
        dist_upper = (latest['channel_high'] - latest['close']) / latest['close'] * 100
        dist_lower = (latest['close'] - latest['channel_low']) / latest['close'] * 100
        print(f"  信号: 无突破 (距上轨{dist_upper:.2f}%, 距下轨{dist_lower:.2f}%)")

def main():
    print("="*60)
    print("中证500ETF数据抓取")
    print("="*60)
    
    # 抓取数据
    df = fetch_etf_daily_sina("510500")
    
    if df is not None:
        # 计算指标
        df = calculate_indicators(df)
        
        # 保存数据
        save_data(df)
        
        # 分析数据
        analyze_data(df)
        
        print("\n" + "="*60)
        print("数据抓取完成！")
        print("="*60)
        print("\n可用于回测的文件:")
        print("  - data/zz500_510500_daily.csv (Excel可读)")
        print("  - data/zz500_510500_daily.pkl (Python快速读取)")
        print("  - data/zz500_510500_daily.json (通用格式)")
    else:
        print("\n数据抓取失败，请检查网络连接")

if __name__ == "__main__":
    main()
