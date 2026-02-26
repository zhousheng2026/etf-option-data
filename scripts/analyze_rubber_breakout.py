#!/usr/bin/env python3
"""
分析橡胶突破布林带上轨的统计意义
计算西格玛值和出现频率
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import math

def get_rubber_data():
    """获取橡胶历史数据"""
    try:
        df = ak.futures_zh_daily_sina(symbol="RU0")
        df.columns = [col.lower() for col in df.columns]
        for col in ['close', 'high', 'low', 'open']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['close'])
        return df.sort_values('date').reset_index(drop=True)
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None

def calculate_bollinger_stats(df, period=20, std_dev=2.0):
    """计算布林带统计信息"""
    df['MA'] = df['close'].rolling(window=period).mean()
    df['STD'] = df['close'].rolling(window=period).std()
    df['UPPER'] = df['MA'] + std_dev * df['STD']
    df['LOWER'] = df['MA'] - std_dev * df['STD']
    
    # 计算收盘价与中轨的距离（以标准差为单位）
    df['Z_SCORE'] = (df['close'] - df['MA']) / df['STD']
    
    # 判断是否突破上轨
    df['ABOVE_UPPER'] = df['close'] > df['UPPER']
    
    return df

def analyze_breakout_frequency(df, period=20):
    """分析突破频率"""
    # 只取有完整布林带数据的部分
    valid_data = df[period:].copy()
    
    total_days = len(valid_data)
    breakout_days = valid_data['ABOVE_UPPER'].sum()
    breakout_rate = breakout_days / total_days * 100
    
    print(f"\n{'='*60}")
    print(f"橡胶突破布林带上轨统计分析")
    print(f"{'='*60}")
    print(f"\n数据范围: {valid_data['date'].iloc[0]} 至 {valid_data['date'].iloc[-1]}")
    print(f"总交易日: {total_days} 天")
    print(f"突破上轨天数: {breakout_days} 天")
    print(f"突破频率: {breakout_rate:.2f}%")
    print(f"平均出现周期: 约 {total_days/breakout_days:.1f} 天一次" if breakout_days > 0 else "无突破记录")
    
    return valid_data

def normal_cdf(x):
    """标准正态分布CDF近似"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def analyze_current_breakout(df):
    """分析当前突破的西格玛值"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 当前Z分数（距离中轨几个标准差）
    current_z = latest['Z_SCORE']
    
    # 突破上轨需要的Z分数（通常是2.0）
    upper_z = 2.0
    
    # 实际突破强度（超过2.0的部分）
    excess_z = current_z - upper_z
    
    print(f"\n{'='*60}")
    print(f"当前突破分析 ({latest['date']})")
    print(f"{'='*60}")
    print(f"\n收盘价: {latest['close']:.2f}")
    print(f"布林中轨: {latest['MA']:.2f}")
    print(f"布林上轨: {latest['UPPER']:.2f}")
    print(f"20日标准差: {latest['STD']:.2f}")
    print(f"\n当前Z分数: {current_z:.3f} σ")
    print(f"突破上轨所需: {upper_z:.1f} σ")
    print(f"超额突破: {excess_z:.3f} σ")
    
    # 计算概率（使用正态分布近似）
    # 正态分布下，Z>2的概率
    prob_above_2sigma = 1 - normal_cdf(2.0)
    prob_above_current = 1 - normal_cdf(current_z)
    
    print(f"\n{'='*60}")
    print(f"统计概率分析（假设正态分布）")
    print(f"{'='*60}")
    print(f"Z > 2.0 (突破上轨) 的理论概率: {prob_above_2sigma*100:.3f}%")
    print(f"Z > {current_z:.3f} (当前水平) 的理论概率: {prob_above_current*100:.4f}%")
    print(f"当前突破的罕见程度: 约 {1/prob_above_current:.0f} 天出现一次")
    
    # 检查是否是连续突破
    is_confirmed = prev['ABOVE_UPPER']
    print(f"\n{'='*60}")
    print(f"突破确认状态")
    print(f"{'='*60}")
    print(f"昨日是否突破: {'是' if is_confirmed else '否'}")
    print(f"突破类型: {'✅ 确认突破（连续2天）' if is_confirmed else '⚠️ 首次突破'}")
    
    return current_z, excess_z

def analyze_historical_extremes(df, period=20):
    """分析历史极端值"""
    valid_data = df[period:].copy()
    
    max_z = valid_data['Z_SCORE'].max()
    min_z = valid_data['Z_SCORE'].min()
    max_z_date = valid_data.loc[valid_data['Z_SCORE'].idxmax(), 'date']
    min_z_date = valid_data.loc[valid_data['Z_SCORE'].idxmin(), 'date']
    
    print(f"\n{'='*60}")
    print(f"历史极端值统计")
    print(f"{'='*60}")
    print(f"历史最大Z分数: {max_z:.3f} σ ({max_z_date})")
    print(f"历史最小Z分数: {min_z:.3f} σ ({min_z_date})")
    print(f"当前Z分数排名: 历史前 {(valid_data['Z_SCORE'] >= valid_data['Z_SCORE'].iloc[-1]).sum()} / {len(valid_data)}")
    
    # 统计不同Z分数区间的出现频率
    print(f"\n{'='*60}")
    print(f"Z分数分布统计")
    print(f"{'='*60}")
    
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    for t in thresholds:
        count = (valid_data['Z_SCORE'] > t).sum()
        freq = len(valid_data) / count if count > 0 else float('inf')
        print(f"Z > {t}: {count}次, 约 {freq:.0f} 天一次")

def main():
    print("正在获取橡胶历史数据...")
    df = get_rubber_data()
    
    if df is None or len(df) < 100:
        print("数据获取失败或数据不足")
        return
    
    print(f"获取到 {len(df)} 条历史数据")
    
    # 计算布林带
    df = calculate_bollinger_stats(df)
    
    # 分析突破频率
    valid_data = analyze_breakout_frequency(df)
    
    # 分析当前突破
    current_z, excess_z = analyze_current_breakout(df)
    
    # 分析历史极端值
    analyze_historical_extremes(df)
    
    print(f"\n{'='*60}")
    print(f"结论")
    print(f"{'='*60}")
    print(f"当前橡胶突破布林带上轨，Z分数为 {current_z:.3f} σ")
    print(f"在正态分布假设下，这种水平约 {1/(1-normal_cdf(current_z)):.0f} 天出现一次")
    print(f"但实际市场分布有肥尾效应，极端值出现频率高于理论值")

if __name__ == "__main__":
    main()
