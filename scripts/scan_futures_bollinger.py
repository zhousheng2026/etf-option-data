#!/usr/bin/env python3
"""
扫描带期权的商品期货品种，检测日K线布林带突破上轨信号
使用新浪数据接口
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# 带期权的商品期货品种（主力合约代码）
OPTION_UNDERLYINGS = {
    # 上期所
    'CU': '沪铜',
    'AL': '沪铝', 
    'ZN': '沪锌',
    'PB': '沪铅',
    'NI': '沪镍',
    'SN': '沪锡',
    'AU': '沪金',
    'AG': '沪银',
    'RB': '螺纹钢',
    'HC': '热卷',
    'RU': '橡胶',
    'SP': '纸浆',
    'FU': '燃油',
    'BU': '沥青',
    'SC': '原油',
}

def calculate_bollinger_bands(df, period=20, std_dev=2.0):
    """计算布林带"""
    df['MA'] = df['close'].rolling(window=period).mean()
    df['STD'] = df['close'].rolling(window=period).std()
    df['UPPER'] = df['MA'] + std_dev * df['STD']
    df['LOWER'] = df['MA'] - std_dev * df['STD']
    return df

def get_futures_daily_sina(symbol):
    """使用新浪接口获取期货日线数据"""
    try:
        # 使用futures_zh_daily_sina获取日线数据
        df = ak.futures_zh_daily_sina(symbol=symbol)
        
        if df is None or df.empty:
            return None
        
        # 标准化列名
        df.columns = [col.lower() for col in df.columns]
        
        # 转换数值
        for col in ['close', 'high', 'low', 'open']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['close'])
        return df.sort_values('date') if 'date' in df.columns else df
    except Exception as e:
        return None

def check_bollinger_breakout(df, period=20, std_dev=2.0):
    """检查是否突破布林带上轨"""
    if df is None or len(df) < period + 5:
        return None
    
    df = calculate_bollinger_bands(df, period, std_dev)
    
    # 获取最近3天数据
    recent = df.tail(3)
    
    if len(recent) < 3:
        return None
    
    # 检查信号
    latest = recent.iloc[-1]
    prev = recent.iloc[-2]
    
    signal = {
        'breakout': False,
        'close': latest['close'],
        'upper': latest['UPPER'],
        'ma': latest['MA'],
        'lower': latest['LOWER'],
        'breakout_strength': 0,
        'confirmed': False,
    }
    
    # 突破上轨条件：收盘价 > 上轨
    if latest['close'] > latest['UPPER'] and not pd.isna(latest['UPPER']):
        signal['breakout'] = True
        signal['breakout_strength'] = (latest['close'] - latest['UPPER']) / latest['UPPER'] * 100
    
    # 检查前一天是否也突破（确认趋势）
    if not pd.isna(prev['UPPER']) and prev['close'] > prev['UPPER']:
        signal['confirmed'] = True
    
    return signal

def scan_all_futures():
    """扫描所有带期权的期货品种"""
    results = []
    errors = []
    
    print("=" * 80)
    print(f"商品期货布林带突破扫描 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    print(f"共扫描 {len(OPTION_UNDERLYINGS)} 个品种\n")
    
    for code, name in OPTION_UNDERLYINGS.items():
        print(f"扫描 {code} ({name})...", end=" ")
        
        try:
            # 尝试获取连续合约数据（格式：代码0）
            symbol = f"{code}0"  # 连续合约
            df = get_futures_daily_sina(symbol)
            
            if df is None or df.empty:
                errors.append(f"{code}({name}): 无数据")
                print("❌ 无数据")
                continue
            
            signal = check_bollinger_breakout(df)
            if signal is None:
                errors.append(f"{code}({name}): 数据不足")
                print("❌ 数据不足")
                continue
            
            if signal['breakout']:
                status = "✅ 确认" if signal['confirmed'] else "⚠️ 首次"
                print(f"{status}突破! 强度: {signal['breakout_strength']:.2f}%")
                results.append({
                    'code': code,
                    'name': name,
                    'close': signal['close'],
                    'upper': signal['upper'],
                    'ma': signal['ma'],
                    'lower': signal['lower'],
                    'strength': signal['breakout_strength'],
                    'confirmed': signal['confirmed']
                })
            else:
                distance = (signal['upper'] - signal['close']) / signal['close'] * 100 if not pd.isna(signal['upper']) else 0
                print(f"无信号 (距上轨: {distance:.2f}%)")
                
        except Exception as e:
            errors.append(f"{code}({name}): {str(e)}")
            print(f"❌ 错误")
        
        time.sleep(0.3)  # 避免请求过快
    
    # 输出结果
    print("\n" + "=" * 80)
    print("扫描结果 - 突破布林带上轨的品种")
    print("=" * 80)
    
    if not results:
        print("\n暂无品种突破布林带上轨")
    else:
        # 按突破强度排序
        results.sort(key=lambda x: x['strength'], reverse=True)
        
        print(f"\n共 {len(results)} 个品种出现突破信号:\n")
        print(f"{'代码':<6} {'名称':<10} {'收盘价':<10} {'布林上轨':<10} {'突破强度':<10} {'确认':<6}")
        print("-" * 60)
        
        for r in results:
            confirm = "✅" if r['confirmed'] else "⚠️"
            print(f"{r['code']:<6} {r['name']:<10} {r['close']:<10.2f} {r['upper']:<10.2f} {r['strength']:<10.2f}% {confirm:<6}")
    
    return results

if __name__ == "__main__":
    scan_all_futures()
