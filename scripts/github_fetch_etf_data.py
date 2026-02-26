#!/usr/bin/env python3
"""
GitHub Actions用 - 抓取ETF和期权数据
"""

import akshare as ak
import pandas as pd
import json
from datetime import datetime
import os

# ETF配置
ETF_LIST = {
    '510050': '50ETF',
    '510300': '300ETF',
    '510500': '500ETF',
    '159915': '创业板ETF',
    '588000': '科创50ETF',
}

def fetch_etf_daily(symbol, name):
    """获取ETF日线数据"""
    try:
        df = ak.fund_etf_hist_em(symbol=symbol, period="daily", adjust="qfq")
        if df.empty:
            return None
        
        # 只保留最近60天
        df = df.tail(60)
        
        # 计算布林带
        df['ma20'] = df['收盘'].rolling(window=20).mean()
        df['std20'] = df['收盘'].rolling(window=20).std()
        df['upper'] = df['ma20'] + 2 * df['std20']
        df['lower'] = df['ma20'] - 2 * df['std20']
        df['z_score'] = (df['收盘'] - df['ma20']) / df['std20']
        
        # 检查突破信号
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signal = {
            'symbol': symbol,
            'name': name,
            'date': str(latest['日期']),
            'close': float(latest['收盘']),
            'ma20': float(latest['ma20']),
            'upper': float(latest['upper']),
            'lower': float(latest['lower']),
            'z_score': float(latest['z_score']),
            'breakout_up': bool(latest['收盘'] > latest['upper']),
            'breakout_down': bool(latest['收盘'] < latest['lower']),
            'confirmed_up': bool(prev['收盘'] > prev['upper']) if not pd.isna(prev['upper']) else False,
            'confirmed_down': bool(prev['收盘'] < prev['lower']) if not pd.isna(prev['lower']) else False,
        }
        
        return signal
    except Exception as e:
        print(f"获取{name}失败: {e}")
        return None

def fetch_rubber_data():
    """获取橡胶期货数据"""
    try:
        df = ak.futures_zh_daily_sina(symbol="RU0")
        df.columns = [col.lower() for col in df.columns]
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        
        # 计算布林带
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['upper'] = df['ma20'] + 2 * df['std20']
        df['lower'] = df['ma20'] - 2 * df['std20']
        df['z_score'] = (df['close'] - df['ma20']) / df['std20']
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        return {
            'symbol': 'RU',
            'name': '橡胶',
            'date': str(latest['date']),
            'close': float(latest['close']),
            'ma20': float(latest['ma20']),
            'upper': float(latest['upper']),
            'lower': float(latest['lower']),
            'z_score': float(latest['z_score']),
            'breakout_up': bool(latest['close'] > latest['upper']),
            'breakout_down': bool(latest['close'] < latest['lower']),
            'confirmed_up': bool(prev['close'] > prev['upper']) if not pd.isna(prev['upper']) else False,
            'confirmed_down': bool(prev['close'] < prev['lower']) if not pd.isna(prev['lower']) else False,
        }
    except Exception as e:
        print(f"获取橡胶失败: {e}")
        return None

def main():
    print(f"开始抓取数据 - {datetime.now()}")
    
    results = {
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'etf_signals': [],
        'futures_signals': [],
    }
    
    # 抓取ETF数据
    for symbol, name in ETF_LIST.items():
        print(f"抓取 {name}...")
        signal = fetch_etf_daily(symbol, name)
        if signal:
            results['etf_signals'].append(signal)
    
    # 抓取橡胶数据
    print("抓取橡胶...")
    rubber = fetch_rubber_data()
    if rubber:
        results['futures_signals'].append(rubber)
    
    # 保存结果
    os.makedirs('data', exist_ok=True)
    
    # 保存最新信号
    with open('data/latest_signals.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存历史记录
    history_file = f"data/history_{datetime.now().strftime('%Y%m%d')}.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据已保存到 data/latest_signals.json")
    print(f"历史记录: {history_file}")
    
    # 打印信号汇总
    print("\n=== 信号汇总 ===")
    for s in results['etf_signals']:
        if s['breakout_up']:
            status = "✅确认突破" if s['confirmed_up'] else "⚠️首次突破"
            print(f"{s['name']}: {status}上轨 (Z={s['z_score']:.2f})")
        elif s['breakout_down']:
            status = "✅确认突破" if s['confirmed_down'] else "⚠️首次突破"
            print(f"{s['name']}: {status}下轨 (Z={s['z_score']:.2f})")
    
    for s in results['futures_signals']:
        if s['breakout_up']:
            status = "✅确认突破" if s['confirmed_up'] else "⚠️首次突破"
            print(f"{s['name']}: {status}上轨 (Z={s['z_score']:.2f})")
        elif s['breakout_down']:
            status = "✅确认突破" if s['confirmed_down'] else "⚠️首次突破"
            print(f"{s['name']}: {status}下轨 (Z={s['z_score']:.2f})")

if __name__ == "__main__":
    main()
