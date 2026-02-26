#!/usr/bin/env python3
"""
GitHub Actions用 - 抓取ETF期权数据（可用接口版本）
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import time

def fetch_option_risk_data():
    """获取期权风险指标数据（可用）"""
    try:
        df = ak.option_risk_indicator_sse()
        if df is not None and not df.empty:
            # 处理数据
            df['TRADE_DATE'] = pd.to_datetime(df['TRADE_DATE'])
            
            # 计算每个合约的统计
            summary = {
                'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_contracts': len(df),
                'unique_dates': df['TRADE_DATE'].nunique(),
                'avg_iv': float(df['IMPLC_VOLATLTY'].mean()),
                'contracts': df['CONTRACT_SYMBOL'].unique().tolist()[:20]  # 前20个合约
            }
            
            # 保存详细数据
            data_path = 'data/options/option_risk_data.csv'
            os.makedirs('data/options', exist_ok=True)
            df.to_csv(data_path, index=False, encoding='utf-8-sig')
            
            # 保存摘要
            summary_path = 'data/options/option_risk_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            print(f"✓ 期权风险数据: {len(df)}条合约")
            return summary
    except Exception as e:
        print(f"✗ 期权风险数据失败: {e}")
    return None

def fetch_option_lhb():
    """获取期权龙虎榜数据（可用）"""
    try:
        df = ak.option_lhb_em()
        if df is not None and not df.empty:
            data_path = 'data/options/option_lhb.csv'
            df.to_csv(data_path, index=False, encoding='utf-8-sig')
            print(f"✓ 期权龙虎榜: {len(df)}条")
            return len(df)
    except Exception as e:
        print(f"✗ 期权龙虎榜失败: {e}")
    return 0

def fetch_option_daily_stats():
    """获取期权日度统计（可用）"""
    try:
        df = ak.option_daily_stats_sse()
        if df is not None and not df.empty:
            data_path = 'data/options/option_daily_stats.csv'
            df.to_csv(data_path, index=False, encoding='utf-8-sig')
            print(f"✓ 期权日度统计: {len(df)}条")
            return len(df)
    except Exception as e:
        print(f"✗ 期权日度统计失败: {e}")
    return 0

def generate_option_summary():
    """生成期权数据摘要"""
    summary = {
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_sources': [
            'option_risk_data.csv - 期权风险指标（Greeks、IV）',
            'option_lhb.csv - 期权龙虎榜',
            'option_daily_stats.csv - 期权日度统计'
        ],
        'note': '由于网络限制，部分实时数据接口无法访问，已使用可用接口'
    }
    
    summary_path = 'data/options/option_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary

def main():
    print("="*80)
    print(f"ETF期权数据抓取 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)
    print()
    
    # 抓取可用数据
    risk_data = fetch_option_risk_data()
    lhb_count = fetch_option_lhb()
    stats_count = fetch_option_daily_stats()
    
    # 生成摘要
    summary = generate_option_summary()
    
    print()
    print("="*80)
    print("抓取完成")
    print("="*80)
    print(f"数据文件保存在: data/options/")
    print(f"  - option_risk_data.csv - 期权Greeks和隐含波动率")
    print(f"  - option_lhb.csv - 期权龙虎榜")
    print(f"  - option_daily_stats.csv - 日度统计")
    print(f"  - option_summary.json - 数据摘要")

if __name__ == "__main__":
    main()
