#!/usr/bin/env python3
"""
GitHub Actions用 - 抓取ETF期权数据（含分钟线历史）
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time

# ETF期权标的配置
ETF_OPTIONS = {
    '510050': {'name': '50ETF', 'exchange': '上交所'},
    '510300': {'name': '300ETF', 'exchange': '上交所'},
    '510500': {'name': '500ETF', 'exchange': '上交所'},
    '159915': {'name': '创业板ETF', 'exchange': '深交所'},
    '588000': {'name': '科创50ETF', 'exchange': '上交所'},
}

def fetch_option_minute_history(symbol='10005245C02700', period='30'):
    """
    抓取期权分钟线历史数据
    
    参数:
    - symbol: 期权合约代码（如 10005245C02700 = 510500购2月2700）
    - period: 分钟周期（1, 5, 15, 30, 60）
    """
    try:
        # 使用akshare获取期权分钟数据
        df = ak.option_commodity_hist_sina(symbol=symbol)
        
        if df is None or df.empty:
            return None
        
        # 标准化列名
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        # 转换数值
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 计算技术指标
        if len(df) >= 20:
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['std20'] = df['close'].rolling(window=20).std()
            df['upper'] = df['ma20'] + 2 * df['std20']
            df['lower'] = df['ma20'] - 2 * df['std20']
            df['sigma'] = (df['close'] - df['ma20']) / df['std20']
        
        return df
    except Exception as e:
        print(f"  获取{symbol}分钟数据失败: {e}")
        return None

def fetch_option_daily_history(symbol='10005245C02700'):
    """抓取期权日线历史数据"""
    try:
        df = ak.option_commodity_hist_sina(symbol=symbol)
        
        if df is None or df.empty:
            return None
        
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 计算更多指标
        if len(df) >= 20:
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['std20'] = df['close'].rolling(window=20).std()
            df['upper'] = df['ma20'] + 2 * df['std20']
            df['lower'] = df['ma20'] - 2 * df['std20']
            df['sigma'] = (df['close'] - df['ma20']) / df['std20']
            
            # MACD
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['dif'] = df['ema12'] - df['ema26']
            df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
            df['macd'] = 2 * (df['dif'] - df['dea'])
        
        return df
    except Exception as e:
        print(f"  获取{symbol}日线数据失败: {e}")
        return None

def get_active_option_contracts(underlying='510500'):
    """获取活跃的期权合约代码"""
    try:
        # 获取ETF当前价格
        etf_df = ak.fund_etf_hist_em(symbol=underlying, period="daily", adjust="qfq")
        if etf_df.empty:
            return []
        
        current_price = float(etf_df['收盘'].iloc[-1])
        
        # 生成平值和虚值合约代码
        # 格式：10005 + 到期月份 + C/P + 行权价
        # 例如：100052503C06000 = 510500购3月6000
        
        contracts = []
        expiry_months = ['2503', '2504', '2506']  # 3月、4月、6月到期
        
        for expiry in expiry_months:
            # 平值附近行权价
            base_strike = round(current_price / 0.05) * 0.05
            strikes = [base_strike + i * 0.05 for i in range(-5, 6)]
            
            for strike in strikes:
                strike_str = str(int(strike * 100)).zfill(5)
                
                # 认购期权
                call_code = f"10005{expiry}C{strike_str}" if underlying == '510500' else f"10000{expiry}C{strike_str}"
                contracts.append({
                    'code': call_code,
                    'underlying': underlying,
                    'type': '认购',
                    'strike': strike,
                    'expiry': expiry
                })
                
                # 认沽期权
                put_code = f"10005{expiry}P{strike_str}" if underlying == '510500' else f"10000{expiry}P{strike_str}"
                contracts.append({
                    'code': put_code,
                    'underlying': underlying,
                    'type': '认沽',
                    'strike': strike,
                    'expiry': expiry
                })
        
        return contracts
    except Exception as e:
        print(f"获取合约列表失败: {e}")
        return []

def fetch_all_option_data():
    """抓取所有期权数据"""
    print("="*80)
    print(f"ETF期权数据抓取 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)
    
    all_data = {
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'daily': {},
        'minute': {},
        'contracts': []
    }
    
    for symbol, info in ETF_OPTIONS.items():
        print(f"\n抓取 {info['name']} ({symbol}) 期权...")
        
        # 获取活跃合约
        contracts = get_active_option_contracts(symbol)
        print(f"  生成 {len(contracts)} 个合约代码")
        
        # 抓取更多合约（用于回测）
        for contract in contracts[:15]:  # 增加到15个合约
            code = contract['code']
            print(f"  抓取 {code}...", end=" ")
            
            # 抓取日线历史
            daily_df = fetch_option_daily_history(code)
            if daily_df is not None and not daily_df.empty:
                all_data['daily'][code] = {
                    'underlying': symbol,
                    'type': contract['type'],
                    'strike': contract['strike'],
                    'expiry': contract['expiry'],
                    'records': len(daily_df),
                    'latest_price': float(daily_df['close'].iloc[-1]),
                    'data': daily_df.to_dict('records')  # 全部历史数据
                }
                print(f"日线{len(daily_df)}条", end=" ")
            
            # 抓取分钟历史（30分钟）
            minute_df = fetch_option_minute_history(code, period='30')
            if minute_df is not None and not minute_df.empty:
                all_data['minute'][code] = {
                    'underlying': symbol,
                    'type': contract['type'],
                    'strike': contract['strike'],
                    'expiry': contract['expiry'],
                    'records': len(minute_df),
                    'latest_price': float(minute_df['close'].iloc[-1]),
                    'data': minute_df.to_dict('records')  # 全部分钟数据
                }
                print(f"分钟{len(minute_df)}条", end=" ")
            
            print("✓")
            time.sleep(0.5)  # 避免请求过快
        
        all_data['contracts'].extend(contracts[:5])
    
    # 保存数据
    os.makedirs('data/options', exist_ok=True)
    
    # 保存JSON
    json_path = 'data/options/option_history.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 数据已保存: {json_path}")
    
    # 保存摘要
    summary = {
        'update_time': all_data['update_time'],
        'daily_contracts': len(all_data['daily']),
        'minute_contracts': len(all_data['minute']),
        'total_contracts': len(all_data['contracts'])
    }
    
    summary_path = 'data/options/option_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✓ 摘要已保存: {summary_path}")
    
    return all_data

def main():
    fetch_all_option_data()

if __name__ == "__main__":
    main()
