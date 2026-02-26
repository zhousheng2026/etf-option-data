#!/usr/bin/env python3
"""
GitHub Actions用 - 抓取ETF期权数据
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
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

def get_etf_price(symbol):
    """获取ETF当前价格"""
    try:
        df = ak.fund_etf_hist_em(symbol=symbol, period="daily", adjust="qfq")
        if not df.empty:
            return float(df['收盘'].iloc[-1])
        return None
    except:
        return None

def fetch_option_quotes_em(symbol):
    """使用东财接口获取期权T型报价"""
    try:
        # 东财接口获取期权实时行情
        df = ak.option_current_em()
        
        # 筛选对应ETF的期权
        # 期权代码格式：10005xxx（510500期权）
        symbol_prefix = {
            '510050': '10000',
            '510300': '10001',
            '510500': '10005',
            '159915': '1599',
            '588000': '5880',
        }
        
        prefix = symbol_prefix.get(symbol)
        if prefix and df is not None and not df.empty:
            # 筛选对应标的的期权
            filtered = df[df['合约代码'].astype(str).str.startswith(prefix)]
            return filtered
        return None
    except Exception as e:
        print(f"  东财接口失败: {e}")
        return None

def fetch_option_chain_sina(symbol, etf_price):
    """使用新浪接口获取期权链"""
    try:
        # 生成可能的期权合约代码
        # 格式：10005503C06000 = 510500购3月6000
        
        contracts = []
        expiry = "2503"  # 3月到期
        
        # 生成行权价（围绕ETF价格）
        if symbol == '510500':
            step = 0.05
            base = round(etf_price / step) * step
            strikes = [round(base + i * step, 2) for i in range(-10, 11)]
        elif symbol == '510050':
            step = 0.05
            base = round(etf_price / step) * step
            strikes = [round(base + i * step, 2) for i in range(-10, 11)]
        else:
            step = 0.05
            base = round(etf_price / step) * step
            strikes = [round(base + i * step, 2) for i in range(-8, 9)]
        
        for strike in strikes:
            strike_str = str(int(strike * 100)).zfill(5)
            
            # 认购期权
            try:
                call_code = f"10005{expiry}C{strike_str}" if symbol == '510500' else f"10000{expiry}C{strike_str}"
                df_call = ak.option_commodity_hist_sina(symbol=call_code)
                if not df_call.empty:
                    latest = df_call.iloc[-1]
                    contracts.append({
                        'code': call_code,
                        'underlying': symbol,
                        'type': '认购',
                        'strike': strike,
                        'close': float(latest['close']),
                        'volume': int(latest['volume']),
                        'date': str(latest['date'])
                    })
            except:
                pass
            
            # 认沽期权
            try:
                put_code = f"10005{expiry}P{strike_str}" if symbol == '510500' else f"10000{expiry}P{strike_str}"
                df_put = ak.option_commodity_hist_sina(symbol=put_code)
                if not df_put.empty:
                    latest = df_put.iloc[-1]
                    contracts.append({
                        'code': put_code,
                        'underlying': symbol,
                        'type': '认沽',
                        'strike': strike,
                        'close': float(latest['close']),
                        'volume': int(latest['volume']),
                        'date': str(latest['date'])
                    })
            except:
                pass
            
            time.sleep(0.1)
        
        return pd.DataFrame(contracts) if contracts else None
    except Exception as e:
        print(f"  新浪接口失败: {e}")
        return None

def fetch_all_options():
    """抓取所有ETF期权数据"""
    print("=" * 80)
    print(f"ETF期权数据抓取 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    
    all_options = []
    
    for symbol, info in ETF_OPTIONS.items():
        print(f"\n抓取 {info['name']} ({symbol}) 期权...")
        
        # 获取ETF价格
        etf_price = get_etf_price(symbol)
        if etf_price:
            print(f"  ETF价格: {etf_price:.3f}")
        
        # 尝试获取期权数据
        df = fetch_option_chain_sina(symbol, etf_price)
        
        if df is not None and not df.empty:
            df['underlying_name'] = info['name']
            df['etf_price'] = etf_price
            all_options.append(df)
            print(f"  ✓ 获取到 {len(df)} 个合约")
        else:
            print(f"  ✗ 未获取到数据")
    
    if all_options:
        return pd.concat(all_options, ignore_index=True)
    return None

def analyze_options(df):
    """分析期权数据"""
    print("\n" + "=" * 80)
    print("期权数据分析")
    print("=" * 80)
    
    print(f"\n总合约数: {len(df)}")
    print(f"标的数量: {df['underlying'].nunique()}")
    
    # 按标的统计
    for symbol in df['underlying'].unique():
        subset = df[df['underlying'] == symbol]
        name = subset['underlying_name'].iloc[0]
        etf_price = subset['etf_price'].iloc[0]
        
        calls = subset[subset['type'] == '认购']
        puts = subset[subset['type'] == '认沽']
        
        print(f"\n{name} ({symbol}):")
        print(f"  ETF价格: {etf_price:.3f}")
        print(f"  认购期权: {len(calls)}个")
        print(f"  认沽期权: {len(puts)}个")
        
        # 找出平值期权
        if not calls.empty:
            atm_call = calls.iloc[(calls['strike'] - etf_price).abs().argsort()[:1]]
            print(f"  平值认购: 行权价{atm_call['strike'].values[0]:.2f}, 价格{atm_call['close'].values[0]:.4f}")
        
        if not puts.empty:
            atm_put = puts.iloc[(puts['strike'] - etf_price).abs().argsort()[:1]]
            print(f"  平值认沽: 行权价{atm_put['strike'].values[0]:.2f}, 价格{atm_put['close'].values[0]:.4f}")

def save_options_data(df):
    """保存期权数据"""
    if df is None or df.empty:
        return
    
    os.makedirs('data/options', exist_ok=True)
    
    # 保存CSV
    csv_path = 'data/options/etf_options_latest.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 期权数据: {csv_path}")
    
    # 保存JSON
    json_path = 'data/options/etf_options_latest.json'
    df.to_json(json_path, orient='records', force_ascii=False, indent=2)
    print(f"✓ JSON格式: {json_path}")
    
    # 保存元数据
    meta = {
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_contracts': len(df),
        'underlyings': df['underlying'].unique().tolist(),
        'date_range': f"{df['date'].min()} 至 {df['date'].max()}" if 'date' in df.columns else 'N/A',
    }
    
    meta_path = 'data/options/etf_options_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✓ 元数据: {meta_path}")

def main():
    # 抓取期权数据
    df = fetch_all_options()
    
    if df is not None and not df.empty:
        # 分析数据
        analyze_options(df)
        
        # 保存数据
        save_options_data(df)
        
        print("\n" + "=" * 80)
        print("ETF期权数据抓取完成！")
        print("=" * 80)
    else:
        print("\n未获取到期权数据")

if __name__ == "__main__":
    main()
