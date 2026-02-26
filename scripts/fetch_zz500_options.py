#!/usr/bin/env python3
"""
抓取中证500ETF期权（510500）历史数据
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import time

# 中证500ETF期权合约代码格式
# 510500C2503M06000 = 510500购3月6000
# 510500P2503M06000 = 510500沽3月6000

def get_option_contracts():
    """获取中证500ETF期权合约列表"""
    try:
        # 获取ETF当前价格
        etf_df = ak.fund_etf_hist_em(symbol="510500", period="daily", adjust="qfq")
        if etf_df.empty:
            return None, None
        
        current_price = float(etf_df['收盘'].iloc[-1])
        print(f"中证500ETF当前价格: {current_price:.3f}")
        
        # 生成行权价列表（围绕当前价格）
        # 中证500ETF期权行权价间距通常是0.05或0.1
        base = round(current_price / 0.05) * 0.05
        strikes = []
        for i in range(-10, 11):
            strikes.append(round(base + i * 0.05, 2))
        
        return strikes, current_price
    except Exception as e:
        print(f"获取ETF价格失败: {e}")
        return None, None

def fetch_option_data(strike, option_type, expiry="2503"):
    """抓取单个期权合约数据"""
    try:
        # 构造合约代码
        # 格式: 510500C2503M06000 (购) 或 510500P2503M06000 (沽)
        strike_str = str(int(strike * 100)).zfill(5)  # 6.00 -> 00600
        code = f"510500{option_type}{expiry}M{strike_str}"
        
        # 使用新浪接口获取历史数据
        df = ak.option_commodity_hist_sina(symbol=code)
        
        if df.empty:
            return None
        
        df['code'] = code
        df['strike'] = strike
        df['type'] = '认购' if option_type == 'C' else '认沽'
        df['underlying'] = '510500'
        
        return df
    except Exception as e:
        return None

def fetch_all_options():
    """抓取所有期权合约"""
    strikes, etf_price = get_option_contracts()
    
    if strikes is None:
        return None
    
    print(f"\n准备抓取 {len(strikes)} 个行权价的期权数据...")
    print(f"行权价范围: {min(strikes):.2f} - {max(strikes):.2f}")
    
    all_options = []
    
    for strike in strikes:
        # 抓取认购期权
        call_df = fetch_option_data(strike, 'C')
        if call_df is not None and not call_df.empty:
            all_options.append(call_df)
            print(f"✓ 购 {strike:.2f}: {len(call_df)}条数据")
        
        time.sleep(0.2)  # 避免请求过快
        
        # 抓取认沽期权
        put_df = fetch_option_data(strike, 'P')
        if put_df is not None and not put_df.empty:
            all_options.append(put_df)
            print(f"✓ 沽 {strike:.2f}: {len(put_df)}条数据")
        
        time.sleep(0.2)
    
    if all_options:
        return pd.concat(all_options, ignore_index=True), etf_price
    return None, etf_price

def analyze_options(df, etf_price):
    """分析期权数据"""
    print("\n" + "="*60)
    print("期权数据分析")
    print("="*60)
    
    print(f"\n总合约数: {df['code'].nunique()}")
    print(f"总记录数: {len(df)}")
    
    # 按类型统计
    call_df = df[df['type'] == '认购']
    put_df = df[df['type'] == '认沽']
    
    print(f"\n认购期权: {call_df['code'].nunique()}个合约, {len(call_df)}条记录")
    print(f"认沽期权: {put_df['code'].nunique()}个合约, {len(put_df)}条记录")
    
    # 最新数据
    latest_date = df['date'].max()
    latest_df = df[df['date'] == latest_date]
    
    print(f"\n最新数据 ({latest_date}):")
    print(f"  活跃合约数: {len(latest_df)}")
    
    # 找出平值期权
    atm_call = call_df.iloc[(call_df['strike'] - etf_price).abs().argsort()[:1]]
    atm_put = put_df.iloc[(put_df['strike'] - etf_price).abs().argsort()[:1]]
    
    print(f"\n平值期权:")
    if not atm_call.empty:
        print(f"  认购: 行权价{atm_call['strike'].values[0]:.2f}, 最新价{atm_call['close'].values[0]:.4f}")
    if not atm_put.empty:
        print(f"  认沽: 行权价{atm_put['strike'].values[0]:.2f}, 最新价{atm_put['close'].values[0]:.4f}")
    
    return latest_df

def save_options_data(df, etf_price):
    """保存期权数据"""
    if df is None or df.empty:
        return
    
    os.makedirs('data/options', exist_ok=True)
    
    # 保存原始数据
    csv_path = 'data/options/zz500_options_510500.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 期权数据: {csv_path}")
    
    # 保存为pickle
    pkl_path = 'data/options/zz500_options_510500.pkl'
    df.to_pickle(pkl_path)
    print(f"✓ Pickle格式: {pkl_path}")
    
    # 保存元数据
    meta = {
        'underlying': '510500',
        'underlying_name': '中证500ETF',
        'underlying_price': etf_price,
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_contracts': df['code'].nunique(),
        'total_records': len(df),
        'date_range': f"{df['date'].min()} 至 {df['date'].max()}",
    }
    
    meta_path = 'data/options/zz500_options_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✓ 元数据: {meta_path}")

def main():
    print("="*60)
    print("中证500ETF期权数据抓取")
    print("="*60)
    
    # 抓取所有期权数据
    df, etf_price = fetch_all_options()
    
    if df is not None and not df.empty:
        # 分析数据
        analyze_options(df, etf_price)
        
        # 保存数据
        save_options_data(df, etf_price)
        
        print("\n" + "="*60)
        print("期权数据抓取完成！")
        print("="*60)
    else:
        print("\n未获取到期权数据")

if __name__ == "__main__":
    main()
