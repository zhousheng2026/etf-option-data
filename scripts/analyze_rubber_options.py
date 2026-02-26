#!/usr/bin/env python3
"""
获取橡胶期权数据并分析
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime

def get_rubber_option_contracts():
    """获取橡胶期权合约列表"""
    try:
        # 橡胶主力合约是ru2505，期权合约格式：ru2505C18000（认购）/ ru2505P18000（认沽）
        # 当前橡胶价格约17240，获取平值和虚值期权
        
        strikes = [17000, 17250, 17500, 17750, 18000]
        contracts = []
        
        for strike in strikes:
            # 认购期权
            try:
                call_code = f"ru2505C{strike}"
                df_call = ak.option_commodity_hist_sina(symbol=call_code)
                if not df_call.empty:
                    latest = df_call.iloc[-1]
                    contracts.append({
                        'code': call_code,
                        'type': '认购',
                        'strike': strike,
                        'close': latest['close'],
                        'volume': latest['volume'],
                        'date': latest['date']
                    })
            except:
                pass
            
            # 认沽期权
            try:
                put_code = f"ru2505P{strike}"
                df_put = ak.option_commodity_hist_sina(symbol=put_code)
                if not df_put.empty:
                    latest = df_put.iloc[-1]
                    contracts.append({
                        'code': put_code,
                        'type': '认沽',
                        'strike': strike,
                        'close': latest['close'],
                        'volume': latest['volume'],
                        'date': latest['date']
                    })
            except:
                pass
        
        return pd.DataFrame(contracts)
    except Exception as e:
        print(f"获取期权合约失败: {e}")
        return None

def analyze_rubber_options():
    """分析橡胶期权"""
    print("=" * 80)
    print(f"橡胶期权分析 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    print("\n当前橡胶期货价格: 约 17,240")
    print("主力合约: ru2505")
    print("\n获取期权合约数据...\n")
    
    df = get_rubber_option_contracts()
    
    if df is None or df.empty:
        print("未获取到期权数据")
        return
    
    print(f"获取到 {len(df)} 个期权合约:\n")
    print(df.to_string(index=False))
    
    # 分析
    print("\n" + "=" * 80)
    print("策略分析")
    print("=" * 80)
    
    # 塔勒布式买入Put分析
    puts = df[df['type'] == '认沽']
    if not puts.empty:
        print("\n【买入Put策略 - 塔勒布式】")
        print("-" * 60)
        
        for _, row in puts.iterrows():
            strike = row['strike']
            price = row['close']
            
            # 计算盈亏比（假设橡胶跌到15000）
            target_price = 15000
            if strike > target_price:
                intrinsic_value = strike - target_price
                # 橡胶期权合约乘数是10吨/手
                profit = (intrinsic_value - price) * 10
                cost = price * 10
                reward_ratio = profit / cost if cost > 0 else 0
                
                print(f"\n行权价 {strike} Put:")
                print(f"  权利金: {price} 元/吨 = {cost:.0f} 元/手")
                print(f"  若跌至15000，内在价值: {intrinsic_value} 元/吨")
                print(f"  盈亏: {profit:.0f} 元/手")
                print(f"  盈亏比: {reward_ratio:.1f}x")
    
    # 顺势买入Call分析（基于突破信号）
    calls = df[df['type'] == '认购']
    if not calls.empty:
        print("\n【买入Call策略 - 顺势突破】")
        print("-" * 60)
        
        # 推荐虚值2-3档
        recommended = calls[calls['strike'] >= 17500]
        
        for _, row in recommended.iterrows():
            strike = row['strike']
            price = row['close']
            cost = price * 10
            
            # 计算涨到19000的盈亏
            target_price = 19000
            if strike < target_price:
                intrinsic_value = target_price - strike
                profit = (intrinsic_value - price) * 10
                reward_ratio = profit / cost if cost > 0 else 0
                
                print(f"\n行权价 {strike} Call:")
                print(f"  权利金: {price} 元/吨 = {cost:.0f} 元/手")
                print(f"  若涨至19000，盈亏比: {reward_ratio:.1f}x")

if __name__ == "__main__":
    analyze_rubber_options()
