#!/usr/bin/env python3
"""
获取ETF期权历史数据并分析
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime

# ETF期权标的
ETF_OPTIONS = {
    '510050': '50ETF',
    '510300': '300ETF',
    '510500': '500ETF',
    '159915': '创业板ETF',
    '588000': '科创50ETF',
}

def get_etf_option_hist(symbol, expiry='2503'):
    """获取ETF期权历史数据（新浪接口）"""
    try:
        # 构造期权代码格式：如 510050C2503M02700
        # 获取不同行权价的期权
        
        # 先获取当前ETF价格来估算平值期权
        from akshare import fund_etf_hist_em
        etf_df = fund_etf_hist_em(symbol=symbol, period="daily", adjust="qfq")
        if etf_df.empty:
            return None
        
        current_price = etf_df['收盘'].iloc[-1]
        
        # 根据价格确定行权价范围
        if symbol == '510050':  # 50ETF
            base = round(current_price / 0.05) * 0.05
            strikes = [base - 0.15, base - 0.1, base - 0.05, base, base + 0.05, base + 0.1, base + 0.15]
        elif symbol == '510300':  # 300ETF
            base = round(current_price / 0.05) * 0.05
            strikes = [base - 0.3, base - 0.2, base - 0.1, base, base + 0.1, base + 0.2, base + 0.3]
        elif symbol == '588000':  # 科创50ETF
            base = round(current_price / 0.05) * 0.05
            strikes = [base - 0.1, base - 0.05, base, base + 0.05, base + 0.1]
        else:
            base = round(current_price / 0.05) * 0.05
            strikes = [base - 0.2, base - 0.1, base, base + 0.1, base + 0.2]
        
        contracts = []
        
        for strike in strikes:
            strike_str = f"{strike:.4f}".replace('.', '').zfill(5) if '.' in f"{strike:.4f}" else f"{int(strike)}00"
            
            # 认购期权
            try:
                call_code = f"{symbol}C{expiry}M{strike_str}"
                df_call = ak.option_commodity_hist_sina(symbol=call_code)
                if not df_call.empty:
                    latest = df_call.iloc[-1]
                    contracts.append({
                        'code': call_code,
                        'underlying': symbol,
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
                put_code = f"{symbol}P{expiry}M{strike_str}"
                df_put = ak.option_commodity_hist_sina(symbol=put_code)
                if not df_put.empty:
                    latest = df_put.iloc[-1]
                    contracts.append({
                        'code': put_code,
                        'underlying': symbol,
                        'type': '认沽',
                        'strike': strike,
                        'close': latest['close'],
                        'volume': latest['volume'],
                        'date': latest['date']
                    })
            except:
                pass
        
        return pd.DataFrame(contracts), current_price
    except Exception as e:
        print(f"  获取失败: {e}")
        return None, None

def analyze_etf_option(symbol, name):
    """分析单个ETF期权"""
    print(f"\n{'='*80}")
    print(f"{name} ({symbol}) 期权分析")
    print(f"{'='*80}")
    
    df, current_price = get_etf_option_hist(symbol)
    
    if df is None or df.empty:
        print("未获取到期权数据")
        return
    
    print(f"\n当前ETF价格: {current_price:.3f}")
    print(f"获取到 {len(df)} 个期权合约\n")
    
    # 显示合约
    print(df.to_string(index=False))
    
    # 策略分析
    print(f"\n{'='*80}")
    print("策略分析")
    print(f"{'='*80}")
    
    # 买入Call分析（顺势）
    calls = df[df['type'] == '认购']
    if not calls.empty:
        print("\n【买入Call策略 - 顺势突破】")
        print("-" * 60)
        
        # 选择虚值2-3档
        otm_calls = calls[calls['strike'] > current_price].sort_values('strike')
        
        for _, row in otm_calls.head(3).iterrows():
            strike = row['strike']
            price = row['close']
            
            # 假设涨5%的盈亏
            target = current_price * 1.05
            if strike < target:
                intrinsic = target - strike
                profit = (intrinsic - price) * 10000  # ETF期权乘数10000
                cost = price * 10000
                ratio = profit / cost if cost > 0 else 0
                
                print(f"\n行权价 {strike:.3f} Call:")
                print(f"  权利金: {price:.4f} = {cost:.0f}元/张")
                print(f"  若涨5%到{target:.3f}，盈亏比: {ratio:.1f}x")
    
    # 买入Put分析（塔勒布式）
    puts = df[df['type'] == '认沽']
    if not puts.empty:
        print("\n【买入Put策略 - 塔勒布式】")
        print("-" * 60)
        
        # 选择虚值2-3档
        otm_puts = puts[puts['strike'] < current_price].sort_values('strike', ascending=False)
        
        for _, row in otm_puts.head(3).iterrows():
            strike = row['strike']
            price = row['close']
            
            # 假设跌10%的盈亏
            target = current_price * 0.90
            if strike > target:
                intrinsic = strike - target
                profit = (intrinsic - price) * 10000
                cost = price * 10000
                ratio = profit / cost if cost > 0 else 0
                
                print(f"\n行权价 {strike:.3f} Put:")
                print(f"  权利金: {price:.4f} = {cost:.0f}元/张")
                print(f"  若跌10%到{target:.3f}，盈亏比: {ratio:.1f}x")

def main():
    print("=" * 80)
    print(f"ETF期权历史数据分析 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    
    for symbol, name in ETF_OPTIONS.items():
        try:
            analyze_etf_option(symbol, name)
        except Exception as e:
            print(f"\n{name} 分析失败: {e}")

if __name__ == "__main__":
    main()
