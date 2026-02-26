#!/usr/bin/env python3
"""
使用AKShare实时数据回测30分钟趋势通道突破策略
备用方案：使用stock_zh_a_hist接口
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# ETF期权标的配置（使用股票代码格式）
ETF_UNDERLYINGS = {
    'sh510050': '50ETF',
    'sh510300': '300ETF', 
    'sh510500': '500ETF',
    'sz159915': '创业板ETF',
    'sh588000': '科创50ETF',
    'sh588080': '科创板50ETF',
}

def get_etf_daily_data(symbol):
    """获取ETF日线数据"""
    try:
        # 转换代码格式
        if symbol.startswith('sh'):
            code = symbol[2:] + '.SH'
        elif symbol.startswith('sz'):
            code = symbol[2:] + '.SZ'
        else:
            code = symbol
        
        # 获取日线数据
        df = ak.fund_etf_hist_em(symbol=code.split('.')[0], period="daily", adjust="qfq")
        
        if df is None or df.empty:
            return None
        
        # 标准化列名
        df.columns = [col.lower() for col in df.columns]
        
        # 转换数值
        for col in ['close', 'high', 'low', 'open', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['close'])
        return df.sort_values('date').reset_index(drop=True)
    except Exception as e:
        print(f"  获取失败: {e}")
        return None

def calculate_bollinger_bands(df, period=20, std_dev=2.0):
    """计算布林带"""
    df['ma'] = df['close'].rolling(window=period).mean()
    df['std'] = df['close'].rolling(window=period).std()
    df['upper'] = df['ma'] + std_dev * df['std']
    df['lower'] = df['ma'] - std_dev * df['std']
    return df

def check_bollinger_breakout(df, period=20):
    """检查布林带突破信号"""
    if df is None or len(df) < period + 5:
        return None
    
    df = calculate_bollinger_bands(df, period)
    
    # 获取最近3天数据
    recent = df.tail(3)
    if len(recent) < 3:
        return None
    
    latest = recent.iloc[-1]
    prev = recent.iloc[-2]
    
    signal = {
        'date': latest.get('date', ''),
        'close': latest['close'],
        'upper': latest['upper'],
        'lower': latest['lower'],
        'ma': latest['ma'],
        'z_score': (latest['close'] - latest['ma']) / latest['std'] if latest['std'] > 0 else 0,
        'breakout_up': latest['close'] > latest['upper'],
        'breakout_down': latest['close'] < latest['lower'],
        'confirmed_up': prev['close'] > prev['upper'] if not pd.isna(prev['upper']) else False,
        'confirmed_down': prev['close'] < prev['lower'] if not pd.isna(prev['lower']) else False,
    }
    
    return signal

def scan_etf_options():
    """扫描所有ETF期权标的"""
    results = []
    
    print("=" * 80)
    print(f"ETF期权布林带突破扫描 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    print(f"共扫描 {len(ETF_UNDERLYINGS)} 个标的\n")
    
    for symbol, name in ETF_UNDERLYINGS.items():
        print(f"扫描 {symbol} ({name})...", end=" ")
        
        try:
            df = get_etf_daily_data(symbol)
            
            if df is None or df.empty:
                print("❌ 无数据")
                continue
            
            signal = check_bollinger_breakout(df)
            if signal is None:
                print("❌ 数据不足")
                continue
            
            # 判断信号
            if signal['breakout_up']:
                if signal['confirmed_up']:
                    status = "✅ 确认突破上轨"
                else:
                    status = "⚠️ 首次突破上轨"
                results.append({
                    'symbol': symbol,
                    'name': name,
                    'signal': '突破上轨',
                    'confirmed': signal['confirmed_up'],
                    **signal
                })
            elif signal['breakout_down']:
                if signal['confirmed_down']:
                    status = "✅ 确认突破下轨"
                else:
                    status = "⚠️ 首次突破下轨"
                results.append({
                    'symbol': symbol,
                    'name': name,
                    'signal': '突破下轨',
                    'confirmed': signal['confirmed_down'],
                    **signal
                })
            else:
                # 计算距离上轨/下轨的距离
                dist_upper = (signal['upper'] - signal['close']) / signal['close'] * 100
                dist_lower = (signal['close'] - signal['lower']) / signal['close'] * 100
                status = f"无信号 (距上轨{dist_upper:.1f}%, 距下轨{dist_lower:.1f}%)"
            
            print(status)
            
        except Exception as e:
            print(f"❌ 错误: {e}")
        
        time.sleep(0.5)
    
    # 输出结果
    print("\n" + "=" * 80)
    print("扫描结果 - 布林带突破信号")
    print("=" * 80)
    
    if not results:
        print("\n暂无品种出现布林带突破信号")
    else:
        print(f"\n共 {len(results)} 个标的出现信号:\n")
        print(f"{'代码':<12} {'名称':<12} {'信号':<12} {'收盘价':<10} {'Z分数':<8} {'确认':<6}")
        print("-" * 70)
        
        for r in results:
            confirm = "✅" if r['confirmed'] else "⚠️"
            print(f"{r['symbol']:<12} {r['name']:<12} {r['signal']:<12} {r['close']:<10.3f} {r['z_score']:<8.2f} {confirm:<6}")
    
    return results

def backtest_strategy(symbol, name, lookback_days=60):
    """回测最近N天的策略表现"""
    print(f"\n{'='*80}")
    print(f"回测 {name} ({symbol}) 最近{lookback_days}天")
    print(f"{'='*80}")
    
    try:
        df = get_etf_daily_data(symbol)
        if df is None or len(df) < lookback_days:
            print("数据不足")
            return
        
        # 只取最近N天
        df = df.tail(lookback_days)
        df = calculate_bollinger_bands(df)
        
        # 模拟交易
        trades = []
        position = 0
        entry_price = 0
        
        for i in range(20, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            # 买入信号：突破上轨
            if position == 0 and row['close'] > row['upper']:
                position = 1
                entry_price = row['close']
                trades.append({
                    'date': row.get('date', ''),
                    'action': '买入',
                    'price': row['close'],
                    'z_score': (row['close'] - row['ma']) / row['std']
                })
            
            # 卖出信号：跌破中轨或持有5天
            elif position == 1:
                exit_reason = None
                if row['close'] < row['ma']:
                    exit_reason = '跌破中轨'
                elif len(trades) > 0 and i - df.index[df['date'] == trades[-1]['date']].tolist()[0] >= 5:
                    exit_reason = '时间止损'
                
                if exit_reason:
                    pnl = (row['close'] - entry_price) / entry_price * 100
                    trades.append({
                        'date': row.get('date', ''),
                        'action': '卖出',
                        'price': row['close'],
                        'reason': exit_reason,
                        'pnl': pnl
                    })
                    position = 0
        
        # 统计结果
        if len(trades) > 0:
            buy_trades = [t for t in trades if t['action'] == '买入']
            sell_trades = [t for t in trades if t['action'] == '卖出']
            
            print(f"\n总交易次数: {len(buy_trades)} 次")
            print(f"完整交易: {len(sell_trades)} 次")
            
            if sell_trades:
                wins = [t for t in sell_trades if t['pnl'] > 0]
                losses = [t for t in sell_trades if t['pnl'] <= 0]
                win_rate = len(wins) / len(sell_trades) * 100
                avg_pnl = sum([t['pnl'] for t in sell_trades]) / len(sell_trades)
                
                print(f"胜率: {win_rate:.1f}% ({len(wins)}胜/{len(losses)}负)")
                print(f"平均盈亏: {avg_pnl:.2f}%")
                print(f"\n最近5笔交易:")
                for t in trades[-10:]:
                    if t['action'] == '买入':
                        print(f"  {t['date']}: 买入 @ {t['price']:.3f} (Z={t['z_score']:.2f})")
                    else:
                        print(f"  {t['date']}: 卖出 @ {t['price']:.3f} ({t['reason']}) 盈亏: {t['pnl']:+.2f}%")
        else:
            print("\n最近无交易信号")
            
    except Exception as e:
        print(f"回测失败: {e}")

if __name__ == "__main__":
    # 扫描当前信号
    results = scan_etf_options()
    
    # 回测最强信号标的
    if results:
        top = results[0]
        backtest_strategy(top['symbol'], top['name'])
    else:
        # 回测50ETF作为示例
        backtest_strategy('sh510050', '50ETF')
