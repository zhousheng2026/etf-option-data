#!/usr/bin/env python3
"""
SigmaBurst策略 - 修正版

Sigma计算修正：
- 用爆发前的布林带参数作为固定基准
- 爆发前 = 横盘整理期（中轨企稳阶段）的布林带
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class SigmaBurstStrategyV2:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.sigma_sold = set()
        
        # 固定基准（爆发前的布林带参数）
        self.base_ma20 = None
        self.base_std20 = None
        
    def calculate_indicators(self, df):
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['upper'] = df['ma20'] + 2 * df['std20']
        df['lower'] = df['ma20'] - 2 * df['std20']
        df['sigma_dynamic'] = (df['close'] - df['ma20']) / df['std20']
        
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['prev_high'] = df['high'].rolling(window=20).max().shift(1)
        
        return df
    
    def calculate_fixed_sigma(self, current_price):
        """用固定基准计算Sigma"""
        if self.base_std20 and self.base_std20 > 0:
            return (current_price - self.base_ma20) / self.base_std20
        return 0
    
    def check_buy_signals(self, df, idx):
        if idx < 20:
            return []
        
        row = df.iloc[idx]
        prev = df.iloc[idx-1]
        signals = []
        
        # 中轨附近判断
        near_ma = abs(row['close'] - row['ma20']) / row['ma20'] < 0.05
        low_volume = row['volume'] < row['volume_ma20'] * 0.7
        
        # 信号1：中轨企稳（设置固定基准）
        if (near_ma and low_volume and 
            -0.5 < row['sigma_dynamic'] < 0.5 and
            self.position < 0.3):
            
            # 设置固定基准（横盘期的布林带参数）
            self.base_ma20 = row['ma20']
            self.base_std20 = row['std20']
            
            signals.append(('中轨企稳', 0.30))
        
        # 信号2：突破前高
        if (row['close'] > row['prev_high'] and
            row['volume'] > row['volume_ma20'] * 1.3 and
            self.position < 0.7):
            signals.append(('突破前高', 0.70))
        
        # 信号3：放量确认
        if (self.position > 0 and
            row['volume'] > row['volume_ma20'] * 1.5 and
            row['close'] > prev['close'] and
            self.position < 1.0):
            signals.append(('放量确认', 1.0))
        
        return signals
    
    def check_sell_signals(self, row):
        if self.position <= 0:
            return [], 0
        
        signals = []
        current_price = row['close']
        
        # 用固定基准计算Sigma
        sigma = self.calculate_fixed_sigma(current_price)
        pnl = (current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0
        
        # Sigma止盈
        if sigma >= 12 and '12sigma' not in self.sigma_sold:
            signals.append(('12σ清仓', 0.90))
            self.sigma_sold.add('12sigma')
        elif sigma >= 10 and '10sigma' not in self.sigma_sold:
            signals.append(('10σ减仓', 0.30))
            self.sigma_sold.add('10sigma')
        elif sigma >= 8 and '8sigma' not in self.sigma_sold:
            signals.append(('8σ减仓', 0.30))
            self.sigma_sold.add('8sigma')
        elif sigma >= 6 and '6sigma' not in self.sigma_sold:
            signals.append(('6σ减仓', 0.20))
            self.sigma_sold.add('6sigma')
        
        # 止损
        if current_price < row['ma20'] and pnl < -0.20:
            signals.append(('跌破中轨止损', 1.0))
        if pnl < -0.50:
            signals.append(('亏损50%止损', 1.0))
        
        return signals, sigma
    
    def backtest(self, df, verbose=True):
        df = self.calculate_indicators(df)
        
        if verbose:
            print("="*60)
            print("SigmaBurst V2 - 回测")
            print("="*60)
        
        for i in range(20, len(df)):
            row = df.iloc[i]
            
            # 买入
            buy_signals = self.check_buy_signals(df, i)
            for signal_name, target_pos in buy_signals:
                if self.position < target_pos:
                    self.position = target_pos
                    if self.entry_price == 0:
                        self.entry_price = row['close']
                    
                    fixed_sigma = self.calculate_fixed_sigma(row['close'])
                    self.trades.append({
                        'datetime': str(row['datetime']),
                        'action': '买入',
                        'signal': signal_name,
                        'price': float(row['close']),
                        'position': float(self.position),
                        'fixed_sigma': float(fixed_sigma)
                    })
                    
                    if verbose:
                        print(f"{row['datetime']} 买入[{signal_name}] 价格:{row['close']:.4f} 仓位:{self.position*100:.0f}% 固定σ:{fixed_sigma:.2f}")
            
            # 卖出
            sell_signals, sigma = self.check_sell_signals(row)
            for signal_name, sell_pct in sell_signals:
                if self.position > 0:
                    actual_sell = min(sell_pct, self.position)
                    pnl = (row['close'] - self.entry_price) / self.entry_price
                    
                    self.trades.append({
                        'datetime': str(row['datetime']),
                        'action': '卖出',
                        'signal': signal_name,
                        'price': float(row['close']),
                        'pnl': float(pnl * 100),
                        'fixed_sigma': float(sigma)
                    })
                    
                    if verbose:
                        print(f"{row['datetime']} 卖出[{signal_name}] 价格:{row['close']:.4f} 盈亏:{pnl*100:+.1f}% 固定σ:{sigma:.2f}")
                    
                    self.position -= actual_sell
                    if self.position <= 0.1:
                        self.position = 0
                        self.entry_price = 0
                        self.sigma_sold.clear()
        
        return self.generate_report(verbose)
    
    def generate_report(self, verbose=True):
        buy_trades = [t for t in self.trades if t['action'] == '买入']
        sell_trades = [t for t in self.trades if t['action'] == '卖出']
        
        if not sell_trades:
            if verbose:
                print("\n无完整交易")
            return {'trades': self.trades, 'summary': {}}
        
        final_pnl = sell_trades[-1]['pnl']
        max_pnl = max([t['pnl'] for t in sell_trades])
        
        summary = {
            'total_buys': len(buy_trades),
            'total_sells': len(sell_trades),
            'final_pnl': round(final_pnl, 2),
            'max_pnl': round(max_pnl, 2)
        }
        
        if verbose:
            print("\n" + "="*60)
            print(f"最终盈亏: {final_pnl:+.2f}%")
            print(f"最大浮盈: {max_pnl:+.2f}%")
            print("="*60)
        
        return {'trades': self.trades, 'summary': summary}

def main():
    # 生成测试数据
    data = []
    base_time = datetime(2026, 1, 14, 9, 30)
    
    # 前30根：建立基准（标准差约0.003）
    np.random.seed(123)
    for i in range(30):
        price = 0.08 + np.random.normal(0, 0.003)
        data.append({
            'datetime': base_time + timedelta(minutes=30*i),
            'open': price-0.001, 'high': price+0.003, 'low': price-0.003, 'close': price,
            'volume': 1000
        })
    
    # 横盘企稳（10根，缩量，中轨附近）
    for i in range(10):
        data.append({
            'datetime': data[-1]['datetime'] + timedelta(minutes=30),
            'open': 0.078, 'high': 0.081, 'low': 0.077, 'close': 0.079,
            'volume': 600
        })
    
    # 突破
    for i in range(5):
        price = 0.08 + 0.005 * (i+1)
        data.append({
            'datetime': data[-1]['datetime'] + timedelta(minutes=30),
            'open': price, 'high': price+0.004, 'low': price-0.002, 'close': price+0.003,
            'volume': 2000
        })
    
    # 爆发到0.29
    burst_prices = [0.10, 0.12, 0.15, 0.18, 0.21, 0.24, 0.26, 0.28, 0.29, 0.285, 0.28, 0.27]
    for price in burst_prices:
        data.append({
            'datetime': data[-1]['datetime'] + timedelta(minutes=30),
            'open': price-0.005, 'high': price+0.008, 'low': price-0.008, 'close': price,
            'volume': 4000
        })
    
    df = pd.DataFrame(data)
    
    # 运行回测
    strategy = SigmaBurstStrategyV2(initial_capital=100000)
    result = strategy.backtest(df, verbose=True)
    
    # 验证Sigma计算
    print("\n" + "="*60)
    print("验证Sigma计算:")
    print("="*60)
    entry_idx = 40  # 入场点
    entry_ma = df.iloc[entry_idx:entry_idx+5]['close'].mean()  # 横盘期中轨
    entry_std = df.iloc[entry_idx:entry_idx+5]['close'].std()   # 横盘期标准差
    high_price = df['close'].max()
    
    print(f"横盘期中轨: {entry_ma:.4f}")
    print(f"横盘期标准差: {entry_std:.5f}")
    print(f"最高价格: {high_price:.4f}")
    print(f"固定Sigma: ({high_price:.4f} - {entry_ma:.4f}) / {entry_std:.5f} = {(high_price-entry_ma)/entry_std:.2f}σ")

if __name__ == "__main__":
    main()
