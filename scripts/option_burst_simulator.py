#!/usr/bin/env python3
"""
期权爆发策略 - 基于图中案例的模拟回测
验证复合买入 + Sigma激进止盈策略
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class OptionBurstSimulator:
    """
    模拟图中500ETF购3月9000的走势
    验证策略逻辑
    """
    
    def __init__(self):
        self.trades = []
        self.position = 0
        self.entry_price = 0
        self.sigma_sold = set()
        
    def generate_simulated_data(self):
        """生成模拟数据（类似图中走势）"""
        # 模拟图中走势：从0.08爆发到0.29
        data = []
        
        # 阶段1：高位回落后横盘（0.08附近）
        base_price = 0.08
        for i in range(20):
            price = base_price + np.random.normal(0, 0.005)
            sigma = np.random.normal(-0.5, 0.3)
            volume = np.random.normal(1000, 200)
            data.append({
                'datetime': datetime(2026, 1, 14, 9, 30) + timedelta(minutes=30*i),
                'close': max(0.01, price),
                'high': max(0.01, price + 0.005),
                'low': max(0.01, price - 0.005),
                'volume': max(100, volume),
                'sigma': sigma,
                'ma20': 0.075,
                'volume_ma20': 1500
            })
        
        # 阶段2：突破前高，开始上涨（0.08 -> 0.15）
        for i in range(10):
            progress = i / 10
            price = 0.08 + (0.15 - 0.08) * progress + np.random.normal(0, 0.008)
            sigma = 2 + progress * 2 + np.random.normal(0, 0.5)
            volume = 2000 + progress * 1000 + np.random.normal(0, 300)
            data.append({
                'datetime': datetime(2026, 1, 14, 19, 30) + timedelta(minutes=30*i),
                'close': max(0.01, price),
                'high': max(0.01, price + 0.01),
                'low': max(0.01, price - 0.008),
                'volume': max(500, volume),
                'sigma': sigma,
                'ma20': 0.10,
                'volume_ma20': 1500
            })
        
        # 阶段3：加速上涨（0.15 -> 0.29）
        for i in range(15):
            progress = i / 15
            # 非线性上涨，后期加速
            price = 0.15 + (0.29 - 0.15) * (progress ** 0.7) + np.random.normal(0, 0.01)
            sigma = 4 + progress * 10 + np.random.normal(0, 1)
            volume = 3000 + np.random.normal(0, 500)
            data.append({
                'datetime': datetime(2026, 1, 15, 9, 30) + timedelta(minutes=30*i),
                'close': max(0.01, min(0.30, price)),
                'high': max(0.01, min(0.30, price + 0.015)),
                'low': max(0.01, min(0.30, price - 0.01)),
                'volume': max(500, volume),
                'sigma': min(15, sigma),
                'ma20': 0.15,
                'volume_ma20': 2000
            })
        
        # 阶段4：高位回落
        for i in range(10):
            progress = i / 10
            price = 0.29 - (0.29 - 0.18) * progress + np.random.normal(0, 0.008)
            sigma = 12 - progress * 8 + np.random.normal(0, 0.5)
            volume = 2500 + np.random.normal(0, 400)
            data.append({
                'datetime': datetime(2026, 1, 15, 17, 0) + timedelta(minutes=30*i),
                'close': max(0.01, price),
                'high': max(0.01, price + 0.01),
                'low': max(0.01, price - 0.012),
                'volume': max(500, volume),
                'sigma': max(2, sigma),
                'ma20': 0.22,
                'volume_ma20': 2200
            })
        
        return pd.DataFrame(data)
    
    def check_buy_signals(self, row, prev_row):
        """检查买入信号"""
        signals = []
        
        # 信号1：回踩中轨 + 缩量
        if (abs(row['close'] - row['ma20']) / row['ma20'] < 0.05 and
            row['volume'] < row['volume_ma20'] * 0.8 and
            row['sigma'] > -1 and row['sigma'] < 1):
            signals.append(('中轨企稳', 0.30))
        
        # 信号2：突破前高（简化：比前一根高且放量）
        if (row['close'] > prev_row['close'] * 1.03 and
            row['volume'] > row['volume_ma20'] * 1.3):
            signals.append(('突破上涨', 0.40))
        
        # 信号3：放量确认（已有仓位）
        if (self.position > 0 and
            row['volume'] > row['volume_ma20'] * 1.5 and
            row['close'] > prev_row['close']):
            signals.append(('放量确认', 0.30))
        
        return signals
    
    def check_sell_signals(self, row):
        """检查卖出信号（Sigma激进止盈）"""
        if self.position <= 0:
            return []
        
        signals = []
        sigma = row['sigma']
        pnl = (row['close'] - self.entry_price) / self.entry_price if self.entry_price > 0 else 0
        
        # Sigma止盈（激进版）
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
        if pnl < -0.50:
            signals.append(('亏损50%止损', 1.0))
        
        return signals
    
    def run_backtest(self):
        """运行回测"""
        print("="*80)
        print("期权爆发策略 - 模拟回测")
        print("="*80)
        
        df = self.generate_simulated_data()
        print(f"\n模拟数据: {len(df)} 条")
        print(f"价格范围: {df['close'].min():.4f} - {df['close'].max():.4f}")
        print(f"Sigma范围: {df['sigma'].min():.2f} - {df['sigma'].max():.2f}")
        
        print("\n" + "-"*80)
        print("交易记录")
        print("-"*80)
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # 买入
            buy_signals = self.check_buy_signals(row, prev_row)
            for signal_name, target_pos in buy_signals:
                if self.position < target_pos:
                    buy_pct = target_pos - self.position
                    self.position = target_pos
                    if self.entry_price == 0:
                        self.entry_price = row['close']
                    
                    self.trades.append({
                        'time': row['datetime'].strftime('%m-%d %H:%M'),
                        'action': '买入',
                        'signal': signal_name,
                        'price': row['close'],
                        'position': self.position,
                        'sigma': row['sigma']
                    })
                    print(f"{row['datetime'].strftime('%m-%d %H:%M')} 买入[{signal_name}] 价格:{row['close']:.4f} 仓位:{self.position*100:.0f}% σ:{row['sigma']:.2f}")
            
            # 卖出
            sell_signals = self.check_sell_signals(row)
            for signal_name, sell_pct in sell_signals:
                if self.position > 0:
                    actual_sell = min(sell_pct, self.position)
                    pnl = (row['close'] - self.entry_price) / self.entry_price * 100
                    
                    self.trades.append({
                        'time': row['datetime'].strftime('%m-%d %H:%M'),
                        'action': '卖出',
                        'signal': signal_name,
                        'price': row['close'],
                        'pnl': pnl,
                        'sigma': row['sigma']
                    })
                    
                    print(f"{row['datetime'].strftime('%m-%d %H:%M')} 卖出[{signal_name}] 价格:{row['close']:.4f} 盈亏:{pnl:+.1f}% σ:{row['sigma']:.2f}")
                    
                    self.position -= actual_sell
                    if self.position <= 0.1:
                        self.position = 0
        
        self.generate_report()
        return self.trades
    
    def generate_report(self):
        """生成报告"""
        print("\n" + "="*80)
        print("回测报告")
        print("="*80)
        
        buy_trades = [t for t in self.trades if t['action'] == '买入']
        sell_trades = [t for t in self.trades if t['action'] == '卖出']
        
        print(f"\n总交易: {len(buy_trades)}次买入, {len(sell_trades)}次卖出")
        
        if sell_trades:
            final_pnl = sell_trades[-1]['pnl']
            max_pnl = max([t['pnl'] for t in sell_trades])
            
            print(f"\n最终盈亏: {final_pnl:+.1f}%")
            print(f"最大浮盈: {max_pnl:+.1f}%")
            
            # 统计各sigma卖出
            sigma_sells = {}
            for t in sell_trades:
                sig = t['signal']
                sigma_sells[sig] = sigma_sells.get(sig, 0) + 1
            
            print(f"\nSigma卖出统计:")
            for sig, count in sigma_sells.items():
                print(f"  {sig}: {count}次")
        
        print("\n" + "="*80)

def main():
    sim = OptionBurstSimulator()
    trades = sim.run_backtest()
    
    # 保存结果
    with open('backtest_results/option_burst_simulation.json', 'w') as f:
        json.dump(trades, f, indent=2)
    print("\n结果已保存: backtest_results/option_burst_simulation.json")

if __name__ == "__main__":
    main()
