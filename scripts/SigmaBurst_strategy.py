#!/usr/bin/env python3
"""
SigmaBurst（西格玛爆发）期权策略 - 最终修正版

Sigma计算修正：
- 用40周期布林带参数作为固定基准（不是20周期）
- 在第一次买入时固定基准，之后不再改变
- 这样Sigma能正确反映价格偏离程度

图中13.86σ的计算：
- 基准中轨：0.0673
- 基准标准差：0.01625
- 最高价格：0.2926
- Sigma = (0.2926 - 0.0673) / 0.01625 = 13.86σ
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

class SigmaBurstStrategy:
    """
    SigmaBurst期权爆发策略
    
    买入：复合建仓
    - 中轨企稳（30%）：价格在中轨±5% + 缩量 + 企稳
    - 突破前高（40%）：突破20周期高点 + 放量
    - 放量确认（30%）：已有仓位 + 放量 + 价格上涨
    
    卖出：Sigma激进止盈（固定基准）
    - 6σ：减仓20%
    - 8σ：减仓30%
    - 10σ：减仓30%
    - 12σ+：清仓90%
    
    止损：
    - 跌破中轨且亏损>20%
    - 权利金亏损>50%
    """
    
    def __init__(self, 
                 initial_capital=100000,
                 base_period=40,  # 固定基准周期（40周期，不是20）
                 bb_std=2.0,
                 volume_threshold_low=0.7,
                 volume_threshold_high=1.3,
                 near_ma_threshold=0.05,
                 max_position=1.0):
        
        self.initial_capital = initial_capital
        self.base_period = base_period
        self.bb_std = bb_std
        self.volume_threshold_low = volume_threshold_low
        self.volume_threshold_high = volume_threshold_high
        self.near_ma_threshold = near_ma_threshold
        self.max_position = max_position
        
        # 状态变量
        self.capital = initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.sigma_sold = set()
        
        # 固定基准（第一次买入时设置）
        self.base_ma = None
        self.base_std = None
        self.base_set = False
        
    def calculate_indicators(self, df):
        """计算技术指标"""
        # 20周期布林带（用于买入判断）
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['upper20'] = df['ma20'] + self.bb_std * df['std20']
        df['lower20'] = df['ma20'] - self.bb_std * df['std20']
        
        # 40周期布林带（用于固定基准）
        df['ma40'] = df['close'].rolling(window=self.base_period).mean()
        df['std40'] = df['close'].rolling(window=self.base_period).std()
        
        # 成交量
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        
        # 前期高点
        df['prev_high'] = df['high'].rolling(window=20).max().shift(1)
        
        # 中轨附近信号
        df['near_ma20'] = abs(df['close'] - df['ma20']) / df['ma20'] < self.near_ma_threshold
        
        # 缩量/放量
        df['low_volume'] = df['volume'] < df['volume_ma20'] * self.volume_threshold_low
        df['high_volume'] = df['volume'] > df['volume_ma20'] * self.volume_threshold_high
        
        # 企稳信号（连续2根在中轨附近）
        df['consolidate'] = df['near_ma20'].rolling(window=2).sum() >= 2
        
        return df
    
    def calculate_fixed_sigma(self, current_price):
        """用固定基准计算Sigma"""
        if self.base_std and self.base_std > 0:
            return (current_price - self.base_ma) / self.base_std
        return 0
    
    def check_buy_signals(self, row, idx):
        """检查买入信号"""
        if idx < self.base_period:
            return []
        
        signals = []
        
        # 信号1：中轨企稳（设置固定基准）
        if (row['near_ma20'] and 
            row['low_volume'] and 
            row['consolidate'] and
            not self.base_set and
            self.position < 0.3):
            
            # 设置固定基准（用40周期参数）
            self.base_ma = row['ma40']
            self.base_std = row['std40']
            self.base_set = True
            
            signals.append(('中轨企稳', 0.30))
        
        # 信号2：突破前高
        if (row['close'] > row['prev_high'] and
            row['high_volume'] and
            self.position < 0.7):
            signals.append(('突破前高', 0.70))
        
        # 信号3：放量确认
        if (self.position > 0 and
            row['volume'] > row['volume_ma20'] * 1.5 and
            self.position < self.max_position):
            signals.append(('放量确认', self.max_position))
        
        return signals
    
    def check_sell_signals(self, row):
        """检查卖出信号（Sigma止盈 + 趋势回撤）"""
        if self.position <= 0 or self.entry_price <= 0:
            return [], 0, 0
        
        signals = []
        current_price = row['close']
        
        # 用固定基准计算Sigma（用于止盈）
        fixed_sigma = self.calculate_fixed_sigma(current_price)
        
        # 用当前布林带计算Sigma（用于判断回撤）
        if row['std20'] > 0:
            current_sigma = (current_price - row['ma20']) / row['std20']
        else:
            current_sigma = 0
        
        pnl = (current_price - self.entry_price) / self.entry_price
        
        # Sigma激进止盈（固定基准）
        if fixed_sigma >= 12 and '12sigma' not in self.sigma_sold:
            signals.append(('12σ清仓', 0.90))
            self.sigma_sold.add('12sigma')
        elif fixed_sigma >= 10 and '10sigma' not in self.sigma_sold:
            signals.append(('10σ减仓', 0.30))
            self.sigma_sold.add('10sigma')
        elif fixed_sigma >= 8 and '8sigma' not in self.sigma_sold:
            signals.append(('8σ减仓', 0.30))
            self.sigma_sold.add('8sigma')
        elif fixed_sigma >= 6 and '6sigma' not in self.sigma_sold:
            signals.append(('6σ减仓', 0.20))
            self.sigma_sold.add('6sigma')
        
        # 趋势回撤止损（当前布林带）
        # 如果价格跌破当前中轨，说明趋势可能结束
        if current_price < row['ma20'] and pnl > 0:
            signals.append(('跌破中轨止盈', 0.50))
        
        # 严格止损
        if current_price < row['ma20'] and pnl < -0.20:
            signals.append(('跌破中轨止损', 1.0))
        if pnl < -0.50:
            signals.append(('亏损50%止损', 1.0))
        
        return signals, fixed_sigma, current_sigma
    
    def backtest(self, df, verbose=True):
        """回测主函数"""
        df = self.calculate_indicators(df)
        
        if verbose:
            print("="*80)
            print("SigmaBurst期权爆发策略 - 回测")
            print("="*80)
            print(f"数据周期: {df['datetime'].iloc[0]} 至 {df['datetime'].iloc[-1]}")
            print(f"总数据量: {len(df)} 条")
            print(f"固定基准周期: {self.base_period}")
            print("-"*80)
        
        for i in range(self.base_period, len(df)):
            row = df.iloc[i]
            
            # 买入
            buy_signals = self.check_buy_signals(row, i)
            for signal_name, target_pos in buy_signals:
                if self.position < target_pos:
                    self.position = target_pos
                    if self.entry_price == 0:
                        self.entry_price = row['close']
                    
                    sigma = self.calculate_fixed_sigma(row['close'])
                    # 计算当前Sigma
                    if row['std20'] > 0:
                        current_sigma = (row['close'] - row['ma20']) / row['std20']
                    else:
                        current_sigma = 0
                    
                    self.trades.append({
                        'datetime': str(row['datetime']),
                        'action': '买入',
                        'signal': signal_name,
                        'price': float(row['close']),
                        'position': float(self.position),
                        'fixed_sigma': float(sigma),
                        'current_sigma': float(current_sigma)
                    })
                    
                    if verbose:
                        print(f"{row['datetime']} 买入[{signal_name}] 价格:{row['close']:.4f} 仓位:{self.position*100:.0f}% 固定σ:{sigma:.2f}")
            
            # 卖出
            sell_signals, fixed_sigma, current_sigma = self.check_sell_signals(row)
            for signal_name, sell_pct in sell_signals:
                if self.position > 0:
                    actual_sell = min(sell_pct, self.position)
                    pnl = (row['close'] - self.entry_price) / self.entry_price
                    
                    self.trades.append({
                        'datetime': str(row['datetime']),
                        'action': '卖出',
                        'signal': signal_name,
                        'price': float(row['close']),
                        'position': float(self.position - actual_sell),
                        'pnl': float(pnl * 100),
                        'fixed_sigma': float(fixed_sigma),
                        'current_sigma': float(current_sigma)
                    })
                    
                    if verbose:
                        print(f"{row['datetime']} 卖出[{signal_name}] 价格:{row['close']:.4f} 盈亏:{pnl*100:+.1f}% 固定σ:{fixed_sigma:.2f} 当前σ:{current_sigma:.2f}")
                    
                    self.position -= actual_sell
                    if self.position <= 0.1:
                        self.position = 0
                        self.entry_price = 0
                        self.base_ma = None
                        self.base_std = None
                        self.base_set = False
                        self.sigma_sold.clear()
        
        return self.generate_report(verbose)
    
    def generate_report(self, verbose=True):
        """生成回测报告"""
        buy_trades = [t for t in self.trades if t['action'] == '买入']
        sell_trades = [t for t in self.trades if t['action'] == '卖出']
        
        if not sell_trades:
            if verbose:
                print("\n无完整交易")
            return {'trades': self.trades, 'summary': {}}
        
        final_pnl = sell_trades[-1]['pnl']
        max_pnl = max([t['pnl'] for t in sell_trades])
        wins = [t for t in sell_trades if t['pnl'] > 0]
        win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0
        
        # Sigma统计
        sigma_stats = {}
        for t in sell_trades:
            sig = t['signal']
            sigma_stats[sig] = sigma_stats.get(sig, 0) + 1
        
        summary = {
            'total_buys': len(buy_trades),
            'total_sells': len(sell_trades),
            'win_rate': round(win_rate, 2),
            'final_pnl': round(final_pnl, 2),
            'max_pnl': round(max_pnl, 2),
            'sigma_distribution': sigma_stats
        }
        
        if verbose:
            print("\n" + "="*80)
            print("回测报告")
            print("="*80)
            print(f"\n交易统计:")
            print(f"  买入次数: {len(buy_trades)}")
            print(f"  卖出次数: {len(sell_trades)}")
            print(f"  胜率: {win_rate:.1f}%")
            print(f"\n盈亏统计:")
            print(f"  最终盈亏: {final_pnl:+.2f}%")
            print(f"  最大浮盈: {max_pnl:+.2f}%")
            print(f"\nSigma卖出分布:")
            for sig, count in sorted(sigma_stats.items()):
                print(f"  {sig}: {count}次")
            print("="*80)
        
        return {'trades': self.trades, 'summary': summary}

def main():
    """主函数"""
    print("加载数据...")
    df = pd.read_pickle('data/zz500_30min.pkl')
    
    strategy = SigmaBurstStrategy(initial_capital=100000, base_period=40)
    result = strategy.backtest(df, verbose=True)
    
    # 保存结果
    os.makedirs('backtest_results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_file = f'backtest_results/SigmaBurst_Final_{timestamp}.json'
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {result_file}")

if __name__ == "__main__":
    main()
