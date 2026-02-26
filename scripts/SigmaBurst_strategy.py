#!/usr/bin/env python3
"""
期权爆发策略 - 最终版本
策略名称：SigmaBurst（西格玛爆发）

核心逻辑：
1. 买入：复合建仓（中轨企稳 + 突破加仓 + 放量确认）
2. 卖出：Sigma激进止盈（6/8/10/12σ分批减仓）
3. 止损：跌破中轨或亏损50%

适用标的：ETF期权（50ETF、300ETF、500ETF、创业板ETF、科创50ETF）
周期：30分钟K线
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

class SigmaBurstStrategy:
    """
    SigmaBurst期权爆发策略
    
    参数说明：
    - bb_period: 布林带周期，默认20
    - bb_std: 布林带标准差，默认2.0
    - volume_threshold_low: 缩量阈值，默认0.7（70%均量）
    - volume_threshold_high: 放量阈值，默认1.3（130%均量）
    - near_ma_threshold: 中轨附近阈值，默认0.05（5%）
    - max_position: 最大仓位，默认1.0（100%）
    """
    
    def __init__(self, 
                 initial_capital=100000,
                 bb_period=20,
                 bb_std=2.0,
                 volume_threshold_low=0.7,
                 volume_threshold_high=1.3,
                 near_ma_threshold=0.05,
                 max_position=1.0):
        
        self.initial_capital = initial_capital
        self.bb_period = bb_period
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
        
    def calculate_indicators(self, df):
        """计算技术指标"""
        # 布林带
        df['ma20'] = df['close'].rolling(window=self.bb_period).mean()
        df['std20'] = df['close'].rolling(window=self.bb_period).std()
        df['upper'] = df['ma20'] + self.bb_std * df['std20']
        df['lower'] = df['ma20'] - self.bb_std * df['std20']
        df['sigma'] = (df['close'] - df['ma20']) / df['std20']
        
        # 成交量
        df['volume_ma20'] = df['volume'].rolling(window=self.bb_period).mean()
        
        # 前期高点
        df['prev_high'] = df['high'].rolling(window=self.bb_period).max()
        df['prev_high_shift'] = df['prev_high'].shift(1)
        
        # 中轨附近信号
        df['near_ma20'] = abs(df['close'] - df['ma20']) / df['ma20'] < self.near_ma_threshold
        
        # 缩量/放量
        df['low_volume'] = df['volume'] < df['volume_ma20'] * self.volume_threshold_low
        df['high_volume'] = df['volume'] > df['volume_ma20'] * self.volume_threshold_high
        
        # 企稳信号（连续2根在中轨附近）
        df['consolidate'] = df['near_ma20'].rolling(window=2).sum() >= 2
        
        return df
    
    def check_buy_signals(self, df, idx):
        """
        检查买入信号
        
        信号1：中轨企稳（建仓30%）
        - 价格在中轨附近（±5%）
        - 缩量（<70%均量）
        - 企稳（连续2根在中轨附近）
        - Sigma在-1到1之间
        
        信号2：突破前高（加仓40%）
        - 价格突破前20周期高点
        - 放量（>130%均量）
        
        信号3：放量确认（加满30%）
        - 已有仓位
        - 放量（>150%均量）
        - 价格上涨
        """
        if idx < self.bb_period:
            return []
        
        row = df.iloc[idx]
        prev = df.iloc[idx-1]
        signals = []
        
        # 信号1：中轨企稳
        if (row['near_ma20'] and 
            row['low_volume'] and 
            row['consolidate'] and
            -1 < row['sigma'] < 1 and
            self.position < 0.3):
            signals.append(('中轨企稳', 0.30))
        
        # 信号2：突破前高
        if (row['close'] > row['prev_high_shift'] and
            row['high_volume'] and
            self.position < 0.7):
            signals.append(('突破前高', 0.70))
        
        # 信号3：放量确认
        if (self.position > 0 and
            row['volume'] > row['volume_ma20'] * 1.5 and
            row['close'] > prev['close'] and
            self.position < self.max_position):
            signals.append(('放量确认', self.max_position))
        
        return signals
    
    def check_sell_signals(self, row, entry_ma20, entry_std20):
        """
        检查卖出信号（Sigma激进止盈）
        
        使用入场时的布林带参数计算Sigma，避免均线跟随导致Sigma被压缩
        
        止盈：
        - 6σ：减仓20%
        - 8σ：减仓30%
        - 10σ：减仓30%
        - 12σ+：清仓90%（留10%底仓博更大收益）
        
        止损：
        - 跌破中轨且亏损>20%
        - 权利金亏损>50%
        """
        if self.position <= 0 or self.entry_price <= 0:
            return []
        
        signals = []
        current_price = row['close']
        
        # 用入场时的布林带参数计算Sigma（固定基准）
        if entry_std20 > 0:
            sigma = (current_price - entry_ma20) / entry_std20
        else:
            sigma = 0
        
        pnl = (current_price - self.entry_price) / self.entry_price
        
        # Sigma激进止盈
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
        
        # 止损1：跌破中轨且大幅亏损
        if current_price < row['ma20'] and pnl < -0.20:
            signals.append(('跌破中轨止损', 1.0))
        
        # 止损2：权利金腰斩
        if pnl < -0.50:
            signals.append(('亏损50%止损', 1.0))
        
        return signals
    
    def backtest(self, df, verbose=True):
        """回测主函数"""
        df = self.calculate_indicators(df)
        
        if verbose:
            print("="*80)
            print("SigmaBurst期权爆发策略 - 回测")
            print("="*80)
            print(f"数据周期: {df['datetime'].iloc[0]} 至 {df['datetime'].iloc[-1]}")
            print(f"总数据量: {len(df)} 条")
            print(f"初始资金: {self.initial_capital:,.0f}")
            print("-"*80)
        
        # 记录入场时的布林带参数（用于固定Sigma计算基准）
        entry_ma20 = 0
        entry_std20 = 0
        
        for i in range(self.bb_period, len(df)):
            row = df.iloc[i]
            
            # 买入
            buy_signals = self.check_buy_signals(df, i)
            for signal_name, target_pos in buy_signals:
                if self.position < target_pos:
                    buy_pct = target_pos - self.position
                    self.position = target_pos
                    if self.entry_price == 0:
                        self.entry_price = row['close']
                        # 记录入场时的布林带参数
                        entry_ma20 = row['ma20']
                        entry_std20 = row['std20']
                    
                    self.trades.append({
                        'datetime': str(row['datetime']),
                        'action': '买入',
                        'signal': signal_name,
                        'price': float(row['close']),
                        'position': float(self.position),
                        'sigma': float(row['sigma'])
                    })
                    
                    if verbose:
                        print(f"{row['datetime']} 买入[{signal_name}] 价格:{row['close']:.4f} 仓位:{self.position*100:.0f}% σ:{row['sigma']:.2f}")
            
            # 卖出（使用入场时的布林带参数计算Sigma）
            sell_signals = self.check_sell_signals(row, entry_ma20, entry_std20)
            for signal_name, sell_pct in sell_signals:
                if self.position > 0:
                    actual_sell = min(sell_pct, self.position)
                    pnl = (row['close'] - self.entry_price) / self.entry_price
                    # 计算当前Sigma（用于显示）
                    current_sigma = (row['close'] - entry_ma20) / entry_std20 if entry_std20 > 0 else 0
                    
                    self.trades.append({
                        'datetime': str(row['datetime']),
                        'action': '卖出',
                        'signal': signal_name,
                        'price': float(row['close']),
                        'position': float(self.position - actual_sell),
                        'pnl': float(pnl * 100),
                        'sigma': float(current_sigma)
                    })
                    
                    if verbose:
                        print(f"{row['datetime']} 卖出[{signal_name}] 价格:{row['close']:.4f} 盈亏:{pnl*100:+.1f}% σ:{current_sigma:.2f}")
                    
                    self.position -= actual_sell
                    if self.position <= 0.1:
                        self.position = 0
                        self.entry_price = 0
                        entry_ma20 = 0
                        entry_std20 = 0
                        self.sigma_sold.clear()
        
        return self.generate_report(verbose)
    
    def generate_report(self, verbose=True):
        """生成回测报告"""
        buy_trades = [t for t in self.trades if t['action'] == '买入']
        sell_trades = [t for t in self.trades if t['action'] == '卖出']
        
        if not sell_trades:
            if verbose:
                print("\n无完整交易（只有买入无卖出）")
            return {'trades': self.trades, 'summary': {}}
        
        # 计算指标
        final_pnl = sell_trades[-1]['pnl'] if sell_trades else 0
        max_pnl = max([t['pnl'] for t in sell_trades])
        min_pnl = min([t['pnl'] for t in sell_trades])
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
            'min_pnl': round(min_pnl, 2),
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
            print(f"  最大亏损: {min_pnl:+.2f}%")
            print(f"\nSigma卖出分布:")
            for sig, count in sorted(sigma_stats.items()):
                print(f"  {sig}: {count}次")
            print("="*80)
        
        return {'trades': self.trades, 'summary': summary}

def main():
    """主函数 - 回测中证500ETF数据"""
    print("加载中证500ETF数据...")
    df = pd.read_pickle('data/zz500_30min.pkl')
    
    # 运行回测
    strategy = SigmaBurstStrategy(initial_capital=100000)
    result = strategy.backtest(df, verbose=True)
    
    # 保存结果
    os.makedirs('backtest_results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_file = f'backtest_results/SigmaBurst_{timestamp}.json'
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {result_file}")
    
    return result

if __name__ == "__main__":
    main()
