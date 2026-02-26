#!/usr/bin/env python3
"""
期权爆发策略 - 复合买入 + Sigma激进止盈
基于中证500ETF期权回测
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

class OptionBurstStrategy:
    """
    期权爆发策略
    
    买入逻辑（复合建仓）：
    1. 回踩中轨 + 缩量企稳：建仓30%
    2. 突破前高：加仓40%
    3. 放量确认：加满30%
    
    卖出逻辑（Sigma激进止盈）：
    - 6σ：减仓20%
    - 8σ：减仓30%
    - 10σ：减仓30%
    - 12σ+：清仓或留10%
    
    止损：
    - 跌破中轨且3根K线不收回
    - 权利金亏损50%
    """
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # 持仓比例 0-1
        self.entry_price = 0
        self.trades = []
        self.current_sigma = 0
        self.sigma_levels_sold = set()  # 记录已触发的sigma级别
        
    def calculate_indicators(self, df):
        """计算技术指标"""
        # 布林带 (20, 2)
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['upper'] = df['ma20'] + 2 * df['std20']
        df['lower'] = df['ma20'] - 2 * df['std20']
        df['sigma'] = (df['close'] - df['ma20']) / df['std20']
        
        # 成交量
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        
        # 前期高点（20周期）
        df['prev_high'] = df['high'].rolling(window=20).max()
        
        # 中轨企稳信号（连续3根在中轨附近）
        df['near_ma20'] = abs(df['close'] - df['ma20']) / df['ma20'] < 0.05
        df['consolidate'] = df['near_ma20'].rolling(window=3).sum() >= 2
        
        # 缩量
        df['low_volume'] = df['volume'] < df['volume_ma20'] * 0.7
        
        return df
    
    def check_buy_signals(self, df, idx):
        """检查买入信号"""
        if idx < 20:
            return None
        
        row = df.iloc[idx]
        prev = df.iloc[idx-1]
        
        signals = []
        
        # 信号1：回踩中轨 + 缩量企稳
        if (row['close'] < row['ma20'] * 1.02 and  # 在中轨附近
            row['close'] > row['ma20'] * 0.98 and
            row['low_volume'] and  # 缩量
            row['consolidate']):   # 企稳
            signals.append(('中轨企稳', 0.30))
        
        # 信号2：突破前高
        if (row['close'] > row['prev_high'] and
            row['volume'] > row['volume_ma20'] * 1.3):  # 放量
            signals.append(('突破前高', 0.40))
        
        # 信号3：放量确认（已有仓位后）
        if (self.position > 0 and
            row['volume'] > row['volume_ma20'] * 1.5 and
            row['close'] > prev['close']):
            signals.append(('放量确认', 0.30))
        
        return signals if signals else None
    
    def check_sell_signals(self, row):
        """检查卖出信号（Sigma激进止盈）"""
        if self.position <= 0:
            return None
        
        sigma = row['sigma']
        current_price = row['close']
        
        # 计算盈亏
        pnl_pct = (current_price - self.entry_price) / self.entry_price
        
        sell_signals = []
        
        # Sigma激进止盈
        if sigma >= 12 and '12sigma' not in self.sigma_levels_sold:
            sell_signals.append(('12σ清仓', 0.90))  # 清仓90%
            self.sigma_levels_sold.add('12sigma')
        elif sigma >= 10 and '10sigma' not in self.sigma_levels_sold:
            sell_signals.append(('10σ减仓', 0.30))
            self.sigma_levels_sold.add('10sigma')
        elif sigma >= 8 and '8sigma' not in self.sigma_levels_sold:
            sell_signals.append(('8σ减仓', 0.30))
            self.sigma_levels_sold.add('8sigma')
        elif sigma >= 6 and '6sigma' not in self.sigma_levels_sold:
            sell_signals.append(('6σ减仓', 0.20))
            self.sigma_levels_sold.add('6sigma')
        
        # 止损1：跌破中轨且亏损
        if current_price < row['ma20'] and pnl_pct < -0.20:
            sell_signals.append(('跌破中轨止损', 1.0))
        
        # 止损2：权利金亏损50%
        if pnl_pct < -0.50:
            sell_signals.append(('亏损50%止损', 1.0))
        
        return sell_signals if sell_signals else None
    
    def backtest(self, df):
        """回测主函数"""
        df = self.calculate_indicators(df)
        
        print(f"开始回测，数据周期: {df['datetime'].iloc[0]} 至 {df['datetime'].iloc[-1]}")
        print(f"总数据量: {len(df)} 条")
        
        for i in range(20, len(df)):
            row = df.iloc[i]
            
            # 检查买入信号
            buy_signals = self.check_buy_signals(df, i)
            if buy_signals:
                for signal_name, target_position in buy_signals:
                    if self.position < target_position:
                        # 买入
                        buy_pct = target_position - self.position
                        self.position = target_position
                        if self.entry_price == 0:
                            self.entry_price = row['close']
                        
                        self.trades.append({
                            'datetime': row['datetime'],
                            'action': '买入',
                            'signal': signal_name,
                            'price': row['close'],
                            'position': self.position,
                            'sigma': row['sigma']
                        })
                        print(f"  {row['datetime']} 买入[{signal_name}] 价格:{row['close']:.4f} 仓位:{self.position*100:.0f}% σ:{row['sigma']:.2f}")
            
            # 检查卖出信号
            sell_signals = self.check_sell_signals(row)
            if sell_signals:
                for signal_name, sell_pct in sell_signals:
                    if self.position > 0:
                        # 卖出
                        actual_sell = min(sell_pct, self.position)
                        pnl = (row['close'] - self.entry_price) / self.entry_price
                        
                        self.trades.append({
                            'datetime': row['datetime'],
                            'action': '卖出',
                            'signal': signal_name,
                            'price': row['close'],
                            'position': self.position - actual_sell,
                            'pnl': pnl * 100,
                            'sigma': row['sigma']
                        })
                        
                        print(f"  {row['datetime']} 卖出[{signal_name}] 价格:{row['close']:.4f} 盈亏:{pnl*100:+.1f}% σ:{row['sigma']:.2f}")
                        
                        self.position -= actual_sell
                        if self.position <= 0.1:
                            self.position = 0
                            self.entry_price = 0
                            self.sigma_levels_sold.clear()
        
        return self.generate_report()
    
    def generate_report(self):
        """生成回测报告"""
        print("\n" + "="*80)
        print("回测报告")
        print("="*80)
        
        if not self.trades:
            print("无交易记录")
            return []
        
        # 统计交易
        buy_trades = [t for t in self.trades if t['action'] == '买入']
        sell_trades = [t for t in self.trades if t['action'] == '卖出']
        
        print(f"\n总交易次数: {len(buy_trades)}次买入, {len(sell_trades)}次卖出")
        
        # 计算胜率
        if sell_trades:
            wins = [t for t in sell_trades if t.get('pnl', 0) > 0]
            win_rate = len(wins) / len(sell_trades) * 100
            avg_pnl = np.mean([t.get('pnl', 0) for t in sell_trades])
            max_pnl = max([t.get('pnl', 0) for t in sell_trades])
            min_pnl = min([t.get('pnl', 0) for t in sell_trades])
            
            print(f"\n胜率: {win_rate:.1f}%")
            print(f"平均盈亏: {avg_pnl:+.1f}%")
            print(f"最大盈利: {max_pnl:+.1f}%")
            print(f"最大亏损: {min_pnl:+.1f}%")
        
        # 打印详细交易
        print(f"\n详细交易记录:")
        print("-"*80)
        for t in self.trades:
            if t['action'] == '买入':
                print(f"{t['datetime']} 买入[{t['signal']}] 价格:{t['price']:.4f} 仓位:{t['position']*100:.0f}% σ:{t['sigma']:.2f}")
            else:
                print(f"{t['datetime']} 卖出[{t['signal']}] 价格:{t['price']:.4f} 盈亏:{t.get('pnl', 0):+.1f}% σ:{t['sigma']:.2f}")
        
        # 转换时间戳为字符串
        trades_export = []
        for t in self.trades:
            t_copy = t.copy()
            if isinstance(t_copy['datetime'], pd.Timestamp):
                t_copy['datetime'] = str(t_copy['datetime'])
            trades_export.append(t_copy)
        
        return trades_export

def main():
    # 加载数据
    print("加载中证500ETF数据...")
    df = pd.read_pickle('data/zz500_30min.pkl')
    
    # 运行回测
    strategy = OptionBurstStrategy(initial_capital=100000)
    trades = strategy.backtest(df)
    
    # 保存结果
    os.makedirs('backtest_results', exist_ok=True)
    result_file = f'backtest_results/option_burst_strategy_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {result_file}")

if __name__ == "__main__":
    main()
