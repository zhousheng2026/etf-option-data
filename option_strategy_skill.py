#!/usr/bin/env python3
"""
股指期权布林带均值回归策略 - 优化版
最优参数：15分钟周期 + MACD(8,26,9) + BOLL(15,1.5) + 底背离 + 固定止盈3%
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import json
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StrategyConfig:
    """策略配置"""
    # 周期
    timeframe: str = "15min"
    
    # MACD参数
    macd_fast: int = 8
    macd_slow: int = 26
    macd_signal: int = 9
    
    # 布林带参数
    boll_period: int = 15
    boll_std: float = 1.5
    
    # 背离检测
    divergence_lookback: int = 15
    
    # 入场范围 (0-1, 0.5表示下轨上方50%范围内)
    entry_range: float = 0.50
    
    # 平仓条件类型 - 明天仿真盘测试优化
    exit_type: str = "macd_reverse"  # 先用MACD反向交叉，明天根据实盘调整
    exit_param: float = 0.0  # 备用参数
    
    # 资金配置
    capital: int = 10000
    commission: float = 4.0  # 每手手续费
    option_delta: float = 0.4  # 期权Delta
    
    def to_dict(self):
        return asdict(self)


class OptionStrategyOptimized:
    """优化版股指期权策略"""
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
    
    def prepare_data(self, df_1min: pd.DataFrame) -> pd.DataFrame:
        """将1分钟数据合成目标周期"""
        df = df_1min.resample(self.config.timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        
        # MACD
        df['ema_fast'] = df['close'].ewm(span=self.config.macd_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.config.macd_slow, adjust=False).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=self.config.macd_signal, adjust=False).mean()
        df['macd_golden'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_dead'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # BOLL
        df['boll_mid'] = df['close'].rolling(window=self.config.boll_period).mean()
        df['boll_std'] = df['close'].rolling(window=self.config.boll_period).std()
        df['boll_upper'] = df['boll_mid'] + self.config.boll_std * df['boll_std']
        df['boll_lower'] = df['boll_mid'] - self.config.boll_std * df['boll_std']
        df['boll_range'] = df['boll_upper'] - df['boll_lower']
        df['price_to_lower'] = (df['close'] - df['boll_lower']) / df['boll_range']
        
        # 入场范围
        df['near_lower'] = df['price_to_lower'] <= self.config.entry_range
        df['near_upper'] = df['price_to_lower'] >= (1 - self.config.entry_range)
        
        return df
    
    def detect_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测背离"""
        df = df.copy()
        df['divergence_bull'] = False
        df['divergence_bear'] = False
        
        lb = self.config.divergence_lookback
        for i in range(lb, len(df)):
            window = df.iloc[i-lb:i+1]
            
            # 底背离：价格新低，MACD抬高
            price_low = window['low'].min()
            macd_low = window['macd'].min()
            if (df.iloc[i]['low'] <= price_low * 1.001 and 
                df.iloc[i]['macd'] > macd_low * 0.95 and
                df.iloc[i]['macd'] < 0):
                df.loc[df.index[i], 'divergence_bull'] = True
            
            # 顶背离
            price_high = window['high'].max()
            macd_high = window['macd'].max()
            if (df.iloc[i]['high'] >= price_high * 0.999 and 
                df.iloc[i]['macd'] < macd_high * 1.05 and
                df.iloc[i]['macd'] > 0):
                df.loc[df.index[i], 'divergence_bear'] = True
        
        return df
    
    def backtest(self, df_1min: pd.DataFrame) -> Dict:
        """执行回测"""
        # 数据准备
        df = self.prepare_data(df_1min)
        df = self.calculate_indicators(df)
        df = self.detect_divergence(df)
        df = df.dropna()
        
        if len(df) < 50:
            return {'error': '数据不足'}
        
        # 回测
        position = 0
        equity = self.config.capital
        trades = []
        entry_price = 0
        entry_time = None
        max_price = 0
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # 入场信号（只做多头示例）
            long_signal = row['macd_golden'] and row['near_lower'] and row['divergence_bull']
            short_signal = row['macd_dead'] and row['near_upper'] and row['divergence_bear']
            
            # 平仓信号
            close_signal = False
            
            if position == 1:
                max_price = max(max_price, row['close'])
                
                if self.config.exit_type == 'macd_reverse':
                    close_signal = row['macd_dead']
                elif self.config.exit_type == 'middle_cross':
                    close_signal = (row['close'] > row['boll_mid'] and 
                                   prev_row['close'] <= prev_row['boll_mid'])
                elif self.config.exit_type == 'boll_upper':
                    close_signal = row['high'] >= row['boll_upper']
                elif self.config.exit_type == 'trailing_stop':
                    drawdown = (max_price - row['close']) / max_price
                    close_signal = drawdown >= self.config.exit_param
                elif self.config.exit_type == 'fixed_profit':
                    profit_pct = (row['close'] - entry_price) / entry_price
                    close_signal = profit_pct >= self.config.exit_param
            
            elif position == -1:
                if self.config.exit_type == 'macd_reverse':
                    close_signal = row['macd_golden']
                elif self.config.exit_type == 'middle_cross':
                    close_signal = (row['close'] < row['boll_mid'] and 
                                   prev_row['close'] >= prev_row['boll_mid'])
                elif self.config.exit_type == 'boll_upper':
                    close_signal = row['low'] <= row['boll_lower']
                elif self.config.exit_type == 'trailing_stop':
                    drawup = (row['close'] - entry_price) / entry_price
                    close_signal = drawup >= self.config.exit_param
                elif self.config.exit_type == 'fixed_profit':
                    profit_pct = (entry_price - row['close']) / entry_price
                    close_signal = profit_pct >= self.config.exit_param
            
            # 执行交易
            if position == 0:
                if long_signal:
                    position = 1
                    entry_price = row['close']
                    entry_time = row.name
                    max_price = entry_price
                elif short_signal:
                    position = -1
                    entry_price = row['close']
                    entry_time = row.name
            
            elif position != 0 and close_signal:
                exit_price = row['close']
                
                if position == 1:
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price
                
                pnl = self.config.capital * pnl_pct * self.config.option_delta - self.config.commission
                equity += pnl
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row.name,
                    'direction': 'LONG' if position == 1 else 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct * 100
                })
                position = 0
        
        # 计算指标
        if not trades:
            return {'trades': 0, 'win_rate': 0, 'total_return': 0}
        
        trades_df = pd.DataFrame(trades)
        total = len(trades)
        wins = len(trades_df[trades_df['pnl'] > 0])
        win_rate = wins / total * 100
        total_return = (equity - self.config.capital) / self.config.capital * 100
        
        return {
            'trades': total,
            'wins': wins,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_capital': equity,
            'avg_pnl': trades_df['pnl'].mean(),
            'max_pnl': trades_df['pnl'].max(),
            'min_pnl': trades_df['pnl'].min(),
            'trades_detail': trades
        }
    
    def run(self, data_path: str) -> Dict:
        """运行策略"""
        # 读取数据
        data_lines = open(data_path, 'r', encoding='gbk').readlines()
        records = []
        for line in data_lines[2:]:
            line = line.strip()
            if not line or '时间' in line:
                continue
            parts = line.split('\t')
            if len(parts) >= 6:
                try:
                    dt = pd.to_datetime(parts[0].strip(), format='%Y/%m/%d-%H:%M')
                    records.append({
                        'datetime': dt,
                        'open': float(parts[1]),
                        'high': float(parts[2]),
                        'low': float(parts[3]),
                        'close': float(parts[4]),
                        'volume': float(parts[5]),
                    })
                except:
                    continue
        
        df = pd.DataFrame(records)
        df = df.sort_values('datetime').reset_index(drop=True)
        df.set_index('datetime', inplace=True)
        
        return self.backtest(df)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("="*70)
    print("股指期权布林带均值回归策略 - 优化版")
    print("="*70)
    
    # 最优配置
    config = StrategyConfig(
        timeframe="15min",
        macd_fast=8,
        macd_slow=26,
        macd_signal=9,
        boll_period=15,
        boll_std=1.5,
        divergence_lookback=15,
        entry_range=0.50,
        exit_type="fixed_profit",
        exit_param=0.03,  # 3%止盈
        capital=10000,
        commission=4.0,
        option_delta=0.4
    )
    
    strategy = OptionStrategyOptimized(config)
    
    # 运行回测
    result = strategy.run('/root/openclaw/kimi/downloads/19c8f5bd-03a2-8f41-8000-0000977f629a_159922.txt')
    
    print(f"\n【回测结果】")
    print(f"交易次数: {result['trades']}")
    print(f"盈利次数: {result['wins']}")
    print(f"胜率: {result['win_rate']:.2f}%")
    print(f"总收益率: {result['total_return']:.2f}%")
    print(f"最终资金: {result['final_capital']:.2f}元")
    print(f"平均盈亏: {result['avg_pnl']:.2f}元")
    print(f"最大盈利: {result['max_pnl']:.2f}元")
    print(f"最大亏损: {result['min_pnl']:.2f}元")
    
    # 保存配置
    with open('/root/.openclaw/workspace/strategy_config.json', 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"\n✓ 策略配置已保存至: strategy_config.json")
