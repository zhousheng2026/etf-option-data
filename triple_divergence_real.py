"""
三重底背离策略 - 使用真实期权价格数据
"""
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.insert(0, '/root/.openclaw/workspace/skills/index-options-burst/scripts')
from channel import draw_channel


class TripleDivergenceStrategyReal:
    """
    三重底背离策略 - 使用真实期权价格
    """
    
    def __init__(self,
                 initial_capital: float = 1000000,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 divergence_lookback: int = 30,
                 min_divergence_bars: int = 5,
                 stop_loss_pct: float = 0.30,
                 cooldown_bars: int = 5):
        
        self.initial_capital = initial_capital
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.divergence_lookback = divergence_lookback
        self.min_divergence_bars = min_divergence_bars
        self.stop_loss_pct = stop_loss_pct
        self.cooldown_bars = cooldown_bars
        
        self.capital = initial_capital
        self.position = None
        self.entry_price = 0
        self.cooldown_counter = 0
        self.trades = []
        self.equity_curve = [initial_capital]
        
    def calculate_bollinger_bands(self, df):
        """计算布林带"""
        df = df.copy()
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        bb_std = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + self.bb_std * bb_std
        df['bb_lower'] = df['bb_middle'] - self.bb_std * bb_std
        return df
    
    def calculate_macd(self, df):
        """计算MACD"""
        df = df.copy()
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        return df
    
    def find_local_lows(self, prices, lookback=3):
        """寻找局部低点"""
        lows = []
        for i in range(lookback, len(prices) - lookback):
            window = prices[i-lookback:i+lookback+1]
            if prices[i] == np.min(window):
                lows.append(i)
        return lows
    
    def detect_bollinger_divergence(self, df, idx):
        """检测布林带底背离"""
        if idx < self.divergence_lookback + self.bb_period:
            return False
        
        prices = df['close'].iloc[idx-self.divergence_lookback:idx+1].values
        bb_lower = df['bb_lower'].iloc[idx-self.divergence_lookback:idx+1].values
        
        local_lows = self.find_local_lows(prices, lookback=3)
        
        if len(local_lows) < 2:
            return False
        
        recent_low_idx = local_lows[-1]
        prev_low_idx = local_lows[-2]
        
        if recent_low_idx - prev_low_idx < self.min_divergence_bars:
            return False
        
        price_change = (prices[recent_low_idx] - prices[prev_low_idx]) / prices[prev_low_idx]
        bb_change = (bb_lower[recent_low_idx] - bb_lower[prev_low_idx]) / bb_lower[prev_low_idx]
        
        if price_change < -0.002 and bb_change > price_change * 0.5:
            return True
        
        return False
    
    def detect_macd_divergence(self, df, idx):
        """检测MACD底背离"""
        if idx < self.divergence_lookback + self.macd_slow:
            return False
        
        prices = df['close'].iloc[idx-self.divergence_lookback:idx+1].values
        macd = df['macd'].iloc[idx-self.divergence_lookback:idx+1].values
        
        local_lows = self.find_local_lows(prices, lookback=3)
        
        if len(local_lows) < 2:
            return False
        
        recent_low_idx = local_lows[-1]
        prev_low_idx = local_lows[-2]
        
        if recent_low_idx - prev_low_idx < self.min_divergence_bars:
            return False
        
        price_change = (prices[recent_low_idx] - prices[prev_low_idx]) / prices[prev_low_idx]
        macd_change = macd[recent_low_idx] - macd[prev_low_idx]
        
        if price_change < -0.002 and macd_change > 0:
            return True
        
        return False
    
    def detect_price_divergence(self, df, idx):
        """检测价格底背离（成交量）"""
        if idx < self.divergence_lookback + 5:
            return False
        
        prices = df['close'].iloc[idx-self.divergence_lookback:idx+1].values
        volumes = df['volume'].iloc[idx-self.divergence_lookback:idx+1].values
        
        local_lows = self.find_local_lows(prices, lookback=3)
        
        if len(local_lows) < 2:
            return False
        
        recent_low_idx = local_lows[-1]
        prev_low_idx = local_lows[-2]
        
        if recent_low_idx - prev_low_idx < self.min_divergence_bars:
            return False
        
        price_change = (prices[recent_low_idx] - prices[prev_low_idx]) / prices[prev_low_idx]
        volume_change = (volumes[recent_low_idx] - volumes[prev_low_idx]) / volumes[prev_low_idx]
        
        if price_change < -0.002 and volume_change < -0.05:
            return True
        
        return False
    
    def check_entry_signal(self, df, idx):
        """检查买入信号"""
        if self.position is not None:
            return False, 0, []
        
        if self.cooldown_counter > 0:
            return False, 0, []
        
        bb_div = self.detect_bollinger_divergence(df, idx)
        macd_div = self.detect_macd_divergence(df, idx)
        price_div = self.detect_price_divergence(df, idx)
        
        divergences = []
        if bb_div:
            divergences.append('Bollinger')
        if macd_div:
            divergences.append('MACD')
        if price_div:
            divergences.append('Price')
        
        if len(divergences) >= 2:
            strength = 2 if len(divergences) == 3 else 1
            return True, strength, divergences
        
        return False, 0, []
    
    def check_exit_signal(self, df, idx):
        """检查卖出信号"""
        if self.position is None:
            return False, ""
        
        current_price = df['close'].iloc[idx]
        
        # 止损检查
        pnl_pct = (current_price - self.entry_price) / self.entry_price
        
        if pnl_pct <= -self.stop_loss_pct:
            return True, f"Stop Loss: {pnl_pct:.2%}"
        
        # 通道跌破检查
        if idx >= 20:
            highs = df['high'].iloc[idx-20:idx].values
            lows = df['low'].iloc[idx-20:idx].values
            middle = (np.max(highs) + np.min(lows)) / 2
            
            if current_price < middle:
                return True, f"Channel Breakdown: {current_price:.4f} < {middle:.4f}"
        
        return False, ""
    
    def run_backtest(self, df):
        """运行回测"""
        print(f"开始回测: {len(df)} 根K线")
        
        # 预处理数据
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_macd(df)
        
        start_idx = max(self.bb_period, self.macd_slow, self.divergence_lookback) + 10
        
        for i in range(start_idx, len(df)):
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            
            # 检查离场
            if self.position is not None:
                should_exit, exit_reason = self.check_exit_signal(df, i)
                
                if should_exit:
                    self.close_position(df, i, exit_reason)
                    self.cooldown_counter = self.cooldown_bars
            
            # 检查入场
            if self.position is None:
                should_buy, strength, divergences = self.check_entry_signal(df, i)
                
                if should_buy:
                    self.open_position(df, i, strength, divergences)
            
            # 记录权益
            self.equity_curve.append(self.capital)
        
        return self.calculate_result()
    
    def open_position(self, df, idx, strength, divergences):
        """开仓 - 使用真实期权价格"""
        row = df.iloc[idx]
        
        # 直接使用真实期权收盘价作为入场价格
        option_price = row['close']
        
        # 计算仓位（根据信号强度）
        weight = 0.50 if strength == 2 else 0.30
        position_value = self.capital * weight
        
        # 假设合约乘数为10000，计算合约数量
        contracts = int(position_value / (option_price * 10000))
        contracts = max(contracts, 1)
        
        self.position = {
            'entry_time': df.index[idx],
            'entry_price': option_price,
            'quantity': contracts,
            'divergences': divergences,
            'signal_strength': strength
        }
        
        self.entry_price = option_price
        
        div_str = '+'.join(divergences)
        print(f"[{df.index[idx]}] 开仓 | 背离: {div_str} | 强度: {strength} | "
              f"期权价格: {option_price:.4f} | 数量: {contracts}")
    
    def close_position(self, df, idx, reason):
        """平仓 - 使用真实期权价格计算盈亏"""
        row = df.iloc[idx]
        
        # 使用真实期权收盘价作为出场价格
        exit_price = row['close']
        entry_price = self.position['entry_price']
        contracts = self.position['quantity']
        
        # 计算盈亏（真实期权价格差）
        pnl_per_contract = (exit_price - entry_price) * 10000  # 合约乘数
        total_pnl = pnl_per_contract * contracts
        
        self.capital += total_pnl
        
        pnl_pct = (exit_price - entry_price) / entry_price
        
        trade = {
            'entry_time': self.position['entry_time'],
            'exit_time': df.index[idx],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': contracts,
            'pnl': total_pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': reason,
            'divergences': self.position['divergences'],
            'signal_strength': self.position['signal_strength']
        }
        
        self.trades.append(trade)
        
        div_str = '+'.join(self.position['divergences'])
        print(f"[{df.index[idx]}] 平仓 | 背离: {div_str} | 原因: {reason} | "
              f"盈亏: {total_pnl:,.0f} ({pnl_pct:.2%})")
        
        self.position = None
        self.entry_price = 0
    
    def calculate_result(self):
        """计算回测结果"""
        result = {
            'trades': self.trades,
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital,
        }
        
        if len(self.trades) == 0:
            result.update({
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_drawdown': 0.0,
            })
            return result
        
        # 胜率
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        result['win_rate'] = len(wins) / len(self.trades)
        
        # 盈亏比
        total_wins = sum(t['pnl'] for t in wins)
        total_losses = abs(sum(t['pnl'] for t in losses))
        result['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # 平均盈亏
        result['avg_win'] = np.mean([t['pnl'] for t in wins]) if wins else 0
        result['avg_loss'] = np.mean([t['pnl'] for t in losses]) if losses else 0
        
        # 最大回撤
        peak = self.initial_capital
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        result['max_drawdown'] = max_dd
        
        return result
    
    def print_report(self, result):
        """打印回测报告"""
        print("\n" + "="*70)
        print("三重底背离策略回测报告（真实期权价格）")
        print("="*70)
        print(f"初始资金: {result['initial_capital']:,.0f}")
        print(f"最终资金: {result['final_capital']:,.0f}")
        print(f"总收益率: {result['total_return']:.2%}")
        print(f"最大回撤: {result['max_drawdown']:.2%}")
        print("-"*70)
        print(f"总交易次数: {len(result['trades'])}")
        print(f"总胜率: {result['win_rate']:.2%}")
        print(f"总盈亏比: {result['profit_factor']:.2f}")
        print(f"平均盈利: {result['avg_win']:,.0f}")
        print(f"平均亏损: {result['avg_loss']:,.0f}")
        print("="*70)
        
        if result['trades']:
            print("\n交易明细:")
            for i, t in enumerate(result['trades'], 1):
                div_str = '+'.join(t['divergences'])
                print(f"{i}. [{t['entry_time']}] 入场 | [{t['exit_time']}] 出场")
                print(f"   背离: {div_str} | 盈亏: {t['pnl']:,.0f} ({t['pnl_pct']:.2%})")
                print(f"   原因: {t['exit_reason']}")
