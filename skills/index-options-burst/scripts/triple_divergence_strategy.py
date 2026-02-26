"""
三重底背离策略 - 布林带底背离 + MACD底背离 + 价格底背离
买入信号：至少2种底背离同时出现
卖出信号：30分钟通道跌破（中轨以下）
只做多，不做空
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import sys
import os

# 添加父目录到路径以导入channel模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.channel import draw_channel, Channel


class Direction(Enum):
    LONG = 1   # 做多
    NONE = 0   # 空仓


@dataclass
class DivergenceSignal:
    """背离信号"""
    type: str  # 'bollinger', 'macd', 'price'
    detected: bool
    strength: float  # 背离强度 0-1
    description: str = ""


@dataclass
class Trade:
    """单笔交易记录"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    option_entry: float = 0.0
    option_exit: float = 0.0
    quantity: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    # 买入时的背离信号记录
    divergences: List[str] = field(default_factory=list)
    signal_strength: int = 0  # 1=2种背离, 2=3种背离


@dataclass
class BacktestResult:
    """回测结果"""
    trades: List[Trade] = field(default_factory=list)
    initial_capital: float = 1000000
    final_capital: float = 1000000
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    # 按信号强度统计
    strength_2_trades: List[Trade] = field(default_factory=list)  # 2种背离
    strength_3_trades: List[Trade] = field(default_factory=list)  # 3种背离
    strength_2_win_rate: float = 0.0
    strength_3_win_rate: float = 0.0


class TripleDivergenceStrategy:
    """
    三重底背离策略
    
    买入条件（至少满足2个）：
    1. 布林带底背离：价格创新低，布林带下轨未创新低
    2. MACD底背离：价格创新低，MACD未创新低
    3. 价格底背离：当前价格相对前低，出现企稳迹象
    
    卖出条件：
    - 价格跌破30分钟通道中轨
    - 或达到止损位（期权价格跌30%/50%）
    """
    
    def __init__(self,
                 initial_capital: float = 1000000,
                 channel_bars: int = 20,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 divergence_lookback: int = 30,
                 min_divergence_bars: int = 5,
                 stop_loss_1: float = 0.30,
                 stop_loss_2: float = 0.50,
                 cooldown_bars: int = 5,
                 option_delta: float = 0.30,
                 contract_multiplier: int = 10000):
        
        self.initial_capital = initial_capital
        self.channel_bars = channel_bars
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.divergence_lookback = divergence_lookback
        self.min_divergence_bars = min_divergence_bars
        self.stop_loss_1 = stop_loss_1
        self.stop_loss_2 = stop_loss_2
        self.cooldown_bars = cooldown_bars
        self.option_delta = option_delta
        self.contract_multiplier = contract_multiplier
        
        # 状态
        self.capital = initial_capital
        self.position: Direction = Direction.NONE
        self.current_trade: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.cooldown_counter = 0
        self.equity_curve: List[float] = [initial_capital]
        
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带"""
        df = df.copy()
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        bb_std = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + self.bb_std * bb_std
        df['bb_lower'] = df['bb_middle'] - self.bb_std * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        return df
    
    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标"""
        df = df.copy()
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df
    
    def find_local_lows(self, prices: np.ndarray, lookback: int = 3) -> List[int]:
        """
        寻找局部低点 - 简化版本
        返回局部低点的索引列表
        """
        lows = []
        for i in range(lookback, len(prices) - lookback):
            # 当前点是前后lookback根K线的最低点
            window = prices[i-lookback:i+lookback+1]
            if prices[i] == np.min(window):
                lows.append(i)
        return lows
    
    def detect_bollinger_divergence(self, df: pd.DataFrame, idx: int) -> DivergenceSignal:
        """
        检测布林带底背离
        条件：价格创新低，但布林带下轨未创新低（或跌幅小于价格）
        """
        if idx < self.divergence_lookback + self.bb_period:
            return DivergenceSignal('bollinger', False, 0.0, "数据不足")
        
        # 寻找近期局部低点 - 使用更大的回溯窗口
        prices = df['close'].iloc[idx-self.divergence_lookback:idx+1].values
        bb_lower = df['bb_lower'].iloc[idx-self.divergence_lookback:idx+1].values
        
        local_lows = self.find_local_lows(prices, lookback=3)
        
        if len(local_lows) < 2:
            return DivergenceSignal('bollinger', False, 0.0, f"局部低点不足({len(local_lows)})")
        
        # 取最近两个局部低点
        recent_low_idx = local_lows[-1]
        prev_low_idx = local_lows[-2]
        
        # 确保两个低点之间有一定距离
        if recent_low_idx - prev_low_idx < self.min_divergence_bars:
            return DivergenceSignal('bollinger', False, 0.0, "低点间距不足")
        
        # 价格变化
        price_change = (prices[recent_low_idx] - prices[prev_low_idx]) / prices[prev_low_idx]
        bb_change = (bb_lower[recent_low_idx] - bb_lower[prev_low_idx]) / bb_lower[prev_low_idx]
        
        # 底背离：价格创新低（跌幅更大），但布林带下轨未创新低（跌幅更小或上涨）
        if price_change < -0.002 and bb_change > price_change * 0.5:  # 价格跌，布林带相对强
            strength = min(abs(price_change - bb_change) * 50, 1.0)
            return DivergenceSignal('bollinger', True, strength, 
                f"价格跌{price_change:.2%}, 布林带跌{bb_change:.2%}")
        
        return DivergenceSignal('bollinger', False, 0.0, f"价格{price_change:.4f},BB{bb_change:.4f}")
    
    def detect_macd_divergence(self, df: pd.DataFrame, idx: int) -> DivergenceSignal:
        """
        检测MACD底背离
        条件：价格创新低，但MACD未创新低
        """
        if idx < self.divergence_lookback + self.macd_slow:
            return DivergenceSignal('macd', False, 0.0, "数据不足")
        
        prices = df['close'].iloc[idx-self.divergence_lookback:idx+1].values
        macd = df['macd'].iloc[idx-self.divergence_lookback:idx+1].values
        
        local_lows = self.find_local_lows(prices, lookback=3)
        
        if len(local_lows) < 2:
            return DivergenceSignal('macd', False, 0.0, "局部低点不足")
        
        recent_low_idx = local_lows[-1]
        prev_low_idx = local_lows[-2]
        
        if recent_low_idx - prev_low_idx < self.min_divergence_bars:
            return DivergenceSignal('macd', False, 0.0, "低点间距不足")
        
        # 价格创新低
        price_change = (prices[recent_low_idx] - prices[prev_low_idx]) / prices[prev_low_idx]
        # MACD未创新低（或跌幅更小）
        macd_change = macd[recent_low_idx] - macd[prev_low_idx]
        
        if price_change < -0.002 and macd_change > 0:  # 价格跌，MACD上升
            strength = min(abs(macd_change / (abs(macd[prev_low_idx]) + 0.001)) * 0.5, 1.0)
            return DivergenceSignal('macd', True, strength,
                f"价格跌{price_change:.2%}, MACD上升{macd_change:.4f}")
        
        return DivergenceSignal('macd', False, 0.0, "无背离")
    
    def detect_price_divergence(self, df: pd.DataFrame, idx: int) -> DivergenceSignal:
        """
        检测价格底背离（价格行为）
        条件：价格创新低，但下跌动能减弱（如K线形态、成交量等）
        """
        if idx < self.divergence_lookback + 5:
            return DivergenceSignal('price', False, 0.0, "数据不足")
        
        prices = df['close'].iloc[idx-self.divergence_lookback:idx+1].values
        volumes = df['volume'].iloc[idx-self.divergence_lookback:idx+1].values
        
        local_lows = self.find_local_lows(prices, lookback=3)
        
        if len(local_lows) < 2:
            return DivergenceSignal('price', False, 0.0, "局部低点不足")
        
        recent_low_idx = local_lows[-1]
        prev_low_idx = local_lows[-2]
        
        if recent_low_idx - prev_low_idx < self.min_divergence_bars:
            return DivergenceSignal('price', False, 0.0, "低点间距不足")
        
        # 价格变化
        price_change = (prices[recent_low_idx] - prices[prev_low_idx]) / prices[prev_low_idx]
        
        # 成交量背离：价格跌但成交量萎缩（卖压减弱）
        recent_volume = volumes[recent_low_idx]
        prev_volume = volumes[prev_low_idx]
        volume_change = (recent_volume - prev_volume) / prev_volume if prev_volume > 0 else 0
        
        # 底背离信号：价格创新低 + 成交量萎缩
        if price_change < -0.002 and volume_change < -0.05:  # 价格跌，成交量萎缩
            strength = min(abs(volume_change) * 2, 1.0)
            return DivergenceSignal('price', True, strength,
                f"价格跌{price_change:.2%}, 成交量萎缩{volume_change:.2%}")
        
        return DivergenceSignal('price', False, 0.0, "无背离")
    
    def check_entry_signal(self, df: pd.DataFrame, idx: int) -> Tuple[bool, int, List[str]]:
        """
        检查买入信号
        Returns: (should_buy, strength, divergence_types)
        strength: 1=2种背离, 2=3种背离
        """
        if self.position != Direction.NONE:
            return False, 0, []
        
        if self.cooldown_counter > 0:
            return False, 0, []
        
        # 时间过滤
        current_time = df.index[idx]
        hour = current_time.hour
        minute = current_time.minute
        time_val = hour * 100 + minute
        
        # 9:30-10:00 不入场
        if 930 <= time_val <= 1000:
            return False, 0, []
        
        # 14:30-15:00 不入场
        if 1430 <= time_val <= 1500:
            return False, 0, []
        
        # 检测三种背离
        bb_div = self.detect_bollinger_divergence(df, idx)
        macd_div = self.detect_macd_divergence(df, idx)
        price_div = self.detect_price_divergence(df, idx)
        
        divergences = []
        if bb_div.detected:
            divergences.append('Bollinger')
        if macd_div.detected:
            divergences.append('MACD')
        if price_div.detected:
            divergences.append('Price')
        
        # 至少2种背离才入场
        if len(divergences) >= 2:
            strength = 2 if len(divergences) == 3 else 1
            return True, strength, divergences
        
        return False, 0, []
    
    def check_exit_signal(self, df: pd.DataFrame, idx: int, channel: Channel) -> Tuple[bool, str, bool]:
        """
        检查卖出信号
        Returns: (should_exit, exit_reason, reduce_only)
        """
        if self.position == Direction.NONE or self.current_trade is None:
            return False, "", False
        
        row = df.iloc[idx]
        entry_price = self.current_trade.option_entry
        
        # 模拟当前期权价格
        if self.position == Direction.LONG:
            strike_price = self.current_trade.entry_price * 1.02
        else:
            strike_price = self.current_trade.entry_price * 0.98
        
        current_option_price = self.simulate_option_price(
            row['close'], strike_price, self.position, 20, row.get('iv', 0.20)
        )
        
        # 计算期权盈亏
        pnl_pct = (current_option_price - entry_price) / entry_price
        
        # 止损检查
        if pnl_pct <= -self.stop_loss_2:
            return True, f"Stop Loss 2: {pnl_pct:.2%}", False
        
        if pnl_pct <= -self.stop_loss_1:
            return True, f"Stop Loss 1: {pnl_pct:.2%}", True
        
        # 通道跌破检查 - 跌破中轨清仓
        bar_idx = idx % self.channel_bars
        if row['close'] < channel.middle_at(bar_idx):
            return True, "Channel Breakdown: Below middle line", False
        
        return False, "", False
    
    def simulate_option_price(self,
                             index_price: float,
                             strike_price: float,
                             direction: Direction,
                             days_to_expiry: int,
                             iv: float) -> float:
        """简化期权定价模拟"""
        if direction == Direction.LONG:  # Call
            intrinsic = max(0, index_price - strike_price)
        else:
            intrinsic = max(0, strike_price - index_price)
        
        time_value = index_price * iv * np.sqrt(days_to_expiry / 365) * 0.4
        option_price = (intrinsic + time_value) / self.contract_multiplier
        
        return max(option_price, 0.0001)
    
    def calculate_position_size(self, strength: int, option_price: float) -> int:
        """
        计算仓位大小
        strength: 1=2种背离(30%), 2=3种背离(50%)
        """
        weights = {1: 0.30, 2: 0.50}
        weight = weights.get(strength, 0.30)
        
        position_value = self.capital * weight
        contracts = int(position_value / (option_price * self.contract_multiplier))
        
        return max(contracts, 1)
    
    def open_position(self, df: pd.DataFrame, idx: int, strength: int, divergences: List[str]):
        """开仓"""
        row = df.iloc[idx]
        
        # 计算行权价（虚值2-3档，约2%价外）
        strike_price = row['close'] * 1.02
        
        # 模拟期权价格
        option_price = self.simulate_option_price(
            row['close'], strike_price, Direction.LONG, 20, row.get('iv', 0.20)
        )
        
        # 计算仓位
        contracts = self.calculate_position_size(strength, option_price)
        
        self.position = Direction.LONG
        self.current_trade = Trade(
            entry_time=df.index[idx],
            entry_price=row['close'],
            option_entry=option_price,
            quantity=contracts,
            divergences=divergences,
            signal_strength=strength
        )
        
        div_str = '+'.join(divergences)
        print(f"[{df.index[idx]}] 开仓 LONG | 背离: {div_str} | 强度: {strength} | "
              f"价格: {row['close']:.2f} | 期权: {option_price:.4f} | 数量: {contracts}")
    
    def close_position(self, df: pd.DataFrame, idx: int, reason: str, reduce_only: bool = False):
        """平仓/减仓"""
        row = df.iloc[idx]
        
        strike_price = self.current_trade.entry_price * 1.02
        
        # 获取IV值
        iv = row['iv'] if 'iv' in row else 0.20
        
        option_price = self.simulate_option_price(
            row['close'], strike_price, self.position, 20, iv
        )
        
        pnl_per_contract = (option_price - self.current_trade.option_entry) * self.contract_multiplier
        
        if reduce_only:
            reduce_contracts = self.current_trade.quantity // 2
            total_pnl = pnl_per_contract * reduce_contracts
            self.capital += total_pnl
            self.current_trade.quantity -= reduce_contracts
            print(f"[{df.index[idx]}] 减仓50% | 原因: {reason} | 盈亏: {total_pnl:,.0f}")
        else:
            total_pnl = pnl_per_contract * self.current_trade.quantity
            self.capital += total_pnl
            
            self.current_trade.exit_time = df.index[idx]
            self.current_trade.exit_price = row['close']
            self.current_trade.option_exit = option_price
            self.current_trade.pnl = total_pnl
            self.current_trade.pnl_pct = total_pnl / (self.current_trade.option_entry * self.current_trade.quantity * self.contract_multiplier)
            self.current_trade.exit_reason = reason
            
            self.trades.append(self.current_trade)
            
            div_str = '+'.join(self.current_trade.divergences)
            print(f"[{df.index[idx]}] 平仓 | 背离: {div_str} | 原因: {reason} | "
                  f"盈亏: {total_pnl:,.0f} ({self.current_trade.pnl_pct:.2%})")
            
            self.position = Direction.NONE
            self.current_trade = None
    
    def run_backtest(self, df: pd.DataFrame) -> BacktestResult:
        """运行回测"""
        print(f"开始三重底背离策略回测: {len(df)} 根K线")
        print(f"初始资金: {self.initial_capital:,.0f}")
        print(f"参数: BB({self.bb_period},{self.bb_std}), MACD({self.macd_fast},{self.macd_slow},{self.macd_signal})")
        
        # 预处理数据
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_macd(df)
        
        start_idx = max(self.bb_period, self.macd_slow, self.divergence_lookback) + 10
        
        for i in range(start_idx, len(df)):
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            
            # 获取当前通道
            channel = draw_channel(
                df['high'].iloc[i-self.channel_bars:i].values,
                df['low'].iloc[i-self.channel_bars:i].values,
                df['close'].iloc[i-self.channel_bars:i].values,
                min_bars=self.channel_bars
            )
            
            # 检查离场
            if self.position != Direction.NONE:
                should_exit, exit_reason, reduce_only = self.check_exit_signal(df, i, channel)
                if should_exit:
                    self.close_position(df, i, exit_reason, reduce_only)
                    if not reduce_only:
                        self.cooldown_counter = self.cooldown_bars
            
            # 检查入场
            if self.position == Direction.NONE:
                should_buy, strength, divergences = self.check_entry_signal(df, i)
                if should_buy:
                    self.open_position(df, i, strength, divergences)
            
            self.equity_curve.append(self.capital)
        
        return self.calculate_result()
    
    def calculate_result(self) -> BacktestResult:
        """计算回测结果"""
        result = BacktestResult(
            trades=self.trades,
            initial_capital=self.initial_capital,
            final_capital=self.capital
        )
        
        if len(self.trades) == 0:
            return result
        
        result.total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        result.win_rate = len(wins) / len(self.trades) if self.trades else 0
        
        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        result.avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        result.avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        
        # 按信号强度统计
        result.strength_2_trades = [t for t in self.trades if t.signal_strength == 1]
        result.strength_3_trades = [t for t in self.trades if t.signal_strength == 2]
        
        if result.strength_2_trades:
            s2_wins = [t for t in result.strength_2_trades if t.pnl > 0]
            result.strength_2_win_rate = len(s2_wins) / len(result.strength_2_trades)
        
        if result.strength_3_trades:
            s3_wins = [t for t in result.strength_3_trades if t.pnl > 0]
            result.strength_3_win_rate = len(s3_wins) / len(result.strength_3_trades)
        
        # 最大回撤
        peak = self.initial_capital
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        result.max_drawdown = max_dd
        
        # 年化收益
        if len(self.equity_curve) > 1:
            years = len(self.equity_curve) / (252 * 8)
            if years > 0:
                result.annual_return = (self.capital / self.initial_capital) ** (1/years) - 1
        
        # 夏普比率
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 8)
        
        # 连续盈亏
        consecutive_wins = 0
        consecutive_losses = 0
        max_wins = 0
        max_losses = 0
        
        for trade in self.trades:
            if trade.pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_wins = max(max_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_losses = max(max_losses, consecutive_losses)
        
        result.max_consecutive_wins = max_wins
        result.max_consecutive_losses = max_losses
        
        return result
    
    def print_report(self, result: BacktestResult):
        """打印回测报告"""
        print("\n" + "="*70)
        print("三重底背离策略回测报告")
        print("="*70)
        print(f"策略参数:")
        print(f"  布林带: ({self.bb_period}, {self.bb_std})")
        print(f"  MACD: ({self.macd_fast}, {self.macd_slow}, {self.macd_signal})")
        print(f"  背离回溯: {self.divergence_lookback}根K线")
        print(f"  最小背离间距: {self.min_divergence_bars}根K线")
        print("-"*70)
        print(f"资金表现:")
        print(f"  初始资金: {result.initial_capital:,.0f}")
        print(f"  最终资金: {result.final_capital:,.0f}")
        print(f"  总收益率: {result.total_return:.2%}")
        print(f"  年化收益: {result.annual_return:.2%}")
        print(f"  最大回撤: {result.max_drawdown:.2%}")
        print(f"  夏普比率: {result.sharpe_ratio:.2f}")
        print("-"*70)
        print(f"交易统计:")
        print(f"  总交易次数: {len(result.trades)}")
        print(f"  总胜率: {result.win_rate:.2%}")
        print(f"  总盈亏比: {result.profit_factor:.2f}")
        print(f"  平均盈利: {result.avg_win:,.0f}")
        print(f"  平均亏损: {result.avg_loss:,.0f}")
        print("-"*70)
        print(f"按信号强度统计:")
        if result.strength_2_trades:
            print(f"  【2种背离】次数: {len(result.strength_2_trades)}, 胜率: {result.strength_2_win_rate:.2%}")
        if result.strength_3_trades:
            print(f"  【3种背离】次数: {len(result.strength_3_trades)}, 胜率: {result.strength_3_win_rate:.2%}")
        print("-"*70)
        print(f"其他指标:")
        print(f"  最大连续盈利: {result.max_consecutive_wins}次")
        print(f"  最大连续亏损: {result.max_consecutive_losses}次")
        print("="*70)


def generate_sample_data(n_bars: int = 2000, start_date: str = "2024-01-01", ticker: str = "TEST") -> pd.DataFrame:
    """生成测试数据（带明确的W底形态和背离特征）"""
    np.random.seed(42 + hash(ticker) % 1000)
    
    dates = pd.date_range(start=start_date, periods=n_bars, freq='30min')
    
    # 基础价格
    base = 4000
    closes = np.zeros(n_bars)
    
    # 生成W底形态序列
    i = 0
    while i < n_bars:
        # 每个周期约150根K线，包含一个W底
        cycle_length = min(150, n_bars - i)
        
        if cycle_length < 50:
            # 剩余部分用随机游走填充
            closes[i:] = closes[i-1] if i > 0 else base
            break
        
        # W底形态：下跌 -> 低点A -> 反弹 -> 下跌 -> 低点B（略高于A，形成背离）-> 上涨
        phase1 = int(cycle_length * 0.25)  # 下跌到低点A
        phase2 = int(cycle_length * 0.15)  # 反弹
        phase3 = int(cycle_length * 0.25)  # 下跌到低点B
        phase4 = cycle_length - phase1 - phase2 - phase3  # 上涨
        
        start_price = closes[i-1] if i > 0 else base
        
        # 低点A（较深）
        low_a = start_price * 0.95
        # 低点B（略高，形成底背离）
        low_b = start_price * 0.955
        
        # 下跌到低点A
        for j in range(phase1):
            progress = j / phase1 if phase1 > 0 else 1
            closes[i + j] = start_price - (start_price - low_a) * progress + np.random.randn() * 5
        
        # 反弹
        rebound = low_a + (start_price - low_a) * 0.4
        for j in range(phase2):
            progress = j / phase2 if phase2 > 0 else 1
            idx = i + phase1 + j
            closes[idx] = low_a + (rebound - low_a) * progress + np.random.randn() * 5
        
        # 下跌到低点B（略高，形成背离）
        for j in range(phase3):
            progress = j / phase3 if phase3 > 0 else 1
            idx = i + phase1 + phase2 + j
            closes[idx] = rebound - (rebound - low_b) * progress + np.random.randn() * 5
        
        # 上涨突破
        end_price = start_price * 1.02
        for j in range(phase4):
            progress = j / phase4 if phase4 > 0 else 1
            idx = i + phase1 + phase2 + phase3 + j
            closes[idx] = low_b + (end_price - low_b) * progress + np.random.randn() * 5
        
        i += cycle_length
    
    # 确保价格不会太低
    closes = np.maximum(closes, 3000)
    
    # 生成OHLC
    highs = closes + np.abs(np.random.randn(n_bars) * 12) + 3
    lows = closes - np.abs(np.random.randn(n_bars) * 12) - 3
    opens = closes + np.random.randn(n_bars) * 8
    
    # 成交量：在低点区域萎缩
    volumes = np.random.lognormal(15, 0.5, n_bars)
    for i in range(0, n_bars, 150):
        # 在每个周期的低点附近成交量萎缩
        low_zone_start = i + 20
        low_zone_end = i + 40
        if low_zone_end < n_bars:
            volumes[low_zone_start:low_zone_end] *= 0.5
    
    ivs = 0.20 + np.random.randn(n_bars) * 0.03
    ivs = np.clip(ivs, 0.12, 0.35)
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'iv': ivs
    }, index=dates)
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='三重底背离策略回测')
    parser.add_argument('--symbol', default='588000', help='标的代码')
    parser.add_argument('--start', default='2024-01-01', help='开始日期')
    parser.add_argument('--end', default='2024-12-31', help='结束日期')
    parser.add_argument('--capital', type=float, default=1000000, help='初始资金')
    parser.add_argument('--data', help='数据文件路径 (CSV格式)')
    parser.add_argument('--bb-period', type=int, default=20, help='布林带周期')
    parser.add_argument('--bb-std', type=float, default=2.0, help='布林带标准差')
    parser.add_argument('--macd-fast', type=int, default=12, help='MACD快线')
    parser.add_argument('--macd-slow', type=int, default=26, help='MACD慢线')
    parser.add_argument('--macd-signal', type=int, default=9, help='MACD信号线')
    
    args = parser.parse_args()
    
    # 加载数据
    if args.data:
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    else:
        print(f"使用模拟数据运行回测: {args.symbol}")
        df = generate_sample_data(n_bars=2000, start_date=args.start, ticker=args.symbol)
    
    # 运行回测
    strategy = TripleDivergenceStrategy(
        initial_capital=args.capital,
        bb_period=args.bb_period,
        bb_std=args.bb_std,
        macd_fast=args.macd_fast,
        macd_slow=args.macd_slow,
        macd_signal=args.macd_signal
    )
    result = strategy.run_backtest(df)
    
    # 打印报告
    strategy.print_report(result)
    
    # 保存结果
    output = {
        'params': {
            'symbol': args.symbol,
            'initial_capital': args.capital,
            'bb_period': args.bb_period,
            'bb_std': args.bb_std,
            'macd_fast': args.macd_fast,
            'macd_slow': args.macd_slow,
            'macd_signal': args.macd_signal
        },
        'result': {
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'max_drawdown': result.max_drawdown,
            'sharpe_ratio': result.sharpe_ratio,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_trades': len(result.trades),
            'avg_win': result.avg_win,
            'avg_loss': result.avg_loss,
            'strength_2_win_rate': result.strength_2_win_rate,
            'strength_3_win_rate': result.strength_3_win_rate
        },
        'trades': [
            {
                'entry_time': t.entry_time.isoformat(),
                'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'divergences': t.divergences,
                'signal_strength': t.signal_strength,
                'exit_reason': t.exit_reason
            }
            for t in result.trades
        ]
    }
    
    output_file = f'triple_divergence_result_{args.symbol}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n结果已保存: {output_file}")
