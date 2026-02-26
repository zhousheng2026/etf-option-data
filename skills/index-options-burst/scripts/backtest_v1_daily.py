"""
股指期权趋势爆发策略回测 - 日线20日高低点突破版本
原始策略：日线周期，20日高低点突破入场，10日均线离场
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from enum import Enum
import json


class Direction(Enum):
    LONG = 1
    SHORT = -1
    NONE = 0


@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: Direction = Direction.NONE
    entry_price: float = 0.0
    exit_price: float = 0.0
    option_entry: float = 0.0
    option_exit: float = 0.0
    quantity: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
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


class IndexOptionsStrategyV1:
    """
    股指期权趋势爆发策略 - 原始日线版本
    - 入场: 突破20日高点/跌破20日低点
    - 离场: 跌破/突破10日均线
    """
    
    def __init__(self,
                 initial_capital: float = 1000000,
                 lookback: int = 20,
                 exit_ma: int = 10,
                 volume_threshold: float = 1.2,
                 iv_rank_long_max: float = 50,
                 iv_rank_short_max: float = 60,
                 stop_loss_1: float = 0.30,
                 stop_loss_2: float = 0.50,
                 option_delta: float = 0.30,
                 contract_multiplier: int = 100):
        
        self.initial_capital = initial_capital
        self.lookback = lookback
        self.exit_ma = exit_ma
        self.volume_threshold = volume_threshold
        self.iv_rank_long_max = iv_rank_long_max
        self.iv_rank_short_max = iv_rank_short_max
        self.stop_loss_1 = stop_loss_1
        self.stop_loss_2 = stop_loss_2
        self.option_delta = option_delta
        self.contract_multiplier = contract_multiplier
        
        self.capital = initial_capital
        self.position: Direction = Direction.NONE
        self.current_trade: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
    
    def calculate_iv_rank(self, iv_history: np.ndarray, current_iv: float) -> float:
        if len(iv_history) == 0:
            return 50
        min_iv = np.min(iv_history)
        max_iv = np.max(iv_history)
        if max_iv == min_iv:
            return 50
        return (current_iv - min_iv) / (max_iv - min_iv) * 100
    
    def calculate_position_size(self, confidence: int, option_price: float) -> int:
        weights = {1: 0.30, 2: 0.50, 3: 0.70}
        weight = weights.get(confidence, 0.30)
        position_value = self.capital * weight
        contracts = int(position_value / (option_price * self.contract_multiplier))
        return max(contracts, 1)
    
    def simulate_option_price(self, index_price: float, strike_price: float,
                             direction: Direction, days_to_expiry: int, iv: float) -> float:
        moneyness = index_price / strike_price
        if direction == Direction.LONG:
            intrinsic = max(0, index_price - strike_price)
        else:
            intrinsic = max(0, strike_price - index_price)
        time_value = index_price * iv * np.sqrt(days_to_expiry / 365) * 0.4
        option_price = (intrinsic + time_value) / self.contract_multiplier
        return max(option_price, 0.0001)
    
    def check_entry_signal(self, df: pd.DataFrame, idx: int) -> tuple:
        if self.position != Direction.NONE:
            return Direction.NONE, 0
        
        if idx < self.lookback:
            return Direction.NONE, 0
        
        row = df.iloc[idx]
        
        # 20日高低点
        high_20 = df['high'].iloc[idx-self.lookback:idx].max()
        low_20 = df['low'].iloc[idx-self.lookback:idx].min()
        
        # 成交量
        avg_volume = df['volume'].iloc[idx-20:idx].mean()
        volume_ratio = row['volume'] / avg_volume if avg_volume > 0 else 0
        
        # IV Rank
        iv_history = df['iv'].iloc[max(0, idx-60):idx].values
        iv_rank = self.calculate_iv_rank(iv_history, row['iv'])
        
        # MACD简化判断
        if idx >= 26:
            ema12 = df['close'].iloc[idx-12:idx].mean()
            ema26 = df['close'].iloc[idx-26:idx].mean()
            macd_positive = ema12 > ema26
        else:
            macd_positive = row['close'] > df['close'].iloc[idx-10:idx].mean()
        
        # 做多信号: 突破20日高点
        if row['close'] > high_20:
            conditions_met = 1
            if volume_ratio >= self.volume_threshold:
                conditions_met += 1
            if macd_positive:
                conditions_met += 1
            if iv_rank < self.iv_rank_long_max:
                conditions_met += 1
            
            if conditions_met >= 3:
                return Direction.LONG, conditions_met - 2
        
        # 做空信号: 跌破20日低点
        if row['close'] < low_20:
            conditions_met = 1
            if volume_ratio >= self.volume_threshold:
                conditions_met += 1
            if not macd_positive:
                conditions_met += 1
            if iv_rank < self.iv_rank_short_max:
                conditions_met += 1
            
            if conditions_met >= 3:
                return Direction.SHORT, conditions_met - 2
        
        return Direction.NONE, 0
    
    def check_exit_signal(self, df: pd.DataFrame, idx: int) -> tuple:
        if self.position == Direction.NONE or self.current_trade is None:
            return False, "", False
        
        row = df.iloc[idx]
        entry_price = self.current_trade.option_entry
        
        # 模拟当前期权价格
        if self.position == Direction.LONG:
            strike_price = self.current_trade.entry_price * 1.02
        else:
            strike_price = self.current_trade.entry_price * 0.98
        
        option_price = self.simulate_option_price(
            row['close'], strike_price, self.position, 20, row['iv']
        )
        
        # 计算盈亏
        if self.position == Direction.LONG:
            pnl_pct = (option_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - option_price) / entry_price
        
        # 止损
        if pnl_pct <= -self.stop_loss_2:
            return True, f"Stop Loss 2: {pnl_pct:.2%}", False
        
        if pnl_pct <= -self.stop_loss_1:
            return True, f"Stop Loss 1: {pnl_pct:.2%}", True
        
        # 10日均线离场
        if idx >= self.exit_ma:
            ma10 = df['close'].iloc[idx-self.exit_ma:idx].mean()
            
            if self.position == Direction.LONG and row['close'] < ma10:
                return True, f"Below MA{self.exit_ma}", False
            
            if self.position == Direction.SHORT and row['close'] > ma10:
                return True, f"Above MA{self.exit_ma}", False
        
        return False, "", False
    
    def run_backtest(self, df: pd.DataFrame) -> BacktestResult:
        print(f"开始回测 (日线20日高低点版本): {len(df)} 根K线")
        print(f"初始资金: {self.initial_capital:,.0f}")
        
        for i in range(self.lookback, len(df)):
            # 检查离场
            if self.position != Direction.NONE:
                should_exit, exit_reason, reduce_only = self.check_exit_signal(df, i)
                
                if should_exit:
                    self.close_position(df, i, exit_reason, reduce_only)
            
            # 检查入场
            if self.position == Direction.NONE:
                direction, confidence = self.check_entry_signal(df, i)
                
                if direction != Direction.NONE:
                    self.open_position(df, i, direction, confidence)
            
            self.equity_curve.append(self.capital)
        
        return self.calculate_result()
    
    def open_position(self, df: pd.DataFrame, idx: int, direction: Direction, confidence: int):
        row = df.iloc[idx]
        
        if direction == Direction.LONG:
            strike_price = row['close'] * 1.02
        else:
            strike_price = row['close'] * 0.98
        
        option_price = self.simulate_option_price(
            row['close'], strike_price, direction, 20, row['iv']
        )
        
        contracts = self.calculate_position_size(confidence, option_price)
        
        self.position = direction
        self.current_trade = Trade(
            entry_time=df.index[idx],
            direction=direction,
            entry_price=row['close'],
            option_entry=option_price,
            quantity=contracts
        )
        
        print(f"[{df.index[idx]}] 开仓 {direction.name} | 价格: {row['close']:.2f} | 期权: {option_price:.4f} | 数量: {contracts}")
    
    def close_position(self, df: pd.DataFrame, idx: int, reason: str, reduce_only: bool = False):
        row = df.iloc[idx]
        
        if self.position == Direction.LONG:
            strike_price = self.current_trade.entry_price * 1.02
        else:
            strike_price = self.current_trade.entry_price * 0.98
        
        option_price = self.simulate_option_price(
            row['close'], strike_price, self.position, 20, row['iv']
        )
        
        if self.position == Direction.LONG:
            pnl_per_contract = (option_price - self.current_trade.option_entry) * self.contract_multiplier
        else:
            pnl_per_contract = (self.current_trade.option_entry - option_price) * self.contract_multiplier
        
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
            
            print(f"[{df.index[idx]}] 平仓 | 原因: {reason} | 盈亏: {total_pnl:,.0f} ({self.current_trade.pnl_pct:.2%})")
            
            self.position = Direction.NONE
            self.current_trade = None
    
    def calculate_result(self) -> BacktestResult:
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
        result.win_rate = len(wins) / len(self.trades)
        
        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        result.avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        result.avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        
        peak = self.initial_capital
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        result.max_drawdown = max_dd
        
        if len(self.equity_curve) > 1:
            years = len(self.equity_curve) / 252
            if years > 0:
                result.annual_return = (self.capital / self.initial_capital) ** (1/years) - 1
        
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        return result
    
    def print_report(self, result: BacktestResult):
        print("\n" + "="*60)
        print("回测报告 - 日线20日高低点突破版本")
        print("="*60)
        print(f"初始资金: {result.initial_capital:,.0f}")
        print(f"最终资金: {result.final_capital:,.0f}")
        print(f"总收益率: {result.total_return:.2%}")
        print(f"年化收益: {result.annual_return:.2%}")
        print(f"最大回撤: {result.max_drawdown:.2%}")
        print(f"夏普比率: {result.sharpe_ratio:.2f}")
        print("-"*60)
        print(f"总交易次数: {len(result.trades)}")
        print(f"胜率: {result.win_rate:.2%}")
        print(f"盈亏比: {result.profit_factor:.2f}")
        print(f"平均盈利: {result.avg_win:,.0f}")
        print(f"平均亏损: {result.avg_loss:,.0f}")
        print("="*60)


def generate_daily_data(n_days: int = 500, start_date: str = "2020-01-01") -> pd.DataFrame:
    """生成日线测试数据"""
    np.random.seed(42)
    
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    base = 4000
    trend = np.sin(np.linspace(0, 4*np.pi, n_days)) * 300
    noise = np.cumsum(np.random.randn(n_days) * 20)
    
    closes = base + trend + noise
    highs = closes + np.abs(np.random.randn(n_days) * 25) + 10
    lows = closes - np.abs(np.random.randn(n_days) * 25) - 10
    opens = closes + np.random.randn(n_days) * 15
    
    volumes = np.random.lognormal(16, 0.5, n_days)
    ivs = 0.20 + np.random.randn(n_days) * 0.05
    ivs = np.clip(ivs, 0.10, 0.50)
    
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--capital', type=float, default=1000000)
    args = parser.parse_args()
    
    df = generate_daily_data(n_days=500)
    
    strategy = IndexOptionsStrategyV1(initial_capital=args.capital)
    result = strategy.run_backtest(df)
    strategy.print_report(result)
