"""
股指期权趋势爆发策略 - 纯做多版本
基于30分钟K线通道突破，只做多不做空
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from enum import Enum
import json

from channel import draw_channel, Channel


class Direction(Enum):
    LONG = 1   # 做多
    NONE = 0   # 空仓


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


class IndexOptionsLongOnlyStrategy:
    """
    股指期权趋势爆发策略 - 纯做多版本
    只买入看涨期权，不做空
    """
    
    def __init__(self, 
                 initial_capital: float = 1000000,
                 channel_bars: int = 20,
                 volume_threshold: float = 1.3,
                 body_threshold: float = 1.5,
                 iv_rank_max: float = 55,
                 stop_loss_1: float = 0.30,
                 stop_loss_2: float = 0.50,
                 cooldown_bars: int = 3,
                 option_delta: float = 0.30,
                 contract_multiplier: int = 100):
        
        self.initial_capital = initial_capital
        self.channel_bars = channel_bars
        self.volume_threshold = volume_threshold
        self.body_threshold = body_threshold
        self.iv_rank_max = iv_rank_max
        self.stop_loss_1 = stop_loss_1
        self.stop_loss_2 = stop_loss_2
        self.cooldown_bars = cooldown_bars
        self.option_delta = option_delta
        self.contract_multiplier = contract_multiplier
        
        self.capital = initial_capital
        self.position = False
        self.current_trade: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.cooldown_counter = 0
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
                             days_to_expiry: int, iv: float) -> float:
        moneyness = index_price / strike_price
        intrinsic = max(0, index_price - strike_price)
        time_value = index_price * iv * np.sqrt(days_to_expiry / 365) * 0.4
        option_price = (intrinsic + time_value) / self.contract_multiplier
        return max(option_price, 0.0001)
    
    def check_entry_signal(self, df: pd.DataFrame, idx: int, channel: Channel) -> int:
        """检查做多入场信号，返回confidence (0-3)"""
        if self.position:
            return 0
        
        if self.cooldown_counter > 0:
            return 0
        
        # 时间过滤
        current_time = df.index[idx]
        hour = current_time.hour
        minute = current_time.minute
        time_val = hour * 100 + minute
        
        if 930 <= time_val <= 1000:
            return 0
        if 1430 <= time_val <= 1500:
            return 0
        
        row = df.iloc[idx]
        
        if idx < 20:
            return 0
        
        # 只做多：突破通道上轨
        if channel.type not in ['ascending', 'flat']:
            return 0
        
        if not channel.is_breakout_up(idx % self.channel_bars, row['close']):
            return 0
        
        # 计算条件
        conditions_met = 1  # 突破
        
        # 成交量
        avg_volume = df['volume'].iloc[idx-20:idx].mean()
        volume_ratio = row['volume'] / avg_volume if avg_volume > 0 else 0
        if volume_ratio >= self.volume_threshold:
            conditions_met += 1
        
        # K线实体
        body = abs(row['close'] - row['open'])
        avg_body = df.iloc[idx-5:idx].apply(lambda x: abs(x['close'] - x['open']), axis=1).mean()
        body_ratio = body / avg_body if avg_body > 0 else 0
        if body_ratio >= self.body_threshold:
            conditions_met += 1
        
        # IV Rank
        iv_history = df['iv'].iloc[max(0, idx-60):idx].values
        iv_rank = self.calculate_iv_rank(iv_history, row['iv'])
        if iv_rank < self.iv_rank_max:
            conditions_met += 1
        
        return conditions_met - 1 if conditions_met >= 2 else 0
    
    def check_exit_signal(self, df: pd.DataFrame, idx: int, channel: Channel) -> tuple:
        """检查离场信号"""
        if not self.position or self.current_trade is None:
            return False, "", False
        
        row = df.iloc[idx]
        entry_price = self.current_trade.option_entry
        
        # 模拟当前期权价格
        strike_price = self.current_trade.entry_price * 1.02
        option_price = self.simulate_option_price(row['close'], strike_price, 20, row['iv'])
        
        # 计算盈亏
        pnl_pct = (option_price - entry_price) / entry_price
        
        # 止损
        if pnl_pct <= -self.stop_loss_2:
            return True, f"Stop Loss 2: {pnl_pct:.2%}", False
        
        if pnl_pct <= -self.stop_loss_1:
            return True, f"Stop Loss 1: {pnl_pct:.2%}", True
        
        # 通道跌破
        bar_idx = idx % self.channel_bars
        
        if row['close'] < channel.price_at(bar_idx, 'upper'):
            # 检查是否已在通道内停留3根K线
            in_channel = 0
            for i in range(max(0, idx-3), idx):
                if channel.price_at(i % self.channel_bars, 'lower') < df.iloc[i]['close'] < channel.price_at(i % self.channel_bars, 'upper'):
                    in_channel += 1
            
            if in_channel >= 3:
                return True, "Channel Breakdown: Back in channel", False
            
            if row['close'] < channel.middle_at(bar_idx):
                return True, "Channel Breakdown: Below middle", False
            
            return True, "Channel Breakdown: Below upper", True
        
        return False, "", False
    
    def run_backtest(self, df: pd.DataFrame) -> BacktestResult:
        print(f"开始回测 (纯做多版本): {len(df)} 根K线")
        print(f"初始资金: {self.initial_capital:,.0f}")
        
        for i in range(self.channel_bars, len(df)):
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            
            channel = draw_channel(
                df['high'].iloc[i-self.channel_bars:i].values,
                df['low'].iloc[i-self.channel_bars:i].values,
                df['close'].iloc[i-self.channel_bars:i].values,
                min_bars=self.channel_bars
            )
            
            if self.position:
                should_exit, exit_reason, reduce_only = self.check_exit_signal(df, i, channel)
                if should_exit:
                    self.close_position(df, i, exit_reason, reduce_only)
                    if not reduce_only:
                        self.cooldown_counter = self.cooldown_bars
            else:
                confidence = self.check_entry_signal(df, i, channel)
                if confidence > 0:
                    self.open_position(df, i, confidence)
            
            self.equity_curve.append(self.capital)
        
        return self.calculate_result()
    
    def open_position(self, df: pd.DataFrame, idx: int, confidence: int):
        row = df.iloc[idx]
        strike_price = row['close'] * 1.02
        option_price = self.simulate_option_price(row['close'], strike_price, 20, row['iv'])
        contracts = self.calculate_position_size(confidence, option_price)
        
        self.position = True
        self.current_trade = Trade(
            entry_time=df.index[idx],
            entry_price=row['close'],
            option_entry=option_price,
            quantity=contracts
        )
        
        print(f"[{df.index[idx]}] 开仓 LONG | 价格: {row['close']:.2f} | 期权: {option_price:.4f} | 数量: {contracts}")
    
    def close_position(self, df: pd.DataFrame, idx: int, reason: str, reduce_only: bool = False):
        row = df.iloc[idx]
        strike_price = self.current_trade.entry_price * 1.02
        option_price = self.simulate_option_price(row['close'], strike_price, 20, row['iv'])
        
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
            
            print(f"[{df.index[idx]}] 平仓 | 原因: {reason} | 盈亏: {total_pnl:,.0f} ({self.current_trade.pnl_pct:.2%})")
            
            self.position = False
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
            years = len(self.equity_curve) / (252 * 8)
            if years > 0:
                result.annual_return = (self.capital / self.initial_capital) ** (1/years) - 1
        
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 8)
        
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
        print("\n" + "="*60)
        print("回测报告 - 纯做多版本")
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
        print(f"最大连续盈利: {result.max_consecutive_wins}次")
        print(f"最大连续亏损: {result.max_consecutive_losses}次")
        print("="*60)


def generate_sample_data(n_bars: int = 2000, start_date: str = "2020-01-01") -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start=start_date, periods=n_bars, freq='30min')
    base = 4000
    trend = np.sin(np.linspace(0, 4*np.pi, n_bars)) * 200
    noise = np.cumsum(np.random.randn(n_bars) * 5)
    closes = base + trend + noise
    highs = closes + np.abs(np.random.randn(n_bars) * 15) + 5
    lows = closes - np.abs(np.random.randn(n_bars) * 15) - 5
    opens = closes + np.random.randn(n_bars) * 10
    volumes = np.random.lognormal(15, 0.5, n_bars)
    ivs = 0.20 + np.random.randn(n_bars) * 0.05
    ivs = np.clip(ivs, 0.10, 0.50)
    
    return pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'iv': ivs
    }, index=dates)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='股指期权趋势爆发策略 - 纯做多版本')
    parser.add_argument('--capital', type=float, default=1000000, help='初始资金')
    args = parser.parse_args()
    
    df = generate_sample_data(n_bars=2000)
    
    strategy = IndexOptionsLongOnlyStrategy(initial_capital=args.capital)
    result = strategy.run_backtest(df)
    strategy.print_report(result)
    
    output = {
        'strategy': 'IndexOptionsLongOnlyStrategy',
        'params': {
            'initial_capital': args.capital,
            'channel_bars': 20,
            'volume_threshold': 1.3,
            'body_threshold': 1.5
        },
        'result': {
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'max_drawdown': result.max_drawdown,
            'sharpe_ratio': result.sharpe_ratio,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_trades': len(result.trades),
        }
    }
    
    with open('backtest_long_only_result.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n结果已保存: backtest_long_only_result.json")
