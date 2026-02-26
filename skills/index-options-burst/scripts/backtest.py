"""
股指期权趋势爆发策略回测系统
基于30分钟K线通道突破
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from enum import Enum
import json

from channel import draw_channel, Channel


class Direction(Enum):
    LONG = 1   # 做多
    SHORT = -1 # 做空
    NONE = 0   # 空仓


@dataclass
class Trade:
    """单笔交易记录"""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: Direction = Direction.NONE
    entry_price: float = 0.0  # 指数入场价格
    exit_price: float = 0.0   # 指数离场价格
    option_entry: float = 0.0 # 期权入场价格
    option_exit: float = 0.0  # 期权离场价格
    quantity: int = 0         # 合约数量
    pnl: float = 0.0          # 盈亏金额
    pnl_pct: float = 0.0      # 盈亏百分比
    exit_reason: str = ""     # 离场原因


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
    # 多空分开统计
    long_trades: List[Trade] = field(default_factory=list)
    short_trades: List[Trade] = field(default_factory=list)
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    long_profit_factor: float = 0.0
    short_profit_factor: float = 0.0
    long_avg_win: float = 0.0
    long_avg_loss: float = 0.0
    short_avg_win: float = 0.0
    short_avg_loss: float = 0.0


class IndexOptionsStrategy:
    """
    股指期权趋势爆发策略
    """
    
    def __init__(self, 
                 initial_capital: float = 1000000,
                 channel_bars: int = 20,
                 volume_threshold: float = 1.3,
                 body_threshold: float = 1.5,
                 iv_rank_long_max: float = 55,
                 iv_rank_short_max: float = 65,
                 stop_loss_1: float = 0.30,
                 stop_loss_2: float = 0.50,
                 cooldown_bars: int = 3,
                 option_delta: float = 0.30,
                 contract_multiplier: int = 100):
        
        self.initial_capital = initial_capital
        self.channel_bars = channel_bars
        self.volume_threshold = volume_threshold
        self.body_threshold = body_threshold
        self.iv_rank_long_max = iv_rank_long_max
        self.iv_rank_short_max = iv_rank_short_max
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
        
    def calculate_iv_rank(self, iv_history: np.ndarray, current_iv: float) -> float:
        """计算IV Rank (0-100)"""
        if len(iv_history) == 0:
            return 50
        min_iv = np.min(iv_history)
        max_iv = np.max(iv_history)
        if max_iv == min_iv:
            return 50
        return (current_iv - min_iv) / (max_iv - min_iv) * 100
    
    def calculate_position_size(self, confidence: int, option_price: float) -> int:
        """
        计算仓位大小
        confidence: 1=30%, 2=50%, 3=70%
        """
        weights = {1: 0.30, 2: 0.50, 3: 0.70}
        weight = weights.get(confidence, 0.30)
        
        # 可用资金的weight%
        position_value = self.capital * weight
        
        # 合约数量
        contracts = int(position_value / (option_price * self.contract_multiplier))
        
        return max(contracts, 1)
    
    def simulate_option_price(self, 
                             index_price: float,
                             strike_price: float,
                             direction: Direction,
                             days_to_expiry: int,
                             iv: float) -> float:
        """
        简化期权定价模拟
        实际应使用Black-Scholes或市场数据
        """
        # 简化的近似公式
        moneyness = index_price / strike_price
        
        if direction == Direction.LONG:  # Call
            intrinsic = max(0, index_price - strike_price)
        else:  # Put
            intrinsic = max(0, strike_price - index_price)
        
        # 时间价值近似
        time_value = index_price * iv * np.sqrt(days_to_expiry / 365) * 0.4
        
        option_price = (intrinsic + time_value) / self.contract_multiplier
        
        return max(option_price, 0.0001)
    
    def check_entry_signal(self, 
                          df: pd.DataFrame, 
                          idx: int,
                          channel: Channel) -> tuple:
        """
        检查入场信号
        Returns: (direction, confidence) or (None, 0)
        """
        if self.position != Direction.NONE:
            return Direction.NONE, 0
        
        if self.cooldown_counter > 0:
            return Direction.NONE, 0
        
        # 时间过滤
        current_time = df.index[idx]
        hour = current_time.hour
        minute = current_time.minute
        time_val = hour * 100 + minute
        
        # 9:30-10:00 不入场
        if 930 <= time_val <= 1000:
            return Direction.NONE, 0
        
        # 14:30-15:00 不入场
        if 1430 <= time_val <= 1500:
            return Direction.NONE, 0
        
        row = df.iloc[idx]
        
        # 计算成交量条件
        if idx < 20:
            return Direction.NONE, 0
        
        avg_volume = df['volume'].iloc[idx-20:idx].mean()
        volume_ratio = row['volume'] / avg_volume if avg_volume > 0 else 0
        
        # 计算K线实体
        body = abs(row['close'] - row['open'])
        avg_body = df.iloc[idx-5:idx].apply(lambda x: abs(x['close'] - x['open']), axis=1).mean()
        body_ratio = body / avg_body if avg_body > 0 else 0
        
        # IV Rank
        iv_history = df['iv'].iloc[max(0, idx-60):idx].values
        iv_rank = self.calculate_iv_rank(iv_history, row['iv'])
        
        # 做多信号检查
        if channel.type in ['ascending', 'flat']:
            if channel.is_breakout_up(idx % self.channel_bars, row['close']):
                conditions_met = 1  # 突破
                if volume_ratio >= self.volume_threshold:
                    conditions_met += 1
                if body_ratio >= self.body_threshold:
                    conditions_met += 1
                if iv_rank < self.iv_rank_long_max:
                    conditions_met += 1
                
                if conditions_met >= 2:
                    return Direction.LONG, conditions_met - 1
        
        # 做空信号检查
        if channel.type in ['descending', 'flat']:
            if channel.is_breakdown(idx % self.channel_bars, row['close']):
                conditions_met = 1  # 跌破
                if volume_ratio >= self.volume_threshold:
                    conditions_met += 1
                if body_ratio >= self.body_threshold:
                    conditions_met += 1
                if iv_rank < self.iv_rank_short_max:
                    conditions_met += 1
                
                if conditions_met >= 2:
                    return Direction.SHORT, conditions_met - 1
        
        return Direction.NONE, 0
    
    def check_exit_signal(self,
                         df: pd.DataFrame,
                         idx: int,
                         channel: Channel) -> tuple:
        """
        检查离场信号
        Returns: (should_exit, exit_reason, reduce_only)
        """
        if self.position == Direction.NONE or self.current_trade is None:
            return False, "", False
        
        row = df.iloc[idx]
        entry_price = self.current_trade.option_entry
        current_option_price = row.get('option_price', entry_price)
        
        # 计算期权盈亏
        if self.position == Direction.LONG:
            pnl_pct = (current_option_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_option_price) / entry_price
        
        # 止损检查
        if pnl_pct <= -self.stop_loss_2:
            return True, f"Stop Loss 2: {pnl_pct:.2%}", False
        
        if pnl_pct <= -self.stop_loss_1:
            # 减仓50%
            return True, f"Stop Loss 1: {pnl_pct:.2%}", True
        
        # 通道跌破检查
        bar_idx = idx % self.channel_bars
        
        if self.position == Direction.LONG:
            # 跌破原通道上轨减仓
            if row['close'] < channel.price_at(bar_idx, 'upper'):
                # 检查是否已在通道内停留3根K线
                in_channel = 0
                for i in range(max(0, idx-3), idx):
                    if channel.price_at(i % self.channel_bars, 'lower') < df.iloc[i]['close'] < channel.price_at(i % self.channel_bars, 'upper'):
                        in_channel += 1
                
                if in_channel >= 3:
                    return True, "Channel Breakdown: Back in channel", False
                
                # 跌破中轨清仓
                if row['close'] < channel.middle_at(bar_idx):
                    return True, "Channel Breakdown: Below middle", False
                
                return True, "Channel Breakdown: Below upper", True
        
        else:  # SHORT
            # 突破原通道下轨减仓
            if row['close'] > channel.price_at(bar_idx, 'lower'):
                # 检查是否已在通道内停留3根K线
                in_channel = 0
                for i in range(max(0, idx-3), idx):
                    if channel.price_at(i % self.channel_bars, 'lower') < df.iloc[i]['close'] < channel.price_at(i % self.channel_bars, 'upper'):
                        in_channel += 1
                
                if in_channel >= 3:
                    return True, "Channel Breakout: Back in channel", False
                
                # 突破中轨清仓
                if row['close'] > channel.middle_at(bar_idx):
                    return True, "Channel Breakout: Above middle", False
                
                return True, "Channel Breakout: Above lower", True
        
        return False, "", False
    
    def run_backtest(self, df: pd.DataFrame) -> BacktestResult:
        """
        运行回测
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'iv']
        
        Returns:
            BacktestResult
        """
        print(f"开始回测: {len(df)} 根K线")
        print(f"初始资金: {self.initial_capital:,.0f}")
        
        for i in range(self.channel_bars, len(df)):
            # 更新冷却计数器
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
                    
                    if not reduce_only:  # 完全离场
                        self.cooldown_counter = self.cooldown_bars
            
            # 检查入场
            if self.position == Direction.NONE:
                direction, confidence = self.check_entry_signal(df, i, channel)
                
                if direction != Direction.NONE:
                    self.open_position(df, i, direction, confidence, channel)
            
            # 记录权益曲线
            self.equity_curve.append(self.capital)
        
        # 计算结果
        return self.calculate_result()
    
    def open_position(self, df: pd.DataFrame, idx: int, direction: Direction, confidence: int, channel: Channel):
        """开仓"""
        row = df.iloc[idx]
        
        # 计算行权价（虚值2-3档，约2%价外）
        if direction == Direction.LONG:
            strike_price = row['close'] * 1.02
        else:
            strike_price = row['close'] * 0.98
        
        # 模拟期权价格
        option_price = self.simulate_option_price(
            row['close'], strike_price, direction, 20, row['iv']
        )
        
        # 计算仓位
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
        """平仓/减仓"""
        row = df.iloc[idx]
        
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
            pnl_per_contract = (option_price - self.current_trade.option_entry) * self.contract_multiplier
        else:
            pnl_per_contract = (self.current_trade.option_entry - option_price) * self.contract_multiplier
        
        if reduce_only:
            # 减仓50%
            reduce_contracts = self.current_trade.quantity // 2
            total_pnl = pnl_per_contract * reduce_contracts
            self.capital += total_pnl
            self.current_trade.quantity -= reduce_contracts
            
            print(f"[{df.index[idx]}] 减仓50% | 原因: {reason} | 盈亏: {total_pnl:,.0f}")
        else:
            # 全部平仓
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
        """计算回测结果"""
        result = BacktestResult(
            trades=self.trades,
            initial_capital=self.initial_capital,
            final_capital=self.capital
        )
        
        if len(self.trades) == 0:
            return result
        
        # 基础统计
        result.total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # 胜率
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        result.win_rate = len(wins) / len(self.trades) if self.trades else 0
        
        # 盈亏比
        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # 平均盈亏
        result.avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        result.avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        
        # 多空分开统计
        result.long_trades = [t for t in self.trades if t.direction == Direction.LONG]
        result.short_trades = [t for t in self.trades if t.direction == Direction.SHORT]
        
        # 做多统计
        if result.long_trades:
            long_wins = [t for t in result.long_trades if t.pnl > 0]
            long_losses = [t for t in result.long_trades if t.pnl <= 0]
            result.long_win_rate = len(long_wins) / len(result.long_trades)
            long_total_wins = sum(t.pnl for t in long_wins)
            long_total_losses = abs(sum(t.pnl for t in long_losses))
            result.long_profit_factor = long_total_wins / long_total_losses if long_total_losses > 0 else float('inf')
            result.long_avg_win = np.mean([t.pnl for t in long_wins]) if long_wins else 0
            result.long_avg_loss = np.mean([t.pnl for t in long_losses]) if long_losses else 0
        
        # 做空统计
        if result.short_trades:
            short_wins = [t for t in result.short_trades if t.pnl > 0]
            short_losses = [t for t in result.short_trades if t.pnl <= 0]
            result.short_win_rate = len(short_wins) / len(result.short_trades)
            short_total_wins = sum(t.pnl for t in short_wins)
            short_total_losses = abs(sum(t.pnl for t in short_losses))
            result.short_profit_factor = short_total_wins / short_total_losses if short_total_losses > 0 else float('inf')
            result.short_avg_win = np.mean([t.pnl for t in short_wins]) if short_wins else 0
            result.short_avg_loss = np.mean([t.pnl for t in short_losses]) if short_losses else 0
        
        # 最大回撤
        peak = self.initial_capital
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        result.max_drawdown = max_dd
        
        # 年化收益（假设数据跨度）
        if len(self.equity_curve) > 1:
            years = len(self.equity_curve) / (252 * 8)  # 假设每天8根30分钟K线
            if years > 0:
                result.annual_return = (self.capital / self.initial_capital) ** (1/years) - 1
        
        # 夏普比率（简化计算）
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
        print("\n" + "="*60)
        print("回测报告")
        print("="*60)
        print(f"初始资金: {result.initial_capital:,.0f}")
        print(f"最终资金: {result.final_capital:,.0f}")
        print(f"总收益率: {result.total_return:.2%}")
        print(f"年化收益: {result.annual_return:.2%}")
        print(f"最大回撤: {result.max_drawdown:.2%}")
        print(f"夏普比率: {result.sharpe_ratio:.2f}")
        print("-"*60)
        print(f"总交易次数: {len(result.trades)}")
        print(f"总胜率: {result.win_rate:.2%}")
        print(f"总盈亏比: {result.profit_factor:.2f}")
        print(f"平均盈利: {result.avg_win:,.0f}")
        print(f"平均亏损: {result.avg_loss:,.0f}")
        print("-"*60)
        # 多空分开显示
        if result.long_trades:
            print(f"【做多】")
            print(f"  交易次数: {len(result.long_trades)}")
            print(f"  胜率: {result.long_win_rate:.2%}")
            print(f"  盈亏比: {result.long_profit_factor:.2f}")
            print(f"  平均盈利: {result.long_avg_win:,.0f}")
            print(f"  平均亏损: {result.long_avg_loss:,.0f}")
        if result.short_trades:
            print(f"【做空】")
            print(f"  交易次数: {len(result.short_trades)}")
            print(f"  胜率: {result.short_win_rate:.2%}")
            print(f"  盈亏比: {result.short_profit_factor:.2f}")
            print(f"  平均盈利: {result.short_avg_win:,.0f}")
            print(f"  平均亏损: {result.short_avg_loss:,.0f}")
        print("-"*60)
        print(f"最大连续盈利: {result.max_consecutive_wins}次")
        print(f"最大连续亏损: {result.max_consecutive_losses}次")
        print("="*60)


def generate_sample_data(n_bars: int = 2000, start_date: str = "2020-01-01") -> pd.DataFrame:
    """生成测试数据"""
    np.random.seed(42)
    
    # 生成时间索引（30分钟K线，每天8根，4小时交易）
    dates = pd.date_range(start=start_date, periods=n_bars, freq='30min')
    
    # 生成价格数据（带趋势通道）
    base = 4000
    trend = np.sin(np.linspace(0, 4*np.pi, n_bars)) * 200  # 周期性趋势
    noise = np.cumsum(np.random.randn(n_bars) * 5)  # 随机游走
    
    closes = base + trend + noise
    highs = closes + np.abs(np.random.randn(n_bars) * 15) + 5
    lows = closes - np.abs(np.random.randn(n_bars) * 15) - 5
    opens = closes + np.random.randn(n_bars) * 10
    
    # 成交量
    volumes = np.random.lognormal(15, 0.5, n_bars)
    
    # 隐含波动率 (20%左右)
    ivs = 0.20 + np.random.randn(n_bars) * 0.05
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
    
    parser = argparse.ArgumentParser(description='股指期权趋势爆发策略回测')
    parser.add_argument('--symbol', default='000300', help='标的代码')
    parser.add_argument('--start', default='2020-01-01', help='开始日期')
    parser.add_argument('--end', default='2024-12-31', help='结束日期')
    parser.add_argument('--capital', type=float, default=1000000, help='初始资金')
    parser.add_argument('--data', help='数据文件路径 (CSV格式)')
    
    args = parser.parse_args()
    
    # 加载数据
    if args.data:
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    else:
        print("使用模拟数据运行回测...")
        df = generate_sample_data(n_bars=2000, start_date=args.start)
    
    # 运行回测
    strategy = IndexOptionsStrategy(initial_capital=args.capital)
    result = strategy.run_backtest(df)
    
    # 打印报告
    strategy.print_report(result)
    
    # 保存结果
    output = {
        'params': {
            'initial_capital': args.capital,
            'channel_bars': strategy.channel_bars,
            'volume_threshold': strategy.volume_threshold,
            'body_threshold': strategy.body_threshold
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
            'avg_loss': result.avg_loss
        },
        'trades': [
            {
                'entry_time': t.entry_time.isoformat(),
                'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                'direction': t.direction.name,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'exit_reason': t.exit_reason
            }
            for t in result.trades
        ]
    }
    
    output_file = f'backtest_result_{args.symbol}_{args.start}_{args.end}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n结果已保存: {output_file}")
