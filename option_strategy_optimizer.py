#!/usr/bin/env python3
"""
股指期权布林带均值回归策略 - 参数优化模块
整合底背离检测 + 多周期 + 多资金档位 + 期权合约选择
"""

import pandas as pd
import numpy as np
import itertools
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StrategyParams:
    """策略参数"""
    # MACD参数
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # 布林带参数
    boll_period: int = 20
    boll_std: float = 2.0
    
    # 背离检测参数
    divergence_lookback: int = 20  # 背离检测回看周期
    
    # 入场条件
    entry_threshold: float = 0.0  # 触及布林带极值的阈值
    
    # 平仓条件
    exit_on_middle: bool = True   # 是否在中轨平仓
    exit_on_reverse: bool = True  # 是否在反向MACD交叉平仓
    
    # 周期
    timeframe: str = "15min"  # 1min, 5min, 15min, 30min, 60min
    
    # 资金档位
    capital: int = 5000  # 1000, 5000, 10000
    
    # 期权参数
    strike_otm: int = 2  # 虚值档位 1-5
    contract_month: str = "current"  # current, next
    
    def to_dict(self):
        return asdict(self)


@dataclass
class BacktestResult:
    """回测结果"""
    params: StrategyParams
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    win_loss_ratio: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    final_capital: float
    avg_trade_return: float
    
    def to_dict(self):
        return {
            'params': self.params.to_dict(),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'win_loss_ratio': self.win_loss_ratio,
            'total_return': self.total_return,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'final_capital': self.final_capital,
            'avg_trade_return': self.avg_trade_return
        }


class TechnicalIndicators:
    """技术指标计算"""
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
        """计算MACD指标"""
        df = df.copy()
        df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # MACD金叉/死叉
        df['macd_golden_cross'] = (df['macd'] > df['macd_signal']) & \
                                   (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['macd_dead_cross'] = (df['macd'] < df['macd_signal']) & \
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        return df
    
    @staticmethod
    def calculate_bollinger(df: pd.DataFrame, period: int, std_dev: float) -> pd.DataFrame:
        """计算布林带指标"""
        df = df.copy()
        df['boll_mid'] = df['close'].rolling(window=period).mean()
        df['boll_std'] = df['close'].rolling(window=period).std()
        df['boll_upper'] = df['boll_mid'] + std_dev * df['boll_std']
        df['boll_lower'] = df['boll_mid'] - std_dev * df['boll_std']
        
        # 价格相对于布林带的位置
        df['boll_position'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'])
        
        # 触及上轨/下轨极值
        df['touch_upper'] = df['high'] >= df['boll_upper']
        df['touch_lower'] = df['low'] <= df['boll_lower']
        
        return df
    
    @staticmethod
    def detect_divergence(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """
        检测价格与MACD的底背离
        底背离：价格创新低，MACD未创新低（抬高）
        """
        df = df.copy()
        df['divergence_bull'] = False
        
        for i in range(lookback, len(df)):
            # 获取回看窗口
            window = df.iloc[i-lookback:i+1]
            
            # 找价格低点
            price_low_idx = window['low'].idxmin()
            price_low = window.loc[price_low_idx, 'low']
            
            # 找MACD低点
            macd_low_idx = window['macd'].idxmin()
            macd_low = window.loc[macd_low_idx, 'macd']
            
            # 当前价格接近窗口低点
            current_low = df.iloc[i]['low']
            current_macd = df.iloc[i]['macd']
            
            # 底背离条件：价格创新低或接近新低，但MACD高于前期低点
            price_near_low = current_low <= price_low * 1.001  # 允许0.1%误差
            macd_higher = current_macd > macd_low * 0.95  # MACD未创新低
            
            if price_near_low and macd_higher and current_macd < 0:
                df.loc[df.index[i], 'divergence_bull'] = True
        
        return df


class OptionStrategy:
    """股指期权策略"""
    
    # 手续费配置（元/张，双边）
    COMMISSION_PER_CONTRACT = 4.0  # 开仓2元 + 平仓2元
    
    # 合约乘数
    CONTRACT_MULTIPLIER = 10000  # 股指期权每点10000元
    
    def __init__(self, params: StrategyParams):
        self.params = params
        self.initial_capital = params.capital
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        df = TechnicalIndicators.calculate_macd(
            df, 
            self.params.macd_fast, 
            self.params.macd_slow, 
            self.params.macd_signal
        )
        df = TechnicalIndicators.calculate_bollinger(
            df, 
            self.params.boll_period, 
            self.params.boll_std
        )
        df = TechnicalIndicators.detect_divergence(df, self.params.divergence_lookback)
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = df.copy()
        df['long_signal'] = False
        df['short_signal'] = False
        df['close_signal'] = False
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            
            # 跳过NaN值
            if pd.isna(row['macd']) or pd.isna(row['boll_mid']):
                continue
            
            # 做多信号：MACD金叉 + 价格触及BOLL下轨 + 底背离
            if (row['macd_golden_cross'] and 
                row['touch_lower'] and 
                row['divergence_bull']):
                df.loc[df.index[i], 'long_signal'] = True
            
            # 做空信号：MACD死叉 + 价格触及BOLL上轨
            if row['macd_dead_cross'] and row['touch_upper']:
                df.loc[df.index[i], 'short_signal'] = True
        
        return df
    
    def backtest(self, df: pd.DataFrame) -> BacktestResult:
        """执行回测"""
        # 计算指标和信号
        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
        
        # 初始化
        position = 0  # 0: 空仓, 1: 多头(认购), -1: 空头(认沽)
        capital = self.initial_capital
        trades = []
        equity_curve = [capital]
        
        entry_price = 0
        entry_time = None
        contracts = 0
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # 跳过NaN值
            if pd.isna(row['macd']) or pd.isna(row['boll_mid']):
                equity_curve.append(capital)
                continue
            
            # 生成交易信号
            long_signal = row['long_signal']
            short_signal = row['short_signal']
            close_signal = False
            
            # 平仓信号判断
            if position == 1:  # 多头持仓
                if self.params.exit_on_reverse and row['macd_dead_cross']:
                    close_signal = True
                elif (self.params.exit_on_middle and 
                      prev_row['close'] < prev_row['boll_mid'] and 
                      row['close'] > row['boll_mid']):
                    close_signal = True
                    
            elif position == -1:  # 空头持仓
                if self.params.exit_on_reverse and row['macd_golden_cross']:
                    close_signal = True
                elif (self.params.exit_on_middle and 
                      prev_row['close'] > prev_row['boll_mid'] and 
                      row['close'] < row['boll_mid']):
                    close_signal = True
            
            # 执行交易
            if position == 0:
                if long_signal:
                    position = 1
                    entry_price = row['close']
                    entry_time = row['datetime']
                    # 计算可买合约数（简化：按ETF价格计算）
                    contracts = max(1, int(capital / (entry_price * 100)))
                    
                elif short_signal:
                    position = -1
                    entry_price = row['close']
                    entry_time = row['datetime']
                    contracts = max(1, int(capital / (entry_price * 100)))
                    
            else:
                if close_signal:
                    exit_price = row['close']
                    exit_time = row['datetime']
                    
                    # 计算盈亏（简化计算，实际应按期权价格）
                    if position == 1:
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price
                    
                    # 估算期权盈亏（虚值2档，Delta约0.4）
                    option_delta = 0.4
                    pnl = capital * pnl_pct * option_delta
                    
                    # 扣除手续费
                    commission = self.COMMISSION_PER_CONTRACT * contracts
                    pnl -= commission
                    
                    capital += pnl
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'position': position,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'contracts': contracts,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'capital': capital
                    })
                    
                    position = 0
                    contracts = 0
            
            equity_curve.append(capital)
        
        # 强制平仓
        if position != 0:
            exit_price = df.iloc[-1]['close']
            exit_time = df.iloc[-1]['datetime']
            
            if position == 1:
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
            
            option_delta = 0.4
            pnl = capital * pnl_pct * option_delta
            commission = self.COMMISSION_PER_CONTRACT * contracts
            pnl -= commission
            capital += pnl
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'position': position,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'contracts': contracts,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'capital': capital
            })
        
        # 计算回测指标
        return self._calculate_metrics(trades, equity_curve)
    
    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[float]) -> BacktestResult:
        """计算回测指标"""
        if not trades:
            return BacktestResult(
                params=self.params,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                profit_factor=0,
                win_loss_ratio=0,
                total_return=0,
                max_drawdown=0,
                sharpe_ratio=0,
                final_capital=self.initial_capital,
                avg_trade_return=0
            )
        
        trades_df = pd.DataFrame(trades)
        
        total_trades = len(trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
        total_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum()) if losing_trades > 0 else 1
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].mean()) if losing_trades > 0 else 1
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        final_capital = equity_curve[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        avg_trade_return = total_return / total_trades if total_trades > 0 else 0
        
        # 最大回撤
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 夏普比率（简化）
        returns = equity_series.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        return BacktestResult(
            params=self.params,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            win_loss_ratio=win_loss_ratio,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            final_capital=final_capital,
            avg_trade_return=avg_trade_return
        )


class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results: List[BacktestResult] = []
    
    def optimize(self, 
                 timeframes: List[str] = None,
                 capitals: List[int] = None,
                 otm_levels: List[int] = None) -> List[BacktestResult]:
        """
        执行参数优化
        
        Args:
            timeframes: 周期列表 ['5min', '15min', '30min', '60min']
            capitals: 资金档位 [1000, 5000, 10000]
            otm_levels: 虚值档位 [1, 2, 3, 4, 5]
        """
        if timeframes is None:
            timeframes = ['5min', '15min', '30min', '60min']
        if capitals is None:
            capitals = [1000, 5000, 10000]
        if otm_levels is None:
            otm_levels = [1, 2, 3, 4, 5]
        
        # MACD参数网格
        macd_fast_range = [8, 12, 16]
        macd_slow_range = [21, 26, 30]
        macd_signal_range = [7, 9, 12]
        
        # 布林带参数网格
        boll_period_range = [15, 20, 25]
        boll_std_range = [1.5, 2.0, 2.5]
        
        # 生成所有参数组合
        param_combinations = []
        for tf in timeframes:
            for cap in capitals:
                for otm in otm_levels:
                    for fast in macd_fast_range:
                        for slow in macd_slow_range:
                            for signal in macd_signal_range:
                                if fast >= slow:
                                    continue
                                for boll_p in boll_period_range:
                                    for boll_s in boll_std_range:
                                        params = StrategyParams(
                                            macd_fast=fast,
                                            macd_slow=slow,
                                            macd_signal=signal,
                                            boll_period=boll_p,
                                            boll_std=boll_s,
                                            timeframe=tf,
                                            capital=cap,
                                            strike_otm=otm
                                        )
                                        param_combinations.append(params)
        
        print(f"总参数组合数: {len(param_combinations)}")
        print(f"周期: {timeframes}")
        print(f"资金档位: {capitals}")
        print(f"虚值档位: {otm_levels}")
        print()
        
        # 执行回测
        for i, params in enumerate(param_combinations):
            try:
                strategy = OptionStrategy(params)
                result = strategy.backtest(self.data)
                self.results.append(result)
                
                if (i + 1) % 100 == 0:
                    print(f"  已完成 {i+1}/{len(param_combinations)} 组参数测试")
                    
            except Exception as e:
                print(f"参数组合 {params} 回测失败: {e}")
                continue
        
        print(f"\n✓ 参数优化完成，有效结果: {len(self.results)} 组")
        return self.results
    
    def get_best_params(self, min_trades: int = 5, sort_by: str = 'win_rate') -> List[BacktestResult]:
        """获取最优参数组合"""
        # 过滤交易次数不足的
        filtered = [r for r in self.results if r.total_trades >= min_trades]
        
        if not filtered:
            print(f"警告: 没有交易次数>={min_trades}次的参数组合")
            return sorted(self.results, key=lambda x: x.win_rate, reverse=True)[:10]
        
        # 排序
        if sort_by == 'win_rate':
            sorted_results = sorted(filtered, key=lambda x: x.win_rate, reverse=True)
        elif sort_by == 'profit_factor':
            sorted_results = sorted(filtered, key=lambda x: x.profit_factor, reverse=True)
        elif sort_by == 'sharpe':
            sorted_results = sorted(filtered, key=lambda x: x.sharpe_ratio, reverse=True)
        else:
            sorted_results = sorted(filtered, key=lambda x: x.total_return, reverse=True)
        
        return sorted_results
    
    def save_results(self, filepath: str):
        """保存优化结果"""
        results_dict = [r.to_dict() for r in self.results]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        print(f"✓ 优化结果已保存至: {filepath}")
    
    def generate_report(self, top_n: int = 10) -> str:
        """生成优化报告"""
        if not self.results:
            return "无回测结果"
        
        # 整体统计
        total_trades = sum(r.total_trades for r in self.results)
        total_wins = sum(r.winning_trades for r in self.results)
        overall_win_rate = total_wins / total_trades if total_trades > 0 else 0
        
        # 有交易的参数组合
        has_trades = [r for r in self.results if r.total_trades > 0]
        avg_win_rate = np.mean([r.win_rate for r in has_trades]) if has_trades else 0
        
        # 最优参数
        best = self.get_best_params(min_trades=1)
        
        report = f"""
{'='*70}
股指期权布林带均值回归策略 - 参数优化报告
{'='*70}

【整体统计】
总参数组合数: {len(self.results)}
有交易的参数组合: {len(has_trades)}
所有参数组合总交易次数: {total_trades}
整体胜率: {overall_win_rate*100:.2f}%
平均胜率: {avg_win_rate*100:.2f}%

【最优参数 Top {top_n}】
"""
        
        for i, r in enumerate(best[:top_n], 1):
            p = r.params
            report += f"""
排名 {i}:
  周期: {p.timeframe}, 资金: {p.capital}元, 虚值: {p.strike_otm}档
  MACD: fast={p.macd_fast}, slow={p.macd_slow}, signal={p.macd_signal}
  BOLL: period={p.boll_period}, std={p.boll_std}
  平仓: 中轨={p.exit_on_middle}, 反向交叉={p.exit_on_reverse}
  
  回测结果:
    交易次数: {r.total_trades}
    胜率: {r.win_rate*100:.2f}%
    盈亏比: {r.win_loss_ratio:.2f}
    总收益率: {r.total_return*100:.2f}%
    最大回撤: {r.max_drawdown*100:.2f}%
    夏普比率: {r.sharpe_ratio:.2f}
"""
        
        report += f"\n{'='*70}\n"
        return report


# ==================== 主函数 ====================

def main():
    """主函数 - 参数优化示例"""
    print("="*70)
    print("股指期权布林带均值回归策略 - 参数优化")
    print("="*70)
    
    # 加载数据（示例，实际应从数据源获取）
    try:
        df = pd.read_csv('/root/.openclaw/workspace/zz500_etf_5min.csv')
        # 标准化列名
        df = df.rename(columns={
            '时间': 'datetime',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover'
        })
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"\n数据加载成功: {len(df)} 条")
        print(f"时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 创建优化器
    optimizer = ParameterOptimizer(df)
    
    # 执行参数优化
    print("\n开始参数优化...")
    results = optimizer.optimize(
        timeframes=['5min', '15min'],  # 先测试5分钟和15分钟
        capitals=[5000, 10000],        # 测试5000和10000元
        otm_levels=[2, 3]              # 测试虚值2档和3档
    )
    
    # 生成报告
    print("\n" + optimizer.generate_report(top_n=10))
    
    # 保存结果
    optimizer.save_results('/root/.openclaw/workspace/optimization_results.json')
    
    # 保存最优参数
    best_params = optimizer.get_best_params(min_trades=5)
    if best_params:
        with open('/root/.openclaw/workspace/best_params.json', 'w', encoding='utf-8') as f:
            json.dump(best_params[0].to_dict(), f, ensure_ascii=False, indent=2)
        print("✓ 最优参数已保存至: best_params.json")


if __name__ == "__main__":
    main()
