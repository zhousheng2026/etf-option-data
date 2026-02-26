#!/usr/bin/env python3
"""
中证500ETF MACD + BOLL 通道极值法策略回测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("中证500ETF MACD + BOLL 通道极值法策略回测")
print("=" * 70)

# ==================== 1. 加载数据 ====================
print("\n【1】加载数据")
print("-" * 70)

df = pd.read_csv('/root/.openclaw/workspace/zz500_etf_5min.csv')
print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")

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
df = df.sort_values('datetime').reset_index(drop=True)

print(f"\n数据时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
print(f"数据条数: {len(df)}")
print(f"\n前5行数据:")
print(df.head())

# ==================== 2. 技术指标计算函数 ====================
print("\n【2】技术指标计算")
print("-" * 70)

def calculate_macd(df, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # MACD金叉/死叉
    df['macd_golden_cross'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_dead_cross'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    
    return df

def calculate_bollinger(df, period=20, std_dev=2):
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
    
    # 突破上轨/下轨
    df['break_upper'] = df['close'] > df['boll_upper']
    df['break_lower'] = df['close'] < df['boll_lower']
    
    return df

# 测试默认参数
print("计算MACD和BOLL指标（默认参数）...")
df = calculate_macd(df, fast=12, slow=26, signal=9)
df = calculate_bollinger(df, period=20, std_dev=2)

print(f"\nMACD指标:")
print(f"  - MACD均值: {df['macd'].mean():.4f}")
print(f"  - 金叉次数: {df['macd_golden_cross'].sum()}")
print(f"  - 死叉次数: {df['macd_dead_cross'].sum()}")

print(f"\nBOLL指标:")
print(f"  - 上轨均值: {df['boll_upper'].mean():.4f}")
print(f"  - 下轨均值: {df['boll_lower'].mean():.4f}")
print(f"  - 触及上轨次数: {df['touch_upper'].sum()}")
print(f"  - 触及下轨次数: {df['touch_lower'].sum()}")

# ==================== 3. 策略实现 ====================
print("\n【3】MACD + BOLL 通道极值法策略")
print("-" * 70)

def backtest_strategy(df, params, initial_capital=100000, commission=0.0003):
    """
    MACD + BOLL 通道极值法策略回测
    
    策略逻辑：
    - 做多信号: MACD金叉 + 价格触及/突破BOLL下轨（超卖极值）
    - 做空信号: MACD死叉 + 价格触及/突破BOLL上轨（超买极值）
    - 平仓条件: MACD反向交叉 或 价格回归BOLL中轨
    
    params: {
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'boll_period': 20,
        'boll_std': 2,
        'entry_threshold': 0.0,  # 触及布林带极值的阈值
        'exit_on_middle': True,  # 是否在中轨平仓
        'exit_on_reverse': True  # 是否在反向MACD交叉平仓
    }
    """
    # 复制数据并计算指标
    data = df.copy()
    data = calculate_macd(data, 
                          fast=params['macd_fast'], 
                          slow=params['macd_slow'], 
                          signal=params['macd_signal'])
    data = calculate_bollinger(data, 
                               period=params['boll_period'], 
                               std_dev=params['boll_std'])
    
    # 初始化
    position = 0  # 0: 空仓, 1: 多头, -1: 空头
    capital = initial_capital
    trades = []
    equity_curve = [initial_capital]
    
    for i in range(1, len(data)):
        row = data.iloc[i]
        prev_row = data.iloc[i-1]
        
        # 跳过NaN值
        if pd.isna(row['macd']) or pd.isna(row['boll_mid']):
            equity_curve.append(capital)
            continue
        
        # 生成交易信号
        long_signal = False
        short_signal = False
        close_signal = False
        
        # 做多信号: MACD金叉 + 价格触及下轨
        if row['macd_golden_cross'] and row['touch_lower']:
            long_signal = True
        
        # 做空信号: MACD死叉 + 价格触及上轨
        if row['macd_dead_cross'] and row['touch_upper']:
            short_signal = True
        
        # 平仓信号
        if position == 1:  # 多头持仓
            # MACD死叉平仓
            if params['exit_on_reverse'] and row['macd_dead_cross']:
                close_signal = True
            # 价格突破中轨向上平仓（获利了结）
            elif params['exit_on_middle'] and prev_row['close'] < prev_row['boll_mid'] and row['close'] > row['boll_mid']:
                close_signal = True
                
        elif position == -1:  # 空头持仓
            # MACD金叉平仓
            if params['exit_on_reverse'] and row['macd_golden_cross']:
                close_signal = True
            # 价格跌破中轨向下平仓（获利了结）
            elif params['exit_on_middle'] and prev_row['close'] > prev_row['boll_mid'] and row['close'] < row['boll_mid']:
                close_signal = True
        
        # 执行交易
        if position == 0:
            if long_signal:
                position = 1
                entry_price = row['close']
                entry_time = row['datetime']
                shares = capital / entry_price
            elif short_signal:
                position = -1
                entry_price = row['close']
                entry_time = row['datetime']
                shares = capital / entry_price
        else:
            if close_signal:
                exit_price = row['close']
                exit_time = row['datetime']
                
                # 计算盈亏
                if position == 1:
                    pnl = (exit_price - entry_price) * shares
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - exit_price) * shares
                    pnl_pct = (entry_price - exit_price) / entry_price
                
                # 扣除手续费
                commission_cost = (entry_price + exit_price) * shares * commission
                pnl -= commission_cost
                
                capital += pnl
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'capital': capital
                })
                
                position = 0
                shares = 0
        
        equity_curve.append(capital)
    
    # 如果还有持仓，强制平仓
    if position != 0:
        exit_price = data.iloc[-1]['close']
        exit_time = data.iloc[-1]['datetime']
        
        if position == 1:
            pnl = (exit_price - entry_price) * shares
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) * shares
            pnl_pct = (entry_price - exit_price) / entry_price
        
        commission_cost = (entry_price + exit_price) * shares * commission
        pnl -= commission_cost
        capital += pnl
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'position': position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'capital': capital
        })
    
    return trades, equity_curve, data

# ==================== 4. 参数优化 ====================
print("\n【4】参数优化")
print("-" * 70)

# 定义参数搜索空间
param_grid = {
    'macd_fast': [8, 12, 16],
    'macd_slow': [21, 26, 30],
    'macd_signal': [7, 9, 12],
    'boll_period': [15, 20, 25],
    'boll_std': [1.5, 2.0, 2.5],
    'exit_on_middle': [True, False],
    'exit_on_reverse': [True]
}

# 生成所有参数组合
param_combinations = []
keys = list(param_grid.keys())
values = list(param_grid.values())
for combo in itertools.product(*values):
    param_dict = dict(zip(keys, combo))
    # 确保fast < slow
    if param_dict['macd_fast'] < param_dict['macd_slow']:
        param_combinations.append(param_dict)

print(f"参数组合总数: {len(param_combinations)}")
print(f"\n参数搜索空间:")
for key, val in param_grid.items():
    print(f"  {key}: {val}")

# 执行参数优化
print("\n开始参数优化...")
results = []

for i, params in enumerate(param_combinations):
    try:
        trades, equity_curve, _ = backtest_strategy(df, params)
        
        if len(trades) == 0:
            continue
        
        trades_df = pd.DataFrame(trades)
        
        # 计算回测指标
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
        total_return = (final_capital - 100000) / 100000
        
        # 计算最大回撤
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 计算夏普比率（简化版）
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 48) if returns.std() > 0 else 0  # 年化，假设每天48个5分钟
        
        results.append({
            'params': params,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'win_loss_ratio': win_loss_ratio,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_capital': final_capital,
            'trades': trades,
            'equity_curve': equity_curve
        })
        
        if (i + 1) % 50 == 0:
            print(f"  已完成 {i+1}/{len(param_combinations)} 组参数测试")
            
    except Exception as e:
        continue

print(f"\n✓ 参数优化完成，有效结果: {len(results)} 组")

# 转换为DataFrame
results_df = pd.DataFrame(results)

# ==================== 5. 结果分析 ====================
print("\n【5】回测结果分析")
print("-" * 70)

# 按胜率排序
results_by_winrate = results_df.sort_values('win_rate', ascending=False)

print("\n【胜率最高的Top 10参数组合】")
print("=" * 100)

top10 = results_by_winrate.head(10)
display_cols = ['total_trades', 'win_rate', 'profit_factor', 'win_loss_ratio', 
                'total_return', 'max_drawdown', 'sharpe_ratio']

for idx, row in top10.iterrows():
    print(f"\n排名 {list(top10.index).index(idx) + 1}:")
    print(f"  参数: MACD({row['params']['macd_fast']},{row['params']['macd_slow']},{row['params']['macd_signal']}), "
          f"BOLL({row['params']['boll_period']},{row['params']['boll_std']}), "
          f"中轨平仓={row['params']['exit_on_middle']}")
    print(f"  交易次数: {row['total_trades']}, 胜率: {row['win_rate']:.2%}, "
          f"盈亏比: {row['win_loss_ratio']:.2f}")
    print(f"  总收益率: {row['total_return']:.2%}, 最大回撤: {row['max_drawdown']:.2%}, "
          f"夏普比率: {row['sharpe_ratio']:.2f}")

# 保存详细结果
results_df.to_pickle('/root/.openclaw/workspace/backtest_results.pkl')
print("\n✓ 详细结果已保存至: backtest_results.pkl")

# 保存Top 10参数
with open('/root/.openclaw/workspace/top10_params.txt', 'w') as f:
    f.write("中证500ETF MACD+BOLL策略 - 胜率Top 10参数组合\n")
    f.write("=" * 80 + "\n\n")
    for idx, row in top10.iterrows():
        f.write(f"排名 {list(top10.index).index(idx) + 1}:\n")
        f.write(f"  MACD参数: fast={row['params']['macd_fast']}, slow={row['params']['macd_slow']}, signal={row['params']['macd_signal']}\n")
        f.write(f"  BOLL参数: period={row['params']['boll_period']}, std={row['params']['boll_std']}\n")
        f.write(f"  平仓设置: 中轨平仓={row['params']['exit_on_middle']}, 反向交叉平仓={row['params']['exit_on_reverse']}\n")
        f.write(f"  回测结果: 交易{row['total_trades']}次, 胜率{row['win_rate']:.2%}, 盈亏比{row['win_loss_ratio']:.2f}\n")
        f.write(f"  收益率{row['total_return']:.2%}, 最大回撤{row['max_drawdown']:.2%}, 夏普{row['sharpe_ratio']:.2f}\n\n")

print("✓ Top 10参数已保存至: top10_params.txt")

# ==================== 6. 最优参数详细回测 ====================
print("\n【6】最优参数详细回测")
print("-" * 70)

best_params = top10.iloc[0]['params']
print(f"\n最优参数:")
print(f"  MACD: fast={best_params['macd_fast']}, slow={best_params['macd_slow']}, signal={best_params['macd_signal']}")
print(f"  BOLL: period={best_params['boll_period']}, std={best_params['boll_std']}")
print(f"  平仓条件: 中轨平仓={best_params['exit_on_middle']}, 反向交叉平仓={best_params['exit_on_reverse']}")

trades, equity_curve, data_with_indicators = backtest_strategy(df, best_params)
trades_df = pd.DataFrame(trades)

print(f"\n详细交易统计:")
print(f"  总交易次数: {len(trades)}")
print(f"  盈利次数: {len(trades_df[trades_df['pnl'] > 0])}")
print(f"  亏损次数: {len(trades_df[trades_df['pnl'] <= 0])}")
print(f"  胜率: {top10.iloc[0]['win_rate']:.2%}")
print(f"  平均盈利: ¥{trades_df[trades_df['pnl'] > 0]['pnl'].mean():.2f}" if len(trades_df[trades_df['pnl'] > 0]) > 0 else "  平均盈利: N/A")
print(f"  平均亏损: ¥{trades_df[trades_df['pnl'] <= 0]['pnl'].mean():.2f}" if len(trades_df[trades_df['pnl'] <= 0]) > 0 else "  平均亏损: N/A")
print(f"  盈亏比: {top10.iloc[0]['win_loss_ratio']:.2f}")
print(f"  总收益率: {top10.iloc[0]['total_return']:.2%}")
print(f"  最大回撤: {top10.iloc[0]['max_drawdown']:.2%}")
print(f"  夏普比率: {top10.iloc[0]['sharpe_ratio']:.2f}")

print(f"\n最近5笔交易:")
print(trades_df.tail().to_string())

# 保存交易记录
trades_df.to_csv('/root/.openclaw/workspace/best_strategy_trades.csv', index=False)
print("\n✓ 最优策略交易记录已保存至: best_strategy_trades.csv")

# ==================== 7. 可视化 ====================
print("\n【7】生成可视化图表")
print("-" * 70)

# 创建图表
fig, axes = plt.subplots(4, 1, figsize=(14, 16))

# 1. 价格与布林带
ax1 = axes[0]
ax1.plot(data_with_indicators['datetime'], data_with_indicators['close'], label='Close', color='black', linewidth=0.8)
ax1.plot(data_with_indicators['datetime'], data_with_indicators['boll_upper'], label='BOLL Upper', color='red', linestyle='--', alpha=0.7)
ax1.plot(data_with_indicators['datetime'], data_with_indicators['boll_mid'], label='BOLL Mid', color='blue', linestyle='--', alpha=0.7)
ax1.plot(data_with_indicators['datetime'], data_with_indicators['boll_lower'], label='BOLL Lower', color='green', linestyle='--', alpha=0.7)
ax1.set_title('ZZ500 ETF Price with Bollinger Bands', fontsize=12)
ax1.set_ylabel('Price')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 2. MACD
ax2 = axes[1]
ax2.plot(data_with_indicators['datetime'], data_with_indicators['macd'], label='MACD', color='blue', linewidth=0.8)
ax2.plot(data_with_indicators['datetime'], data_with_indicators['macd_signal'], label='Signal', color='red', linewidth=0.8)
colors = ['green' if h >= 0 else 'red' for h in data_with_indicators['macd_hist']]
ax2.bar(data_with_indicators['datetime'], data_with_indicators['macd_hist'], label='Histogram', color=colors, alpha=0.5, width=0.001)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_title('MACD Indicator', fontsize=12)
ax2.set_ylabel('MACD')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# 3. 资金曲线
ax3 = axes[2]
ax3.plot(data_with_indicators['datetime'][:len(equity_curve)], equity_curve, label='Equity Curve', color='blue', linewidth=1)
ax3.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
ax3.fill_between(data_with_indicators['datetime'][:len(equity_curve)], 100000, equity_curve, 
                  where=[e >= 100000 for e in equity_curve], alpha=0.3, color='green')
ax3.fill_between(data_with_indicators['datetime'][:len(equity_curve)], 100000, equity_curve, 
                  where=[e < 100000 for e in equity_curve], alpha=0.3, color='red')
ax3.set_title('Strategy Equity Curve', fontsize=12)
ax3.set_ylabel('Capital (CNY)')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# 4. 回撤曲线
ax4 = axes[3]
equity_series = pd.Series(equity_curve)
rolling_max = equity_series.expanding().max()
drawdown = (equity_series - rolling_max) / rolling_max * 100
ax4.fill_between(data_with_indicators['datetime'][:len(drawdown)], drawdown, 0, color='red', alpha=0.5)
ax4.plot(data_with_indicators['datetime'][:len(drawdown)], drawdown, color='darkred', linewidth=0.8)
ax4.set_title('Drawdown (%)', fontsize=12)
ax4.set_ylabel('Drawdown (%)')
ax4.set_xlabel('Date')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/.openclaw/workspace/strategy_analysis.png', dpi=150, bbox_inches='tight')
print("✓ 分析图表已保存至: strategy_analysis.png")

# ==================== 8. 汇总报告 ====================
print("\n" + "=" * 70)
print("【策略回测汇总报告】")
print("=" * 70)

print(f"""
策略名称: MACD + BOLL 通道极值法
标的: 中证500ETF (510500) 5分钟数据
数据时间: {df['datetime'].min()} 至 {df['datetime'].max()}
数据条数: {len(df)} 条

【最优参数组合】
- MACD: fast={best_params['macd_fast']}, slow={best_params['macd_slow']}, signal={best_params['macd_signal']}
- BOLL: period={best_params['boll_period']}, std={best_params['boll_std']}
- 平仓条件: 中轨平仓={best_params['exit_on_middle']}, 反向交叉平仓={best_params['exit_on_reverse']}

【策略逻辑】
1. 做多信号: MACD金叉 + 价格触及BOLL下轨（超卖极值）
2. 做空信号: MACD死叉 + 价格触及BOLL上轨（超买极值）
3. 平仓条件: MACD反向交叉 或 价格回归BOLL中轨

【回测结果】
- 总交易次数: {top10.iloc[0]['total_trades']}
- 胜率: {top10.iloc[0]['win_rate']:.2%}
- 盈亏比: {top10.iloc[0]['win_loss_ratio']:.2f}
- 总收益率: {top10.iloc[0]['total_return']:.2%}
- 最大回撤: {top10.iloc[0]['max_drawdown']:.2%}
- 夏普比率: {top10.iloc[0]['sharpe_ratio']:.2f}

【期权数据说明】
期权5分钟数据获取受限，本策略基于ETF数据构建。
实际应用于期权交易时：
1. 可用ETF信号作为期权交易触发条件
2. 购/沽合约选择根据方向确定
3. 建议通过券商API获取实时期权数据

【输出文件】
- zz500_etf_5min.csv: 原始数据
- backtest_results.pkl: 所有参数回测结果
- top10_params.txt: 最优参数组合
- best_strategy_trades.csv: 最优策略交易记录
- strategy_analysis.png: 分析图表
""")

print("=" * 70)
print("回测完成!")
print("=" * 70)
