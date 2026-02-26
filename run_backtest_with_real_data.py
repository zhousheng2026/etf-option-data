"""
用真实数据跑三重底背离策略回测
"""
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# 添加策略路径
sys.path.insert(0, '/root/.openclaw/workspace/skills/index-options-burst/scripts')
from triple_divergence_strategy import TripleDivergenceStrategy

# 读取数据
df = pd.read_excel('/root/openclaw/kimi/downloads/19c94cbe-29c2-8f92-8000-00007057f202_科创50ETF数据.xlsx')

print("="*60)
print("科创50ETF 三重底背离策略回测")
print("="*60)
print(f"\n数据范围: {df['日期'].min()} 至 {df['日期'].max()}")
print(f"数据条数: {len(df)}")

# 转换日期为索引
df['日期'] = pd.to_datetime(df['日期'])
df.set_index('日期', inplace=True)

# 重命名列以匹配策略要求
df.rename(columns={
    '开盘价': 'open',
    '收盘价': 'close',
    '最高价': 'high',
    '最低价': 'low',
    '成交量': 'volume',
    '成交额': 'amount',
    '涨跌幅': 'change_pct'
}, inplace=True)

# 添加模拟的IV数据（实际应该用期权隐含波动率）
df['iv'] = 0.25 + np.random.randn(len(df)) * 0.05
df['iv'] = df['iv'].clip(0.15, 0.40)

print(f"\n价格范围: {df['close'].min():.3f} - {df['close'].max():.3f}")
print(f"平均成交量: {df['volume'].mean():.0f}")

# 运行回测
strategy = TripleDivergenceStrategy(
    initial_capital=1000000,
    bb_period=20,
    bb_std=2.0,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    divergence_lookback=30,
    min_divergence_bars=5,
    stop_loss_1=0.30,
    stop_loss_2=0.50,
    cooldown_bars=5,
    contract_multiplier=10000
)

result = strategy.run_backtest(df)
strategy.print_report(result)

# 保存结果
import json
output = {
    'params': {
        'symbol': '588000',
        'bb_period': 20,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
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
    },
    'trades': [
        {
            'entry_time': t.entry_time.isoformat() if hasattr(t.entry_time, 'isoformat') else str(t.entry_time),
            'exit_time': t.exit_time.isoformat() if hasattr(t.exit_time, 'isoformat') else str(t.exit_time),
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

output_file = '/root/.openclaw/workspace/科创50ETF_三重底背离回测结果.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n详细结果已保存: {output_file}")
