"""
用真实数据回测 - 中证500ETF 5分钟数据转30分钟
"""
import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.append('/root/.openclaw/workspace/skills/index-options-burst/scripts')
from backtest import IndexOptionsStrategy, Direction

# 读取5分钟数据
df = pd.read_csv('/root/.openclaw/workspace/zz500_etf_5min.csv')
df['时间'] = pd.to_datetime(df['时间'])
df.set_index('时间', inplace=True)
df.columns = ['open', 'close', 'high', 'low', 'change_pct', 'change', 'volume', 'amount', 'amplitude', 'turnover']

print(f"原始5分钟数据: {len(df)} 根K线")
print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")

# 转换为30分钟线
df_30min = df.resample('30min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'amount': 'sum'
}).dropna()

print(f"\n30分钟线: {len(df_30min)} 根K线")

# 添加模拟IV (实际应从期权数据获取)
# 用价格波动估算IV
df_30min['returns'] = df_30min['close'].pct_change()
df_30min['volatility'] = df_30min['returns'].rolling(window=20).std() * np.sqrt(48)  # 年化
df_30min['iv'] = df_30min['volatility'].fillna(0.20) + 0.05  # 期权IV通常比历史波动率高
df_30min['iv'] = df_30min['iv'].clip(0.10, 0.50)

# 运行回测
print("\n" + "="*60)
print("开始回测 - 真实数据(中证500ETF 30分钟)")
print("="*60)

strategy = IndexOptionsStrategy(initial_capital=1000000)
result = strategy.run_backtest(df_30min)
strategy.print_report(result)

# 保存结果
import json
output = {
    'data_source': 'zz500_etf_5min.csv',
    'data_range': f"{df.index[0]} to {df.index[-1]}",
    'bars_5min': len(df),
    'bars_30min': len(df_30min),
    'params': {
        'initial_capital': 1000000,
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
        'avg_win': result.avg_win,
        'avg_loss': result.avg_loss
    }
}

with open('backtest_zz500_real_data.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n结果已保存: backtest_zz500_real_data.json")
