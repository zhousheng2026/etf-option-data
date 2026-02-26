"""
基于真实市场特征生成模拟数据回测
使用真实统计参数（波动率、自相关等）生成更真实的模拟数据
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('/root/.openclaw/workspace/skills/index-options-burst/scripts')
from backtest import IndexOptionsStrategy
import json

def generate_realistic_data(n_days: int = 500, start_date: str = "2020-01-01",
                            annual_return: float = 0.08, annual_vol: float = 0.20,
                            trend_strength: float = 0.3) -> pd.DataFrame:
    """
    生成具有真实市场特征的模拟数据
    
    Parameters:
    - n_days: 交易日数
    - annual_return: 年化收益率
    - annual_vol: 年化波动率
    - trend_strength: 趋势强度 (0-1, 越高趋势越明显)
    """
    np.random.seed(42)
    
    dates = pd.date_range(start=start_date, periods=n_days, freq='B')  # 工作日
    
    # 基础参数
    dt = 1/252  # 日时间步长
    mu = annual_return / 252  # 日收益率
    sigma = annual_vol / np.sqrt(252)  # 日波动率
    
    # 生成价格路径 (GBM + 趋势成分)
    returns = np.random.normal(mu, sigma, n_days)
    
    # 添加趋势成分（让数据有通道特征）- 更频繁的趋势变化
    trend_periods = [
        (0, 40, 0.002),       # 上涨
        (40, 70, -0.001),     # 下跌
        (70, 110, 0.0015),    # 上涨
        (110, 140, -0.001),   # 下跌
        (140, 180, 0.001),    # 上涨
        (180, 210, -0.0015),  # 下跌
        (210, 250, 0.002),    # 上涨
        (250, 280, -0.001),   # 下跌
        (280, 320, 0.001),    # 上涨
        (320, 360, -0.0015),  # 下跌
        (360, 400, 0.001),    # 上涨
        (400, 440, -0.001),   # 下跌
        (440, 500, 0.0008),   # 上涨
    ]
    
    for start, end, trend in trend_periods:
        if end <= n_days:
            returns[start:end] += trend
    
    # 添加波动率聚集 (GARCH-like)
    vols = np.ones(n_days) * sigma
    for i in range(1, n_days):
        vols[i] = np.sqrt(0.05 * sigma**2 + 0.85 * vols[i-1]**2 + 0.10 * returns[i-1]**2)
        returns[i] = np.random.normal(mu, vols[i])
    
    # 计算价格
    log_prices = np.cumsum(returns)
    closes = 4000 * np.exp(log_prices)
    
    # 生成OHLC
    daily_vol = vols * np.sqrt(252)
    highs = closes * (1 + np.abs(np.random.randn(n_days)) * daily_vol * 0.3)
    lows = closes * (1 - np.abs(np.random.randn(n_days)) * daily_vol * 0.3)
    opens = closes * (1 + np.random.randn(n_days) * daily_vol * 0.1)
    
    # 成交量 (与波动率正相关)
    base_volume = 100000
    volumes = base_volume * (1 + daily_vol * 5) * np.random.lognormal(0, 0.5, n_days)
    
    # IV (与 realized vol 相关)
    ivs = daily_vol * 1.2 + np.random.randn(n_days) * 0.02
    ivs = np.clip(ivs, 0.10, 0.60)
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'iv': ivs
    }, index=dates)
    
    return df


# 生成数据
print("生成真实市场特征模拟数据...")
df = generate_realistic_data(n_days=500, start_date="2020-01-01",
                             annual_return=0.08, annual_vol=0.20,
                             trend_strength=0.3)

print(f"数据时间范围: {df.index[0]} 至 {df.index[-1]}")
print(f"数据条数: {len(df)}")
print(f"年化收益率: {df['close'].pct_change().mean() * 252:.2%}")
print(f"年化波动率: {df['close'].pct_change().std() * np.sqrt(252):.2%}")
print(f"最大回撤: {(df['close'] / df['close'].cummax() - 1).min():.2%}")

# 运行回测
print("\n" + "="*60)
print("开始回测 - 真实市场特征模拟数据")
print("="*60)

strategy = IndexOptionsStrategy(initial_capital=1000000, channel_bars=20)
result = strategy.run_backtest(df)
strategy.print_report(result)

# 保存结果
output = {
    'data_source': 'realistic_simulation',
    'data_characteristics': {
        'annual_return': 0.08,
        'annual_vol': 0.20,
        'trend_strength': 0.3,
        'actual_annual_return': df['close'].pct_change().mean() * 252,
        'actual_annual_vol': df['close'].pct_change().std() * np.sqrt(252),
    },
    'data_range': f"{df.index[0]} to {df.index[-1]}",
    'total_bars': len(df),
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

with open('backtest_realistic_sim.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n结果已保存: backtest_realistic_sim.json")

# 对比不同波动率环境
print("\n" + "="*60)
print("不同市场环境对比")
print("="*60)

for vol in [0.15, 0.20, 0.25, 0.30]:
    df_test = generate_realistic_data(n_days=500, annual_vol=vol)
    strategy_test = IndexOptionsStrategy(initial_capital=1000000, channel_bars=20)
    result_test = strategy_test.run_backtest(df_test)
    print(f"波动率 {vol:.0%}: 胜率 {result_test.win_rate:.1%}, 收益 {result_test.total_return:.1%}, 交易 {len(result_test.trades)}次")
