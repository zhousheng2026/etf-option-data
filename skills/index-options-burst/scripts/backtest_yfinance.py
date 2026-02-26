"""
用yfinance获取标普500数据回测（美股期权对标）
"""
import yfinance as yf
import pandas as pd
import numpy as np
import sys
sys.path.append('/root/.openclaw/workspace/skills/index-options-burst/scripts')
from backtest import IndexOptionsStrategy
import json

print("获取标普500指数(SPX)数据...")

# 获取SPX数据（用SPY ETF代替，因为SPX本身不是可交易标的）
try:
    spy = yf.Ticker("SPY")
    df = spy.history(start="2023-01-01", end="2025-02-25", interval="1d")
    
    print(f"成功获取数据: {len(df)} 条")
    print(df.head(3))
    
    # 处理数据格式
    df.columns = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    # 计算振幅
    df['amplitude'] = ((df['high'] - df['low']) / df['close'] * 100).round(2)
    
    # 添加模拟IV (美股期权IV通常更高)
    df['iv'] = 0.15 + (df['amplitude'] / 100) * 0.4
    df['iv'] = df['iv'].clip(0.10, 0.60)
    
    print(f"\n数据时间范围: {df.index[0]} 至 {df.index[-1]}")
    print(f"数据条数: {len(df)}")
    
    # 运行回测
    print("\n" + "="*60)
    print("开始回测 - 标普500ETF(SPY) 日线真实数据")
    print("="*60)
    
    strategy = IndexOptionsStrategy(initial_capital=1000000, channel_bars=20)
    result = strategy.run_backtest(df)
    strategy.print_report(result)
    
    # 保存结果
    output = {
        'data_source': 'yfinance - 标普500ETF(SPY)',
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
    
    with open('backtest_spy_daily.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n结果已保存: backtest_spy_daily.json")
    
except Exception as e:
    print(f"获取数据失败: {e}")
    import traceback
    traceback.print_exc()
