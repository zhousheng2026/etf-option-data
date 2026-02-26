"""
用AKShare获取真实历史数据回测 - 备用方案
"""
import akshare as ak
import pandas as pd
import numpy as np
import sys
sys.path.append('/root/.openclaw/workspace/skills/index-options-burst/scripts')
from backtest import IndexOptionsStrategy
import json
import time

print("尝试获取数据...")

# 尝试获取ETF日线数据（更稳定）
try:
    print("获取沪深300ETF(510300)日线数据...")
    time.sleep(1)
    
    # 使用stock_zh_a_hist获取ETF数据
    df = ak.stock_zh_a_hist(symbol="510300", period="daily", 
                            start_date="20230101", end_date="20250225",
                            adjust="qfq")
    
    print(f"成功获取数据: {len(df)} 条")
    print(df.head(3))
    
    # 处理数据
    df['日期'] = pd.to_datetime(df['日期'])
    df.set_index('日期', inplace=True)
    df.columns = ['open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'change_pct', 'change', 'turnover']
    
    # 添加模拟IV
    df['iv'] = 0.15 + (df['amplitude'] / 100) * 0.3
    df['iv'] = df['iv'].clip(0.10, 0.50)
    
    print(f"\n数据时间范围: {df.index[0]} 至 {df.index[-1]}")
    print(f"数据条数: {len(df)}")
    
    # 运行回测
    print("\n" + "="*60)
    print("开始回测 - 沪深300ETF(510300) 日线真实数据")
    print("="*60)
    
    strategy = IndexOptionsStrategy(initial_capital=1000000, channel_bars=20)
    result = strategy.run_backtest(df)
    strategy.print_report(result)
    
    # 保存结果
    output = {
        'data_source': 'akshare - 沪深300ETF(510300)',
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
    
    with open('backtest_hs300_etf_daily.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n结果已保存: backtest_hs300_etf_daily.json")
    
except Exception as e:
    print(f"获取数据失败: {e}")
    import traceback
    traceback.print_exc()
