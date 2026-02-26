"""
用AKShare获取真实历史数据回测
"""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('/root/.openclaw/workspace/skills/index-options-burst/scripts')
from backtest import IndexOptionsStrategy, Direction
import json

print("正在获取沪深300指数历史数据...")

# 获取沪深300指数日线数据
try:
    df_daily = ak.index_zh_a_hist(symbol="000300", period="daily", 
                                   start_date="20230101", end_date="20250225")
    print(f"获取到日线数据: {len(df_daily)} 条")
    print(df_daily.head())
except Exception as e:
    print(f"获取日线数据失败: {e}")
    df_daily = None

if df_daily is not None and len(df_daily) > 0:
    # 处理数据格式
    df_daily['日期'] = pd.to_datetime(df_daily['日期'])
    df_daily.set_index('日期', inplace=True)
    df_daily.columns = ['open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'change_pct', 'change', 'turnover']
    
    # 添加模拟IV (实际应从期权数据获取)
    # 用振幅和波动率估算
    df_daily['iv'] = 0.15 + (df_daily['amplitude'] / 100) * 0.3
    df_daily['iv'] = df_daily['iv'].clip(0.10, 0.50)
    
    print(f"\n数据时间范围: {df_daily.index[0]} 至 {df_daily.index[-1]}")
    print(f"数据条数: {len(df_daily)}")
    
    # 运行回测
    print("\n" + "="*60)
    print("开始回测 - 沪深300指数日线真实数据")
    print("="*60)
    
    strategy = IndexOptionsStrategy(initial_capital=1000000, channel_bars=20)
    result = strategy.run_backtest(df_daily)
    strategy.print_report(result)
    
    # 保存结果
    output = {
        'data_source': 'akshare - 沪深300指数(000300)',
        'data_range': f"{df_daily.index[0]} to {df_daily.index[-1]}",
        'total_bars': len(df_daily),
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
    
    with open('backtest_hs300_akshare.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n结果已保存: backtest_hs300_akshare.json")

# 尝试获取30分钟数据
print("\n" + "="*60)
print("尝试获取30分钟数据...")
print("="*60)

try:
    # 获取ETF的30分钟数据（指数本身可能没有30分钟数据）
    df_30min = ak.fund_etf_hist_em(symbol="510300", period="30", 
                                    start_date="20240101", end_date="20250225",
                                    adjust="qfq")
    print(f"获取到30分钟数据: {len(df_30min)} 条")
    
    if len(df_30min) > 0:
        df_30min['日期'] = pd.to_datetime(df_30min['日期'])
        df_30min.set_index('日期', inplace=True)
        df_30min.columns = ['open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'change_pct', 'change', 'turnover']
        
        # 添加模拟IV
        df_30min['iv'] = 0.15 + (df_30min['amplitude'] / 100) * 0.3
        df_30min['iv'] = df_30min['iv'].clip(0.10, 0.50)
        
        print(f"\n30分钟数据时间范围: {df_30min.index[0]} 至 {df_30min.index[-1]}")
        
        print("\n" + "="*60)
        print("开始回测 - 沪深300ETF(510300) 30分钟真实数据")
        print("="*60)
        
        strategy_30 = IndexOptionsStrategy(initial_capital=1000000, channel_bars=20)
        result_30 = strategy_30.run_backtest(df_30min)
        strategy_30.print_report(result_30)
        
        output_30 = {
            'data_source': 'akshare - 沪深300ETF(510300) 30分钟',
            'data_range': f"{df_30min.index[0]} to {df_30min.index[-1]}",
            'total_bars': len(df_30min),
            'result': {
                'total_return': result_30.total_return,
                'annual_return': result_30.annual_return,
                'max_drawdown': result_30.max_drawdown,
                'sharpe_ratio': result_30.sharpe_ratio,
                'win_rate': result_30.win_rate,
                'profit_factor': result_30.profit_factor,
                'total_trades': len(result_30.trades),
            }
        }
        
        with open('backtest_hs300_30min_akshare.json', 'w') as f:
            json.dump(output_30, f, indent=2, default=str)
            
except Exception as e:
    print(f"获取30分钟数据失败: {e}")
