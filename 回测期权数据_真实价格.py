"""
用真实期权数据跑三重底背离策略回测（修复版）
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, '/root/.openclaw/workspace')
from triple_divergence_real import TripleDivergenceStrategyReal

# 读取期权数据
data_files = [
    ('/root/openclaw/kimi/downloads/19c94f48-3dc2-832a-8000-0000a668b0af_10011030_科创50购9月1700.xlsx', '科创50购9月1700'),
    ('/root/openclaw/kimi/downloads/19c94f48-3c62-864a-8000-0000aec6f292_10011033_科创50购9月1850.xlsx', '科创50购9月1850'),
    ('/root/openclaw/kimi/downloads/19c94f48-3f52-834a-8000-000065a8bcfd_10011034_科创50沽9月1450.xlsx', '科创50沽9月1450'),
    ('/root/openclaw/kimi/downloads/19c94f48-44a2-84a0-8000-000028c3c2d0_10011074_科创50沽9月1400.xlsx', '科创50沽9月1400'),
    ('/root/openclaw/kimi/downloads/19c94f48-43a2-831b-8000-0000cd3c6593_10011100_科创50沽9月1300.xlsx', '科创50沽9月1300'),
]

print("="*70)
print("科创50ETF期权 三重底背离策略回测（真实价格版）")
print("="*70)

all_results = []

for i, (file_path, contract_name) in enumerate(data_files, 1):
    print(f"\n{'='*70}")
    print(f"【{i}/5】回测合约: {contract_name}")
    print(f"{'='*70}")
    
    # 读取数据
    df = pd.read_excel(file_path)
    print(f"数据条数: {len(df)}")
    print(f"数据范围: {df['日期时间'].min()} 至 {df['日期时间'].max()}")
    print(f"价格范围: {df['收盘价'].min():.4f} - {df['收盘价'].max():.4f}")
    
    # 转换日期为索引
    df['日期时间'] = pd.to_datetime(df['日期时间'])
    df.set_index('日期时间', inplace=True)
    
    # 重命名列
    df.rename(columns={
        '开盘价': 'open',
        '收盘价': 'close',
        '最高价': 'high',
        '最低价': 'low',
        '成交量': 'volume',
    }, inplace=True)
    
    # 运行回测
    strategy = TripleDivergenceStrategyReal(
        initial_capital=1000000,
        bb_period=20,
        bb_std=2.0,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        divergence_lookback=30,
        min_divergence_bars=5,
        stop_loss_pct=0.30,
        cooldown_bars=5
    )
    
    result = strategy.run_backtest(df)
    strategy.print_report(result)
    
    # 保存结果
    all_results.append({
        '合约': contract_name,
        '交易次数': len(result['trades']),
        '胜率': f"{result['win_rate']:.2%}",
        '盈亏比': f"{result['profit_factor']:.2f}",
        '总收益率': f"{result['total_return']:.2%}",
        '最大回撤': f"{result['max_drawdown']:.2%}",
        '初始资金': result['initial_capital'],
        '最终资金': result['final_capital'],
        '盈亏金额': result['final_capital'] - result['initial_capital'],
    })

# 汇总结果
print(f"\n{'='*70}")
print("汇总结果")
print(f"{'='*70}")

summary_df = pd.DataFrame(all_results)
print(summary_df.to_string(index=False))

# 保存汇总
summary_file = '/root/.openclaw/workspace/期权回测汇总结果_真实价格.xlsx'
summary_df.to_excel(summary_file, index=False)
print(f"\n汇总结果已保存: {summary_file}")
