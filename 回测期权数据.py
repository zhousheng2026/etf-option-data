"""
用真实期权数据跑三重底背离策略回测
"""
import pandas as pd
import numpy as np
import sys
import os

# 添加策略路径
sys.path.insert(0, '/root/.openclaw/workspace/skills/index-options-burst/scripts')
from triple_divergence_strategy import TripleDivergenceStrategy

# 读取期权数据
data_files = [
    '/root/openclaw/kimi/downloads/19c94f48-3dc2-832a-8000-0000a668b0af_10011030_科创50购9月1700.xlsx',
    '/root/openclaw/kimi/downloads/19c94f48-3c62-864a-8000-0000aec6f292_10011033_科创50购9月1850.xlsx',
    '/root/openclaw/kimi/downloads/19c94f48-3f52-834a-8000-000065a8bcfd_10011034_科创50沽9月1450.xlsx',
    '/root/openclaw/kimi/downloads/19c94f48-44a2-84a0-8000-000028c3c2d0_10011074_科创50沽9月1400.xlsx',
    '/root/openclaw/kimi/downloads/19c94f48-43a2-831b-8000-0000cd3c6593_10011100_科创50沽9月1300.xlsx',
]

print("="*70)
print("科创50ETF期权 三重底背离策略回测")
print("="*70)

all_results = []

for i, file_path in enumerate(data_files, 1):
    contract_name = os.path.basename(file_path).replace('.xlsx', '')
    print(f"\n{'='*70}")
    print(f"【{i}/5】回测合约: {contract_name}")
    print(f"{'='*70}")
    
    # 读取数据
    df = pd.read_excel(file_path)
    print(f"数据条数: {len(df)}")
    print(f"数据范围: {df['日期时间'].min()} 至 {df['日期时间'].max()}")
    
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
        '成交额': 'amount',
        '涨跌幅': 'change_pct'
    }, inplace=True)
    
    # 添加模拟IV数据
    df['iv'] = 0.25 + np.random.randn(len(df)) * 0.03
    df['iv'] = df['iv'].clip(0.15, 0.35)
    
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
    
    # 保存结果
    all_results.append({
        'contract': contract_name,
        'total_trades': len(result.trades),
        'win_rate': result.win_rate,
        'profit_factor': result.profit_factor,
        'total_return': result.total_return,
        'max_drawdown': result.max_drawdown,
    })
    
    # 打印简要结果
    print(f"\n回测结果:")
    print(f"  交易次数: {len(result.trades)}")
    print(f"  胜率: {result.win_rate:.2%}")
    print(f"  盈亏比: {result.profit_factor:.2f}")
    print(f"  总收益率: {result.total_return:.2%}")
    print(f"  最大回撤: {result.max_drawdown:.2%}")

# 汇总结果
print(f"\n{'='*70}")
print("汇总结果")
print(f"{'='*70}")

summary_df = pd.DataFrame(all_results)
print(summary_df.to_string(index=False))

# 保存汇总
summary_file = '/root/.openclaw/workspace/期权回测汇总结果.xlsx'
summary_df.to_excel(summary_file, index=False)
print(f"\n汇总结果已保存: {summary_file}")
