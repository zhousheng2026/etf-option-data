#!/usr/bin/env python3
"""
SigmaBurst策略 - 完整回测（模拟数据）
模拟图中500ETF期权爆发走势
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
sys.path.append('/root/.openclaw/workspace/scripts')

from SigmaBurst_strategy import SigmaBurstStrategy

def generate_burst_data():
    """生成爆发式走势数据（类似图中黄线）"""
    data = []
    base_time = datetime(2026, 1, 14, 9, 30)
    
    # 阶段1：横盘整理（中轨附近，缩量）- 20根K线
    for i in range(20):
        price = 0.08 + np.random.normal(0, 0.003)
        data.append({
            'datetime': base_time + timedelta(minutes=30*i),
            'open': max(0.01, price - 0.002),
            'high': max(0.01, price + 0.005),
            'low': max(0.01, price - 0.005),
            'close': max(0.01, price),
            'volume': np.random.normal(800, 200)
        })
    
    # 阶段2：突破上涨 - 10根K线
    base_time = data[-1]['datetime'] + timedelta(minutes=30)
    for i in range(10):
        progress = i / 10
        price = 0.08 + (0.15 - 0.08) * progress + np.random.normal(0, 0.005)
        data.append({
            'datetime': base_time + timedelta(minutes=30*i),
            'open': max(0.01, price - 0.003),
            'high': max(0.01, price + 0.008),
            'low': max(0.01, price - 0.005),
            'close': max(0.01, price),
            'volume': np.random.normal(2000, 300)
        })
    
    # 阶段3：加速爆发（6σ→12σ+）- 20根K线
    base_time = data[-1]['datetime'] + timedelta(minutes=30)
    for i in range(20):
        progress = i / 20
        # 非线性加速
        price = 0.15 + (0.29 - 0.15) * (progress ** 0.6) + np.random.normal(0, 0.008)
        data.append({
            'datetime': base_time + timedelta(minutes=30*i),
            'open': max(0.01, min(0.30, price - 0.005)),
            'high': max(0.01, min(0.30, price + 0.012)),
            'low': max(0.01, min(0.30, price - 0.008)),
            'close': max(0.01, min(0.30, price)),
            'volume': np.random.normal(3500, 500)
        })
    
    # 阶段4：高位回落 - 10根K线
    base_time = data[-1]['datetime'] + timedelta(minutes=30)
    for i in range(10):
        progress = i / 10
        price = 0.29 - (0.29 - 0.20) * progress + np.random.normal(0, 0.005)
        data.append({
            'datetime': base_time + timedelta(minutes=30*i),
            'open': max(0.01, price + 0.003),
            'high': max(0.01, price + 0.008),
            'low': max(0.01, price - 0.008),
            'close': max(0.01, price),
            'volume': np.random.normal(2800, 400)
        })
    
    df = pd.DataFrame(data)
    return df

def main():
    print("="*80)
    print("SigmaBurst策略 - 完整回测")
    print("="*80)
    
    # 生成模拟数据
    print("\n生成模拟爆发数据...")
    df = generate_burst_data()
    print(f"数据量: {len(df)} 条")
    print(f"价格范围: {df['close'].min():.4f} - {df['close'].max():.4f}")
    print(f"理论涨幅: {(df['close'].max()/df['close'].min()-1)*100:.1f}%")
    
    # 运行回测
    print("\n" + "-"*80)
    strategy = SigmaBurstStrategy(initial_capital=100000)
    result = strategy.backtest(df, verbose=True)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_file = f'backtest_results/SigmaBurst_Final_{timestamp}.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n完整结果: {result_file}")
    
    # 策略总结
    print("\n" + "="*80)
    print("策略总结")
    print("="*80)
    print("""
SigmaBurst（西格玛爆发）期权策略

买入逻辑：
  1. 中轨企稳（30%）- 价格在中轨附近+缩量+企稳
  2. 突破前高（40%）- 突破20周期高点+放量
  3. 放量确认（30%）- 已有仓位后放量上涨

卖出逻辑（激进止盈）：
  6σ 减仓20%
  8σ 减仓30%
  10σ 减仓30%
  12σ+ 清仓90%（留10%博更大收益）

止损：
  - 跌破中轨且亏损>20%
  - 权利金亏损>50%

适用：ETF期权30分钟K线
预期：捕获爆发式行情，盈亏比 3:1 以上
    """)

if __name__ == "__main__":
    main()
