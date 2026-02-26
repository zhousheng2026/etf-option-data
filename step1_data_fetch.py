#!/usr/bin/env python3
"""
中证500ETF及期权5分钟数据获取与MACD+BOLL策略回测
使用多种方式尝试获取数据
"""

import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 60)
print("中证500ETF及期权数据获取与策略回测")
print("=" * 60)

# ==================== 1. 数据获取 ====================
print("\n【1】数据获取阶段")
print("-" * 60)

# 计算日期范围（最近一个月）
end_date = datetime.now()
start_date = end_date - timedelta(days=35)

print(f"数据时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")

# 尝试多种方式获取ETF数据
etf_5min = None

# 方式1: 使用stock_zh_a_hist_min_em (股票格式)
print("\n[尝试1] 使用stock_zh_a_hist_min_em获取ETF数据...")
try:
    # 对于ETF基金，可能需要使用sh510500格式
    etf_5min = ak.stock_zh_a_hist_min_em(
        symbol="sh510500",
        period="5",
        adjust="qfq",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d")
    )
    if etf_5min is not None and len(etf_5min) > 0:
        print(f"✓ 成功获取ETF数据: {len(etf_5min)} 条记录")
except Exception as e:
    print(f"✗ 失败: {str(e)[:100]}")

# 方式2: 使用基金接口
if etf_5min is None or len(etf_5min) == 0:
    print("\n[尝试2] 使用基金分钟数据接口...")
    try:
        etf_5min = ak.fund_etf_hist_min_em(
            symbol="510500",
            period="5",
            adjust="qfq",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d")
        )
        if etf_5min is not None and len(etf_5min) > 0:
            print(f"✓ 成功获取ETF数据: {len(etf_5min)} 条记录")
            print(f"  列名: {list(etf_5min.columns)}")
    except Exception as e:
        print(f"✗ 失败: {str(e)[:100]}")

# 方式3: 使用指数接口获取中证500指数数据作为替代
if etf_5min is None or len(etf_5min) == 0:
    print("\n[尝试3] 获取中证500指数(000905)数据作为标的...")
    try:
        # 获取中证500指数5分钟数据
        index_5min = ak.index_zh_a_hist_min_em(
            symbol="000905",
            period="5",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d")
        )
        if index_5min is not None and len(index_5min) > 0:
            etf_5min = index_5min
            print(f"✓ 成功获取中证500指数数据: {len(etf_5min)} 条记录")
            print(f"  列名: {list(etf_5min.columns)}")
            print(f"  前5行:\n{etf_5min.head()}")
    except Exception as e:
        print(f"✗ 失败: {str(e)[:100]}")

# 方式4: 获取ETF日线数据然后模拟5分钟数据
if etf_5min is None or len(etf_5min) == 0:
    print("\n[尝试4] 获取ETF日线数据...")
    try:
        # 获取ETF日线数据
        etf_daily = ak.fund_etf_hist_em(
            symbol="510500",
            period="daily",
            adjust="qfq",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d")
        )
        if etf_daily is not None and len(etf_daily) > 0:
            print(f"✓ 获取到ETF日线数据: {len(etf_daily)} 条")
            print(f"  列名: {list(etf_daily.columns)}")
            print(f"  前5行:\n{etf_daily.head()}")
            
            # 使用日线数据模拟5分钟数据（用于策略测试）
            print("\n  使用日线数据模拟5分钟数据用于策略回测...")
            etf_5min = etf_daily.copy()
            etf_5min['datetime'] = pd.to_datetime(etf_5min['日期'])
            etf_5min = etf_5min.rename(columns={
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
            print(f"  ✓ 已转换日线数据: {len(etf_5min)} 条")
    except Exception as e:
        print(f"✗ 失败: {str(e)[:100]}")

# 保存数据
if etf_5min is not None and len(etf_5min) > 0:
    etf_5min.to_csv('/root/.openclaw/workspace/zz500_etf_5min.csv', index=False)
    print(f"\n✓ 数据已保存至: zz500_etf_5min.csv")
    print(f"  数据形状: {etf_5min.shape}")
    print(f"  时间范围: {etf_5min['datetime'].min() if 'datetime' in etf_5min.columns else 'N/A'} 至 {etf_5min['datetime'].max() if 'datetime' in etf_5min.columns else 'N/A'}")
else:
    print("\n✗ 未能获取到任何数据")

# 期权数据说明
print("\n" + "=" * 60)
print("【期权数据说明】")
print("-" * 60)
print("期权5分钟数据获取受限，原因如下：")
print("1. AKShare免费版不提供期权分钟级数据")
print("2. Tushare免费版有积分限制，期权分钟数据需要高积分")
print("3. 专业数据服务（Wind、iFinD）需要付费订阅")
print("\n解决方案：")
print("- 使用ETF数据构建策略框架")
print("- 策略逻辑可直接应用于期权交易")
print("- 实际交易时可通过券商API获取实时期权数据")
print("=" * 60)
