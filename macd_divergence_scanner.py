#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股月线MACD底背离筛选器 - 单线程版本
使用AKShare获取数据，筛选月线MACD底背离的股票
"""

import akshare as ak
import pandas as pd
import numpy as np
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    计算MACD指标
    """
    df = data.copy()
    # 计算EMA
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    
    # 计算DIF (MACD线)
    df['DIF'] = ema_fast - ema_slow
    
    # 计算DEA (信号线)
    df['DEA'] = df['DIF'].ewm(span=signal, adjust=False).mean()
    
    # 计算MACD柱状体
    df['MACD_Hist'] = 2 * (df['DIF'] - df['DEA'])
    
    return df

def find_divergence(data, lookback_periods=24):
    """
    识别底背离形态
    底背离：股价创新低，但MACD未创新低（反而抬高）
    """
    if len(data) < lookback_periods:
        return None
    
    df = data.copy()
    
    # 获取最近的数据
    recent_data = df.tail(lookback_periods)
    
    # 当前价格和MACD值（最近一个月）
    current_price = recent_data['close'].iloc[-1]
    current_macd_hist = recent_data['MACD_Hist'].iloc[-1]
    current_dif = recent_data['DIF'].iloc[-1]
    
    # 找出前期低点（过去lookback_periods个月内的最低价）
    past_low_price = recent_data['close'].min()
    past_low_idx = recent_data['close'].idxmin()
    past_low_macd_hist = recent_data.loc[past_low_idx, 'MACD_Hist']
    past_low_dif = recent_data.loc[past_low_idx, 'DIF']
    
    # 底背离条件判断：
    # 1. 当前价格低于前期低点（创新低）
    # 2. MACD柱状体高于前期低点时的值（未创新低，反而抬高）
    # 3. MACD柱状体为负值（在零轴下方）
    
    price_new_low = current_price < past_low_price * 0.98  # 当前价格低于前期低点2%以上
    macd_not_new_low = current_macd_hist > past_low_macd_hist  # MACD柱状体抬高
    macd_negative = current_macd_hist < 0  # MACD在零轴下方
    
    # 背离强度计算
    if price_new_low and macd_not_new_low and macd_negative:
        # 价格跌幅
        price_drop_pct = (past_low_price - current_price) / past_low_price * 100
        # MACD抬升幅度
        macd_rise = current_macd_hist - past_low_macd_hist
        # 背离强度评分 (综合考虑价格跌幅和MACD抬升)
        divergence_score = price_drop_pct * 0.5 + abs(macd_rise) * 10
        
        return {
            'current_price': current_price,
            'past_low_price': past_low_price,
            'price_drop_pct': price_drop_pct,
            'current_macd_hist': current_macd_hist,
            'past_low_macd_hist': past_low_macd_hist,
            'current_dif': current_dif,
            'past_low_dif': past_low_dif,
            'macd_rise': macd_rise,
            'divergence_score': divergence_score,
            'is_divergence': True
        }
    
    return None

def get_stock_monthly_data(stock_code, stock_name, max_retries=3):
    """
    获取单只股票的月线数据，带重试机制
    """
    for attempt in range(max_retries):
        try:
            # 获取月线数据
            df = ak.stock_zh_a_hist(symbol=stock_code, period="monthly", 
                                    start_date="20200101", adjust="qfq")
            if df is None or len(df) < 30:
                return None
            
            # 重命名列
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume'
            })
            
            # 计算MACD
            df = calculate_macd(df)
            
            # 识别底背离
            divergence = find_divergence(df)
            
            if divergence and divergence['is_divergence']:
                return {
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    **divergence
                }
            
            return None
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)  # 递增延迟
            else:
                return None
    
    return None

def get_a_stock_list():
    """
    获取A股所有股票列表
    """
    try:
        # 使用stock_info_a_code_name获取所有A股代码和名称
        stock_df = ak.stock_info_a_code_name()
        
        # 过滤掉ST、*ST、退市股票
        stock_df = stock_df[~stock_df['name'].str.contains('ST|退|N|C', na=False)]
        
        # 只保留6位数字代码
        stock_df = stock_df[stock_df['code'].str.match(r'^\d{6}$', na=False)]
        
        return stock_df[['code', 'name']].values.tolist()
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return []

def process_stocks(stock_list, max_stocks=None):
    """
    处理股票数据 - 单线程版本
    """
    results = []
    
    if max_stocks:
        stock_list = stock_list[:max_stocks]
    
    total = len(stock_list)
    print(f"开始处理 {total} 只股票...")
    print("-" * 80)
    
    start_time = time.time()
    
    for i, (code, name) in enumerate(stock_list, 1):
        try:
            result = get_stock_monthly_data(code, name)
            if result:
                results.append(result)
                print(f"[{i}/{total}] ✓ 发现底背离: {code} {name} (评分: {result['divergence_score']:.2f})")
            else:
                if i % 100 == 0:
                    elapsed = time.time() - start_time
                    speed = i / elapsed
                    remaining = (total - i) / speed if speed > 0 else 0
                    print(f"[{i}/{total}] 处理中... 速度: {speed:.1f}只/秒, 预计剩余: {remaining/60:.1f}分钟")
            
            # 每处理50只股票暂停一下，避免请求过快
            if i % 50 == 0:
                time.sleep(1)
                
        except Exception as e:
            print(f"[{i}/{total}] 错误: {code} {name} - {e}")
            continue
    
    return results

def main():
    print("=" * 80)
    print("A股月线MACD底背离筛选器")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 获取A股列表
    print("\n正在获取A股股票列表...")
    stock_list = get_a_stock_list()
    print(f"获取到 {len(stock_list)} 只股票（已过滤ST/退市/新股）")
    
    if len(stock_list) == 0:
        print("无法获取股票列表，程序退出")
        return
    
    # 处理所有股票
    print("\n开始扫描月线MACD底背离...")
    start_time = time.time()
    
    results = process_stocks(stock_list)
    
    elapsed_time = time.time() - start_time
    print(f"\n扫描完成，耗时: {elapsed_time:.1f}秒")
    
    if not results:
        print("\n未发现月线MACD底背离的股票")
        return
    
    # 按背离强度排序
    results_sorted = sorted(results, key=lambda x: x['divergence_score'], reverse=True)
    
    # 取前50名
    top_50 = results_sorted[:50]
    
    # 创建结果DataFrame
    df_result = pd.DataFrame(top_50)
    
    # 选择并重命名列
    df_display = pd.DataFrame({
        '排名': range(1, len(top_50) + 1),
        '股票代码': df_result['stock_code'],
        '股票名称': df_result['stock_name'],
        '当前价格': df_result['current_price'].round(2),
        '前低价格': df_result['past_low_price'].round(2),
        '价格跌幅(%)': df_result['price_drop_pct'].round(2),
        '当前MACD柱状体': df_result['current_macd_hist'].round(4),
        '前低MACD柱状体': df_result['past_low_macd_hist'].round(4),
        'MACD抬升': df_result['macd_rise'].round(4),
        '背离强度评分': df_result['divergence_score'].round(2)
    })
    
    print("\n" + "=" * 100)
    print("月线MACD底背离股票 TOP 50")
    print("=" * 100)
    print(df_display.to_string(index=False))
    print("=" * 100)
    
    # 保存到CSV
    output_file = f"macd_divergence_{datetime.now().strftime('%Y%m%d')}.csv"
    df_display.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_file}")
    
    # 统计信息
    print(f"\n统计信息:")
    print(f"  扫描股票总数: {len(stock_list)}")
    print(f"  发现底背离股票: {len(results)} 只")
    print(f"  显示前50名")
    print(f"  平均背离强度评分: {df_result['divergence_score'].mean():.2f}")

if __name__ == "__main__":
    main()
