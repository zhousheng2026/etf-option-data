#!/usr/bin/env python3
"""
北交所股票月线MACD底背离筛选工具
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_macd(close_prices, fast=12, slow=26, signal=9):
    """
    计算MACD指标
    返回: DIF, DEA, MACD柱状体
    """
    exp1 = close_prices.ewm(span=fast, adjust=False).mean()
    exp2 = close_prices.ewm(span=slow, adjust=False).mean()
    dif = exp1 - exp2
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = 2 * (dif - dea)
    return dif, dea, macd_hist

def detect_bullish_divergence(df, lookback=48):
    """
    检测底背离形态
    底背离定义：
    1. 股价创新低（当前价格低于前低）
    2. MACD柱状体未创新低，反而抬高
    
    参数:
        df: 包含价格和MACD数据的DataFrame
        lookback: 回溯周期数（月）
    
    返回: 背离信息字典或None
    """
    if len(df) < 12:  # 至少需要12个月数据
        return None
    
    recent_data = df.tail(lookback).copy()
    recent_data = recent_data.reset_index(drop=True)
    
    # 方法1：使用局部最小值检测
    price_lows = []
    
    for i in range(2, len(recent_data) - 2):
        # 价格局部低点（左右各2根K线）
        local_min = recent_data['close'].iloc[i-2:i+3].min()
        if recent_data['close'].iloc[i] == local_min:
            price_lows.append({
                'idx': i,
                'date': recent_data['date'].iloc[i],
                'price': recent_data['close'].iloc[i],
                'macd': recent_data['macd_hist'].iloc[i],
                'dif': recent_data['dif'].iloc[i]
            })
    
    # 方法2：如果没有找到足够低点，使用分段找最低点
    if len(price_lows) < 2:
        n = len(recent_data)
        # 分成两段找最低点
        mid = n // 2
        first_part = recent_data.iloc[:mid]
        second_part = recent_data.iloc[mid:]
        
        if len(first_part) > 0 and len(second_part) > 0:
            first_low_idx = first_part['close'].idxmin()
            second_low_idx = second_part['close'].idxmin() + mid
            
            # 确保两个低点不是同一个
            if abs(second_low_idx - first_low_idx) > 3:
                price_lows = [
                    {
                        'idx': second_low_idx,
                        'date': recent_data['date'].iloc[second_low_idx],
                        'price': recent_data['close'].iloc[second_low_idx],
                        'macd': recent_data['macd_hist'].iloc[second_low_idx],
                        'dif': recent_data['dif'].iloc[second_low_idx]
                    },
                    {
                        'idx': first_low_idx,
                        'date': recent_data['date'].iloc[first_low_idx],
                        'price': recent_data['close'].iloc[first_low_idx],
                        'macd': recent_data['macd_hist'].iloc[first_low_idx],
                        'dif': recent_data['dif'].iloc[first_low_idx]
                    }
                ]
    
    # 如果没有找到足够的低点，返回None
    if len(price_lows) < 2:
        return None
    
    # 按索引排序（时间顺序，最新的在前）
    price_lows_sorted = sorted(price_lows, key=lambda x: x['idx'], reverse=True)
    
    # 检查底背离
    divergences = []
    
    # 取最近的一个低点作为当前低点
    current_low = price_lows_sorted[0]
    
    # 向前找前一个低点进行比较
    for prev_low in price_lows_sorted[1:min(5, len(price_lows_sorted))]:
        # 获取对应位置的MACD和DIF值
        current_macd = recent_data['macd_hist'].iloc[current_low['idx']]
        previous_macd = recent_data['macd_hist'].iloc[prev_low['idx']]
        current_dif = recent_data['dif'].iloc[current_low['idx']]
        previous_dif = recent_data['dif'].iloc[prev_low['idx']]
        
        # 底背离核心条件：
        # 1. 当前价格 < 前低价格（价格创新低）
        # 2. 当前MACD > 前低MACD 或 当前DIF > 前低DIF（指标未创新低）
        
        price_decline_pct = (prev_low['price'] - current_low['price']) / prev_low['price'] * 100
        
        # 价格创新低的条件：当前价格明显低于前低（至少2%）
        price_making_lower_low = price_decline_pct > 2
        
        # MACD未创新低的条件
        macd_diff = current_macd - previous_macd
        macd_not_lower = macd_diff > -0.001  # 允许微小负值（考虑浮点误差）
        
        # DIF未创新低的条件
        dif_diff = current_dif - previous_dif
        dif_not_lower = dif_diff > -0.001
        
        # 底背离确认：价格新低 + (MACD或DIF未新低)
        if price_making_lower_low and (macd_not_lower or dif_not_lower):
            
            # 计算MACD抬升幅度
            if abs(previous_macd) > 0.001:
                macd_rise_pct = (current_macd - previous_macd) / abs(previous_macd) * 100
            else:
                macd_rise_pct = 100 if current_macd > previous_macd else 0
            
            # 计算DIF抬升幅度
            if abs(previous_dif) > 0.001:
                dif_rise_pct = (current_dif - previous_dif) / abs(previous_dif) * 100
            else:
                dif_rise_pct = 100 if current_dif > previous_dif else 0
            
            # 背离强度评分
            # 价格跌幅越大 + 指标抬升越明显 = 强度越高
            price_factor = min(price_decline_pct * 2, 50)  # 价格跌幅贡献最多50分
            macd_factor = max(0, macd_rise_pct) * 0.5  # MACD抬升贡献
            dif_factor = max(0, dif_rise_pct) * 0.5    # DIF抬升贡献
            
            divergence_strength = price_factor + macd_factor + dif_factor
            divergence_strength = min(100, divergence_strength)
            
            if divergence_strength > 10:  # 最小强度阈值
                divergences.append({
                    'current_date': current_low['date'],
                    'previous_date': prev_low['date'],
                    'current_price': round(current_low['price'], 2),
                    'previous_price': round(prev_low['price'], 2),
                    'current_macd': round(current_macd, 4),
                    'previous_macd': round(previous_macd, 4),
                    'current_dif': round(current_dif, 4),
                    'previous_dif': round(previous_dif, 4),
                    'price_decline_pct': round(price_decline_pct, 2),
                    'macd_rise_pct': round(macd_rise_pct, 2),
                    'divergence_strength': round(divergence_strength, 2)
                })
    
    if divergences:
        # 返回最强的一个背离
        return max(divergences, key=lambda x: x['divergence_strength'])
    
    return None

def get_bse_stock_list():
    """
    获取北交所股票列表
    """
    print("正在获取北交所股票列表...")
    
    # 使用AKShare获取北交所股票
    try:
        # 尝试获取北交所股票列表
        stock_df = ak.stock_info_bj_name_code()
        print(f"获取到 {len(stock_df)} 只北交所股票")
        return stock_df
    except Exception as e:
        print(f"获取北交所股票列表失败: {e}")
        # 备用方案：从所有A股中筛选
        try:
            all_stocks = ak.stock_zh_a_spot_em()
            # 北交所股票代码以8或9开头，共6位
            bse_stocks = all_stocks[all_stocks['代码'].str.match(r'^[89]\d{5}$', na=False)]
            print(f"备用方案获取到 {len(bse_stocks)} 只北交所股票")
            return bse_stocks
        except Exception as e2:
            print(f"备用方案也失败: {e2}")
            return pd.DataFrame()

def get_monthly_data(stock_code, stock_name):
    """
    获取单只股票的月线数据
    """
    try:
        # 使用AKShare获取月线数据
        df = ak.stock_zh_a_hist(symbol=stock_code, period="monthly", 
                                 start_date="20200101", adjust="qfq")
        if df is None or len(df) < 12:  # 至少需要12个月数据
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
        df['dif'], df['dea'], df['macd_hist'] = calculate_macd(df['close'])
        
        return df
    except Exception as e:
        return None

def main():
    print("=" * 80)
    print("北交所股票月线MACD底背离筛选")
    print("=" * 80)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 获取北交所股票列表
    stock_list = get_bse_stock_list()
    
    if stock_list.empty:
        print("未能获取到北交所股票列表，程序退出")
        return
    
    results = []
    total = len(stock_list)
    
    print(f"\n开始分析 {total} 只北交所股票...")
    print("-" * 80)
    
    for idx, row in stock_list.iterrows():
        try:
            # 获取股票代码和名称
            if '证券代码' in row:
                stock_code = str(row['证券代码']).zfill(6)
                stock_name = row['证券简称']
            elif '代码' in row:
                stock_code = str(row['代码']).zfill(6)
                stock_name = row['名称']
            else:
                continue
            
            # 跳过非北交所股票
            if not (stock_code.startswith('8') or stock_code.startswith('9')):
                continue
            
            # 显示进度
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"进度: {idx + 1}/{total} - 正在分析: {stock_code} {stock_name}")
            
            # 获取月线数据
            df = get_monthly_data(stock_code, stock_name)
            if df is None:
                continue
            
            # 检测底背离
            divergence = detect_bullish_divergence(df)
            
            if divergence:
                results.append({
                    '股票代码': stock_code,
                    '股票名称': stock_name,
                    '当前价格': divergence['current_price'],
                    '前低价格': divergence['previous_price'],
                    '当前MACD': divergence['current_macd'],
                    '前低MACD': divergence['previous_macd'],
                    '当前DIF': divergence['current_dif'],
                    '前低DIF': divergence['previous_dif'],
                    '价格跌幅(%)': divergence['price_decline_pct'],
                    'MACD抬升(%)': divergence['macd_rise_pct'],
                    '背离强度': divergence['divergence_strength'],
                    '当前低点日期': divergence['current_date'],
                    '前低日期': divergence['previous_date']
                })
                
        except Exception as e:
            continue
    
    print("-" * 80)
    print(f"\n分析完成！共发现 {len(results)} 只底背离股票")
    print()
    
    if results:
        # 创建结果DataFrame并排序
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values('背离强度', ascending=False)
        
        # 重置索引
        result_df = result_df.reset_index(drop=True)
        
        # 显示结果表格
        print("=" * 120)
        print("北交所月线MACD底背离股票列表（按背离强度排序）")
        print("=" * 120)
        
        # 格式化输出
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 20)
        
        print(result_df.to_string(index=True))
        
        # 保存到CSV
        output_file = f"bse_macd_divergence_{datetime.now().strftime('%Y%m%d')}.csv"
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_file}")
        
        # 统计信息
        print("\n" + "=" * 120)
        print("统计信息")
        print("=" * 120)
        print(f"总分析股票数: {total}")
        print(f"发现底背离股票数: {len(results)}")
        print(f"底背离比例: {len(results)/total*100:.2f}%")
        print(f"\n背离强度分布:")
        print(f"  高强度(>80): {len(result_df[result_df['背离强度'] > 80])} 只")
        print(f"  中高强度(60-80): {len(result_df[(result_df['背离强度'] >= 60) & (result_df['背离强度'] <= 80)])} 只")
        print(f"  中等强度(40-60): {len(result_df[(result_df['背离强度'] >= 40) & (result_df['背离强度'] < 60)])} 只")
        print(f"  低强度(<40): {len(result_df[result_df['背离强度'] < 40])} 只")
        
    else:
        print("未发现底背离股票")

if __name__ == "__main__":
    main()
