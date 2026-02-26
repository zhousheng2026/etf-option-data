# -*- coding: utf-8 -*-
"""
本地快速回测 - 科创50ETF期权30分钟通道突破
使用AKShare获取数据，本地运行，速度快
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class FastBacktest:
    """本地快速回测"""
    
    def __init__(self):
        self.channel_period = 20
        self.volume_threshold = 1.3
        self.body_threshold = 1.5
        
        # 只做买方，不做卖方
        self.only_buyer = True
        
        # 标的
        self.underlying_codes = {
            "588000": "华夏科创50ETF",
            "588080": "易方达科创50ETF",
            "510500": "南方中证500ETF",
            "159845": "华夏中证1000ETF",
        }
        
    def get_data_from_akshare(self, code, start_date, end_date):
        """从AKShare获取数据"""
        try:
            import akshare as ak
            
            logger.info(f"获取 {code} 数据...")
            
            # 获取ETF历史数据
            df = ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            
            if df is not None and not df.empty:
                df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_change', 'change', 'turnover']
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"获取成功: {len(df)} 条数据")
                return df
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            return pd.DataFrame()
    
    def generate_30min_data(self, daily_df):
        """将日线数据模拟为30分钟数据（加速回测）"""
        data_30min = []
        
        for _, row in daily_df.iterrows():
            date = row['date']
            open_p = row['open']
            high = row['high']
            low = row['low']
            close = row['close']
            volume = row['volume']
            
            # 一天模拟为8个30分钟K线
            for i in range(8):
                time_offset = timedelta(minutes=30*i)
                
                # 模拟价格波动
                if i == 0:
                    o, h, l, c = open_p, open_p*1.01, open_p*0.99, open_p*1.005
                elif i == 7:
                    o, h, l, c = close*0.995, high, low, close
                else:
                    o = open_p + (close - open_p) * (i/8)
                    h = o * 1.005
                    l = o * 0.995
                    c = o
                
                data_30min.append({
                    'time': date + time_offset,
                    'open': round(o, 4),
                    'high': round(h, 4),
                    'low': round(l, 4),
                    'close': round(c, 4),
                    'volume': volume // 8
                })
        
        return pd.DataFrame(data_30min)
    
    def calculate_channel(self, df):
        """计算通道"""
        if len(df) < self.channel_period:
            return 0, 0, 0
        
        recent = df.tail(self.channel_period)
        upper = recent['high'].max()
        lower = recent['low'].min()
        middle = (upper + lower) / 2
        return upper, lower, middle
    
    def run_backtest(self, start_date="20250101", end_date="20251231"):
        """运行回测"""
        logger.info("="*70)
        logger.info("本地快速回测 - 科创50ETF期权（只做买方）")
        logger.info(f"回测区间: {start_date} - {end_date}")
        logger.info("="*70)
        
        start_time = time.time()
        
        for code, name in self.underlying_codes.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"回测标的: {name} ({code})")
            logger.info(f"{'='*70}")
            
            # 获取日线数据
            daily_df = self.get_data_from_akshare(code, start_date, end_date)
            if daily_df.empty:
                continue
            
            # 转换为30分钟数据
            df = self.generate_30min_data(daily_df)
            logger.info(f"生成30分钟数据: {len(df)} 条")
            
            # 回测
            position = None
            trades = []
            
            for i in range(self.channel_period, len(df)):
                current_df = df.iloc[:i+1]
                current = df.iloc[i]
                
                upper, lower, middle = self.calculate_channel(current_df.iloc[:-1])
                if upper == 0:
                    continue
                
                current_price = current['close']
                
                # 检查离场
                if position:
                    if position['direction'] == 'CALL' and current_price < middle:
                        profit = (current_price - position['entry_price']) / position['entry_price'] * 100
                        trades.append({
                            'direction': '买入认购',
                            'entry': position['entry_price'],
                            'exit': current_price,
                            'profit': profit
                        })
                        position = None
                        
                    elif position['direction'] == 'PUT' and current_price > middle:
                        profit = (position['entry_price'] - current_price) / position['entry_price'] * 100
                        trades.append({
                            'direction': '买入认沽',
                            'entry': position['entry_price'],
                            'exit': current_price,
                            'profit': profit
                        })
                        position = None
                
                # 检查入场（只做买方）
                else:
                    # 突破上轨 - 买入认购（买方）
                    if current_price > upper:
                        position = {
                            'direction': 'CALL',  # 买入认购
                            'entry_price': current_price,
                            'entry_time': current['time']
                        }
                    
                    # 跌破下轨 - 买入认沽（买方）
                    elif current_price < lower:
                        position = {
                            'direction': 'PUT',  # 买入认沽
                            'entry_price': current_price,
                            'entry_time': current['time']
                        }
            
            # 统计结果
            self.print_results(name, trades)
        
        elapsed = time.time() - start_time
        logger.info(f"\n{'='*70}")
        logger.info(f"回测完成，耗时: {elapsed:.2f}秒")
        logger.info(f"{'='*70}")
    
    def print_results(self, name, trades):
        """打印结果"""
        logger.info(f"\n{name} 回测结果:")
        
        if trades:
            total = len(trades)
            wins = len([t for t in trades if t['profit'] > 0])
            win_rate = wins / total * 100
            total_profit = sum([t['profit'] for t in trades])
            
            logger.info(f"交易次数: {total}")
            logger.info(f"胜率: {win_rate:.1f}%")
            logger.info(f"总收益率: {total_profit:.2f}%")
            
            # 买方统计
            call_trades = [t for t in trades if t['direction'] == '买入认购']
            put_trades = [t for t in trades if t['direction'] == '买入认沽']
            
            if call_trades:
                call_win = len([t for t in call_trades if t['profit'] > 0])
                logger.info(f"买入认购: {len(call_trades)}次, 胜率{call_win/len(call_trades)*100:.1f}%")
            
            if put_trades:
                put_win = len([t for t in put_trades if t['profit'] > 0])
                logger.info(f"买入认沽: {len(put_trades)}次, 胜率{put_win/len(put_trades)*100:.1f}%")
        else:
            logger.info("无交易")


if __name__ == "__main__":
    backtest = FastBacktest()
    backtest.run_backtest()
