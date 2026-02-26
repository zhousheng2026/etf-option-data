# -*- coding: utf-8 -*-
"""
模拟回测 - 科创50ETF期权30分钟通道突破策略
排除50/300ETF因国家队干扰
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class MockBacktest:
    """模拟回测 - 科创50ETF期权为主"""
    
    def __init__(self):
        # 策略参数
        self.channel_period = 20
        self.volume_threshold = 1.3
        self.body_threshold = 1.5
        
        # 标的：科创50ETF期权为主（排除50/300ETF因国家队干扰）
        self.underlying_codes = {
            "588000": "华夏科创50ETF（首选，科技股为主，国家队干预少）",
            "588080": "易方达科创50ETF（备选）",
            "510500": "南方中证500ETF（备选，中小盘）",
            "159845": "华夏中证1000ETF（备选，小盘股）",
        }
        
    def generate_mock_data(self, days=60):
        """生成模拟K线数据 - 科创50ETF风格（波动较大）"""
        np.random.seed(42)
        
        # 生成日期
        dates = []
        start_date = datetime(2025, 1, 1)
        for i in range(days * 8):  # 每天8个30分钟K线
            dates.append(start_date + timedelta(minutes=30*i))
        
        # 生成价格（科创50风格：波动大，趋势明显）
        base_price = 1.0  # 科创50ETF价格较低
        prices = []
        trend = 0
        for i in range(len(dates)):
            if i % 20 == 0:  # 每20根K线改变趋势
                trend = np.random.choice([-0.002, 0, 0.002])  # 波动更大
            noise = np.random.normal(0, 0.015)  # 噪声更大
            base_price += trend + noise
            base_price = max(0.8, min(1.3, base_price))  # 限制范围
            prices.append(base_price)
        
        # 生成OHLCV
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close + abs(np.random.normal(0, 0.008))
            low = close - abs(np.random.normal(0, 0.008))
            low = max(0.01, low)  # 确保low>0
            open_price = low + (high - low) * np.random.random()
            volume = np.random.randint(500000, 5000000)  # 成交量更大
            
            data.append({
                'time': date,
                'open': round(open_price, 4),
                'high': round(high, 4),
                'low': round(low, 4),
                'close': round(close, 4),
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def calculate_channel(self, df):
        """计算通道"""
        if len(df) < self.channel_period:
            return 0, 0, 0
        
        recent = df.tail(self.channel_period)
        upper = recent['high'].max()
        lower = recent['low'].min()
        middle = (upper + lower) / 2
        return upper, lower, middle
    
    def run_backtest(self):
        """运行回测"""
        logger.info("="*70)
        logger.info("科创50ETF期权 - 30分钟通道突破策略回测")
        logger.info("（排除50/300ETF因国家队干扰）")
        logger.info("="*70)
        
        for code, name in self.underlying_codes.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"回测标的: {name} ({code})")
            logger.info(f"{'='*70}")
            
            # 生成模拟数据
            df = self.generate_mock_data(days=60)
            logger.info(f"生成模拟数据: {len(df)} 条K线")
            
            # 回测
            position = None
            trades = []
            
            for i in range(self.channel_period, len(df)):
                current_df = df.iloc[:i+1]
                current = df.iloc[i]
                
                upper, lower, middle = self.calculate_channel(current_df.iloc[:-1])
                
                if upper == 0:
                    continue
                
                # 计算成交量和实体比例
                avg_volume = current_df.iloc[:-1].tail(self.channel_period)['volume'].mean()
                avg_body = (current_df.iloc[:-1].tail(self.channel_period)['high'] - 
                           current_df.iloc[:-1].tail(self.channel_period)['low']).mean()
                
                current_volume = current['volume']
                current_body = current['high'] - current['low']
                
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                body_ratio = current_body / avg_body if avg_body > 0 else 0
                
                # 检查离场
                if position:
                    if position['direction'] == 'LONG' and current['close'] < middle:
                        profit = (current['close'] - position['entry_price']) / position['entry_price'] * 100
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current['time'],
                            'direction': 'LONG',
                            'entry_price': position['entry_price'],
                            'exit_price': current['close'],
                            'profit': profit
                        })
                        position = None
                        
                    elif position['direction'] == 'SHORT' and current['close'] > middle:
                        profit = (position['entry_price'] - current['close']) / position['entry_price'] * 100
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current['time'],
                            'direction': 'SHORT',
                            'entry_price': position['entry_price'],
                            'exit_price': current['close'],
                            'profit': profit
                        })
                        position = None
                
                # 检查入场
                else:
                    # 做多信号
                    if current['close'] > upper and volume_ratio > self.volume_threshold and body_ratio > self.body_threshold:
                        position = {
                            'direction': 'LONG',
                            'entry_price': current['close'],
                            'entry_time': current['time']
                        }
                        logger.info(f"【做多】{current['time']} 价格:{current['close']:.4f} 上轨:{upper:.4f}")
                    
                    # 做空信号
                    elif current['close'] < lower and volume_ratio > self.volume_threshold and body_ratio > self.body_threshold:
                        position = {
                            'direction': 'SHORT',
                            'entry_price': current['close'],
                            'entry_time': current['time']
                        }
                        logger.info(f"【做空】{current['time']} 价格:{current['close']:.4f} 下轨:{lower:.4f}")
            
            # 统计结果
            self.print_results(code, name, trades)
    
    def print_results(self, code, name, trades):
        """打印回测结果"""
        logger.info(f"\n{'='*70}")
        logger.info(f"{name} ({code}) 回测结果")
        logger.info(f"{'='*70}")
        
        if trades:
            total_trades = len(trades)
            long_trades = len([t for t in trades if t['direction'] == 'LONG'])
            short_trades = len([t for t in trades if t['direction'] == 'SHORT'])
            win_trades = len([t for t in trades if t['profit'] > 0])
            lose_trades = len([t for t in trades if t['profit'] <= 0])
            
            total_profit = sum([t['profit'] for t in trades])
            avg_profit = total_profit / total_trades
            win_rate = win_trades / total_trades * 100
            
            logger.info(f"总交易次数: {total_trades}")
            logger.info(f"  做多: {long_trades}次")
            logger.info(f"  做空: {short_trades}次")
            logger.info(f"盈利次数: {win_trades}")
            logger.info(f"亏损次数: {lose_trades}")
            logger.info(f"胜率: {win_rate:.2f}%")
            logger.info(f"总收益率: {total_profit:.2f}%")
            logger.info(f"平均收益率: {avg_profit:.2f}%")
            
            # 详细交易记录
            logger.info(f"\n详细交易记录（前5笔）:")
            for i, t in enumerate(trades[:5], 1):
                logger.info(f"{i}. {t['direction']} {t['entry_time']}-> exit")
        else:
            logger.info("无交易记录")


if __name__ == "__main__":
    backtest = MockBacktest()
    backtest.run_backtest()
