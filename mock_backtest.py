# -*- coding: utf-8 -*-
"""
策略回测 - 使用模拟数据测试逻辑
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class MockBacktest:
    """模拟回测"""
    
    def __init__(self):
        self.channel_period = 20
        self.volume_threshold = 1.3
        self.body_threshold = 1.5
        
    def generate_mock_data(self, days=60):
        """生成模拟K线数据"""
        np.random.seed(42)
        
        # 生成日期
        dates = []
        start_date = datetime(2025, 1, 1)
        for i in range(days * 8):  # 每天8个30分钟K线
            dates.append(start_date + timedelta(minutes=30*i))
        
        # 生成价格（带趋势）
        base_price = 2.5
        prices = []
        trend = 0
        for i in range(len(dates)):
            if i % 20 == 0:  # 每20根K线改变趋势
                trend = np.random.choice([-0.001, 0, 0.001])
            noise = np.random.normal(0, 0.01)
            base_price += trend + noise
            prices.append(base_price)
        
        # 生成OHLCV
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close + abs(np.random.normal(0, 0.005))
            low = close - abs(np.random.normal(0, 0.005))
            open_price = low + (high - low) * np.random.random()
            volume = np.random.randint(100000, 1000000)
            
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
        logger.info("模拟数据回测 - 30分钟通道突破策略")
        logger.info("="*70)
        
        # 生成模拟数据
        df = self.generate_mock_data(days=60)
        logger.info(f"生成模拟数据: {len(df)} 条K线")
        logger.info(f"数据区间: {df['time'].min()} ~ {df['time'].max()}")
        
        # 回测
        position = None
        trades = []
        
        for i in range(self.channel_period, len(df)):
            current_df = df.iloc[:i+1]
            current = df.iloc[i]
            
            upper, lower, middle = self.calculate_channel(current_df.iloc[:-1])
            
            # 计算成交量和实体比例
            avg_volume = current_df.iloc[:-1].tail(self.channel_period)['volume'].mean()
            avg_body = (current_df.iloc[:-1].tail(self.channel_period)['high'] - 
                       current_df.iloc[:-1].tail(self.channel_period)['low']).mean()
            
            volume_ratio = current['volume'] / avg_volume if avg_volume > 0 else 0
            body = current['high'] - current['low']
            body_ratio = body / avg_body if avg_body > 0 else 0
            
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
        logger.info("\n" + "="*70)
        logger.info("回测结果")
        logger.info("="*70)
        
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
            logger.info("\n详细交易记录:")
            for i, t in enumerate(trades[:10], 1):  # 只显示前10笔
                logger.info(f"{i}. {t['direction']} {t['entry_time']}->{t['exit_time']} "
                           f"入场:{t['entry_price']:.4f} 离场:{t['exit_price']:.4f} "
                           f"盈亏:{t['profit']:.2f}%")
        else:
            logger.info("无交易记录")
        
        return trades

if __name__ == "__main__":
    backtest = MockBacktest()
    trades = backtest.run_backtest()
