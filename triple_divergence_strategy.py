# -*- coding: utf-8 -*-
"""
三重底背离买入 + 通道跌破卖出策略

买入信号（三重底背离）：
1. 布林带底背离：价格触及下轨，且下轨走平或向上
2. MACD底背离：价格创新低，MACD(DIF)不创新低
3. 价格底背离：连续两个低点，后低点高于前低点

卖出信号：
- 30分钟通道跌破中轨或下轨（沿用通道策略）

只做买方，不做卖方
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class TripleDivergenceStrategy:
    """
    三重底背离买入策略
    """
    
    def __init__(self):
        # 布林带参数
        self.boll_period = 20
        self.boll_std = 2
        
        # MACD参数
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # 通道参数（用于卖出）
        self.channel_period = 20
        
        # 底背离确认参数
        self.divergence_lookback = 10  # 回溯10根K线找低点
        
        # 只做买方
        self.only_buyer = True
        
        # 标的
        self.underlying_codes = {
            "588000": "华夏科创50ETF",
            "588080": "易方达科创50ETF",
            "510500": "南方中证500ETF",
            "159845": "华夏中证1000ETF",
        }
    
    def calculate_bollinger(self, df):
        """计算布林带"""
        df['boll_mid'] = df['close'].rolling(window=self.boll_period).mean()
        df['boll_std'] = df['close'].rolling(window=self.boll_period).std()
        df['boll_upper'] = df['boll_mid'] + self.boll_std * df['boll_std']
        df['boll_lower'] = df['boll_mid'] - self.boll_std * df['boll_std']
        return df
    
    def calculate_macd(self, df):
        """计算MACD"""
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd_dif'] = ema_fast - ema_slow
        df['macd_dea'] = df['macd_dif'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = 2 * (df['macd_dif'] - df['macd_dea'])
        return df
    
    def find_price_lows(self, df, lookback=10):
        """
        找价格低点（用于判断底背离）
        返回最近两个低点的索引和价格
        """
        recent_df = df.tail(lookback)
        lows = []
        
        for i in range(1, len(recent_df) - 1):
            if (recent_df.iloc[i]['low'] < recent_df.iloc[i-1]['low'] and 
                recent_df.iloc[i]['low'] < recent_df.iloc[i+1]['low']):
                lows.append({
                    'index': recent_df.index[i],
                    'price': recent_df.iloc[i]['low'],
                    'macd': recent_df.iloc[i]['macd_dif']
                })
        
        # 返回最近两个低点
        if len(lows) >= 2:
            return lows[-2], lows[-1]  # 前一个低点，最近一个低点
        return None, None
    
    def check_bollinger_divergence(self, df):
        """
        检查布林带底背离
        条件：价格触及下轨，且下轨走平或向上
        """
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 价格触及或跌破下轨
        touch_lower = current['close'] <= current['boll_lower'] * 1.02
        
        # 下轨走平或向上（最近3根）
        recent_lower = df['boll_lower'].tail(3)
        lower_flat_or_up = recent_lower.iloc[-1] >= recent_lower.iloc[0] * 0.998
        
        return touch_lower and lower_flat_or_up
    
    def check_macd_divergence(self, df):
        """
        检查MACD底背离
        条件：价格创新低，MACD(DIF)不创新低
        """
        low1, low2 = self.find_price_lows(df, self.divergence_lookback)
        
        if low1 is None or low2 is None:
            return False
        
        # 价格创新低
        price_lower = low2['price'] < low1['price']
        
        # MACD不创新低（底背离）
        macd_not_lower = low2['macd'] > low1['macd']
        
        return price_lower and macd_not_lower
    
    def check_price_divergence(self, df):
        """
        检查价格底背离
        条件：连续两个低点，后低点高于前低点（抬高）
        """
        low1, low2 = self.find_price_lows(df, self.divergence_lookback)
        
        if low1 is None or low2 is None:
            return False
        
        # 后低点高于前低点（底背离）
        return low2['price'] > low1['price']
    
    def check_triple_divergence_buy_signal(self, df):
        """
        检查三重底背离买入信号
        
        三重底背离：
        1. 布林带底背离
        2. MACD底背离
        3. 价格底背离
        
        满足2个及以上即可买入
        """
        # 计算指标
        df = self.calculate_bollinger(df)
        df = self.calculate_macd(df)
        
        # 检查三个背离
        boll_div = self.check_bollinger_divergence(df)
        macd_div = self.check_macd_divergence(df)
        price_div = self.check_price_divergence(df)
        
        divergence_count = sum([boll_div, macd_div, price_div])
        
        logger.info(f"背离检查：布林带{boll_div}, MACD{macd_div}, 价格{price_div} "
                   f"(共{divergence_count}个)")
        
        # 满足2个及以上背离，产生买入信号
        if divergence_count >= 2:
            signal_type = []
            if boll_div: signal_type.append("布林带")
            if macd_div: signal_type.append("MACD")
            if price_div: signal_type.append("价格")
            
            logger.info(f"【三重底背离买入信号】{'+'.join(signal_type)}")
            return True, signal_type
        
        return False, []
    
    def calculate_channel(self, df):
        """计算通道（用于卖出）"""
        recent = df.tail(self.channel_period)
        upper = recent['high'].max()
        lower = recent['low'].min()
        middle = (upper + lower) / 2
        return upper, lower, middle
    
    def check_channel_exit_signal(self, position, current_price, df):
        """
        检查通道卖出信号（沿用原策略）
        做多持仓：跌破中轨或下轨卖出
        """
        upper, lower, middle = self.calculate_channel(df)
        
        if position['direction'] == 'LONG':
            if current_price < middle:
                logger.info(f"【通道卖出】价格{current_price:.4f}跌破中轨{middle:.4f}")
                return True
        
        return False
    
    def run_backtest(self, days=60):
        """运行回测"""
        logger.info("="*70)
        logger.info("三重底背离买入 + 通道卖出策略")
        logger.info("="*70)
        logger.info("买入条件：布林带/MACD/价格 三重底背离（满足2个即可）")
        logger.info("卖出条件：30分钟通道跌破中轨")
        logger.info("="*70)
        
        for code, name in self.underlying_codes.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"回测标的: {name} ({code})")
            logger.info(f"{'='*70}")
            
            # 生成模拟数据（实际应用真实数据）
            df = self.generate_mock_data(days)
            logger.info(f"生成数据: {len(df)} 条K线")
            
            # 回测
            position = None
            trades = []
            
            for i in range(self.boll_period + self.divergence_lookback, len(df)):
                current_df = df.iloc[:i+1]
                current = df.iloc[i]
                current_price = current['close']
                
                # 检查持仓
                if position:
                    # 检查卖出信号（通道跌破）
                    if self.check_channel_exit_signal(position, current_price, current_df):
                        profit = (current_price - position['entry_price']) / position['entry_price'] * 100
                        trades.append({
                            'direction': '买入认购',
                            'entry_time': position['entry_time'],
                            'exit_time': current['time'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'profit': profit,
                            'signal_type': position['signal_type']
                        })
                        position = None
                
                # 检查买入信号（三重底背离）
                else:
                    has_signal, signal_type = self.check_triple_divergence_buy_signal(current_df)
                    if has_signal:
                        position = {
                            'direction': 'LONG',
                            'entry_price': current_price,
                            'entry_time': current['time'],
                            'signal_type': '+'.join(signal_type)
                        }
                        logger.info(f"【买入】价格:{current_price:.4f} 信号:{'+'.join(signal_type)}")
            
            # 打印结果
            self.print_results(name, trades)
    
    def generate_mock_data(self, days=60):
        """生成模拟数据"""
        np.random.seed(42)
        dates = []
        start = datetime(2025, 1, 1)
        for i in range(days * 8):
            dates.append(start + timedelta(minutes=30*i))
        
        # 生成带趋势的价格
        base = 1.0
        prices = []
        for i in range(len(dates)):
            trend = np.sin(i/20) * 0.001  # 周期性趋势
            noise = np.random.normal(0, 0.01)
            base += trend + noise
            base = max(0.8, min(1.3, base))
            prices.append(base)
        
        data = []
        for date, close in zip(dates, prices):
            high = close + abs(np.random.normal(0, 0.008))
            low = close - abs(np.random.normal(0, 0.008))
            low = max(0.01, low)
            open_p = low + (high - low) * np.random.random()
            
            data.append({
                'time': date,
                'open': round(open_p, 4),
                'high': round(high, 4),
                'low': round(low, 4),
                'close': round(close, 4),
                'volume': np.random.randint(500000, 5000000)
            })
        
        return pd.DataFrame(data)
    
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
            
            # 按信号类型统计
            signal_types = {}
            for t in trades:
                st = t['signal_type']
                signal_types[st] = signal_types.get(st, 0) + 1
            
            logger.info("信号类型分布:")
            for st, count in signal_types.items():
                logger.info(f"  {st}: {count}次")
        else:
            logger.info("无交易")


if __name__ == "__main__":
    strategy = TripleDivergenceStrategy()
    strategy.run_backtest()
