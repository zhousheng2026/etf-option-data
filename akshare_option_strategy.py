# -*- coding: utf-8 -*-
"""
AKShare版本 - 股指期权趋势爆发策略
30分钟通道突破交易系统
使用AKShare免费获取数据
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('akshare_option_trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AKShareOptionStrategy:
    """
    股指期权趋势爆发策略（30分钟通道突破）
    使用AKShare免费获取数据
    
    策略逻辑：
    1. 计算20根30分钟K线的高低点形成通道
    2. 价格突破通道上轨 + 成交量放大1.3倍 + K线实体放大1.5倍 → 做多
    3. 价格跌破通道下轨 + 成交量放大1.3倍 + K线实体放大1.5倍 → 做空
    4. 通道跌破离场
    """
    
    def __init__(self):
        # 策略参数
        self.channel_period = 20  # 通道周期（20根K线）
        self.volume_threshold = 1.3  # 成交量阈值
        self.body_threshold = 1.5  # K线实体阈值
        
        # 时间过滤
        self.no_entry_start = "09:30"
        self.no_entry_end = "10:00"
        self.no_new_position_start = "14:30"
        self.no_new_position_end = "15:00"
        
        # 标的合约（科创50ETF期权为主，排除50/300ETF因国家队干扰）
        self.underlying_codes = {
            "588000": "华夏科创50ETF（首选，科技股为主，国家队干预少）",
            "588080": "易方达科创50ETF（备选）",
            "510500": "南方中证500ETF（备选，中小盘）",
            "159845": "华夏中证1000ETF（备选，小盘股）",
        }
        
        # 状态
        self.positions = {}  # 当前持仓
        self.klines_cache = {}  # K线数据缓存
        
        logger.info("=" * 70)
        logger.info("AKShare版本 - 股指期权趋势爆发策略")
        logger.info("=" * 70)
        logger.info(f"策略版本: 1.0")
        logger.info(f"通道周期: {self.channel_period}根30分钟K线")
        logger.info(f"成交量阈值: {self.volume_threshold}倍")
        logger.info(f"K线实体阈值: {self.body_threshold}倍")
        logger.info("=" * 70)
    
    def get_option_underlying_klines(self, code: str, period: str = "30") -> pd.DataFrame:
        """
        获取期权标的ETF的K线数据
        
        :param code: ETF代码（如 510050）
        :param period: 周期（1, 5, 15, 30, 60分钟）
        :return: K线DataFrame
        """
        try:
            import akshare as ak
            
            logger.info(f"获取 {code} {period}分钟K线数据...")
            
            # 使用AKShare获取ETF分钟数据
            # 注意：AKShare的分钟数据接口可能需要调整
            df = ak.fund_etf_hist_min_em(
                symbol=code,
                period=period,
                adjust="qfq"  # 前复权
            )
            
            if df is not None and not df.empty:
                # 标准化列名
                df.columns = ['time', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'pct_change', 'change', 'turnover']
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time')
                logger.info(f"获取成功，共 {len(df)} 条数据")
                return df
            else:
                logger.warning(f"获取 {code} 数据为空")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取 {code} 数据失败: {e}")
            return pd.DataFrame()
    
    def get_option_chain(self, underlying: str) -> pd.DataFrame:
        """
        获取期权链数据
        
        :param underlying: 标的代码（如 510050）
        :return: 期权合约列表
        """
        try:
            import akshare as ak
            
            logger.info(f"获取 {underlying} 期权链...")
            
            # 根据标的获取对应的期权数据
            if underlying == "510050":
                # 50ETF期权
                df = ak.option_cffex_50etf_spot()
            elif underlying in ["510300", "159919"]:
                # 300ETF期权
                df = ak.option_cffex_300etf_spot()
            elif underlying in ["510500", "159922"]:
                # 500ETF期权（中证500）
                df = ak.option_cffex_1000_index_spot()  # 可能需要调整
            else:
                logger.warning(f"不支持的标的: {underlying}")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            logger.error(f"获取期权链失败: {e}")
            return pd.DataFrame()
    
    def calculate_channel(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        计算趋势通道
        
        :param df: K线DataFrame
        :return: (上轨, 下轨, 中轨)
        """
        if len(df) < self.channel_period:
            return 0.0, 0.0, 0.0
        
        recent_df = df.tail(self.channel_period)
        upper = recent_df['high'].max()
        lower = recent_df['low'].min()
        middle = (upper + lower) / 2
        
        return upper, lower, middle
    
    def check_entry_signal(self, code: str, df: pd.DataFrame) -> Optional[str]:
        """
        检查入场信号
        
        :param code: 合约代码
        :param df: K线DataFrame
        :return: 'LONG', 'SHORT', 或 None
        """
        if len(df) < self.channel_period + 1:
            return None
        
        # 获取当前时间和K线
        current_time = datetime.now().strftime("%H:%M")
        current_kline = df.iloc[-1]
        prev_klines = df.iloc[:-1]
        
        # 时间过滤：9:30-10:00不入场
        if self.no_entry_start <= current_time <= self.no_entry_end:
            logger.debug(f"{code} 时间过滤：{current_time} 不入场")
            return None
        
        # 计算通道
        upper, lower, middle = self.calculate_channel(prev_klines)
        if upper == 0:
            return None
        
        # 计算平均成交量和K线实体
        avg_volume = prev_klines.tail(self.channel_period)['volume'].mean()
        avg_body = (prev_klines.tail(self.channel_period)['high'] - 
                   prev_klines.tail(self.channel_period)['low']).mean()
        
        current_volume = current_kline['volume']
        current_body = abs(current_kline['close'] - current_kline['open'])
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        body_ratio = current_body / avg_body if avg_body > 0 else 0
        
        close_price = current_kline['close']
        
        logger.info(f"{code} 当前价格: {close_price:.4f}, 上轨: {upper:.4f}, 下轨: {lower:.4f}")
        logger.info(f"成交量比: {volume_ratio:.2f}, 实体比: {body_ratio:.2f}")
        
        # 做多信号：突破上轨 + 成交量放大 + K线实体放大
        if (close_price > upper and 
            volume_ratio > self.volume_threshold and 
            body_ratio > self.body_threshold):
            logger.info(f"【做多信号】{code} 突破上轨 {upper:.4f}, "
                       f"成交量{volume_ratio:.2f}倍, 实体{body_ratio:.2f}倍")
            return 'LONG'
        
        # 做空信号：跌破下轨 + 成交量放大 + K线实体放大
        if (close_price < lower and 
            volume_ratio > self.volume_threshold and 
            body_ratio > self.body_threshold):
            logger.info(f"【做空信号】{code} 跌破下轨 {lower:.4f}, "
                       f"成交量{volume_ratio:.2f}倍, 实体{body_ratio:.2f}倍")
            return 'SHORT'
        
        return None
    
    def check_exit_signal(self, code: str, position: Dict, df: pd.DataFrame) -> bool:
        """
        检查离场信号
        
        :param code: 合约代码
        :param position: 持仓信息
        :param df: K线DataFrame
        :return: 是否平仓
        """
        if len(df) < self.channel_period:
            return False
        
        upper, lower, middle = self.calculate_channel(df)
        current_price = df.iloc[-1]['close']
        
        # 做多持仓：跌破中轨或下轨离场
        if position.get('direction') == 'LONG':
            if current_price < middle or current_price < lower:
                logger.info(f"【做多平仓】{code} 价格{current_price:.4f} 跌破中轨/下轨")
                return True
        
        # 做空持仓：突破中轨或上轨离场
        if position.get('direction') == 'SHORT':
            if current_price > middle or current_price > upper:
                logger.info(f"【做空平仓】{code} 价格{current_price:.4f} 突破中轨/上轨")
                return True
        
        return False
    
    def select_option_contract(self, underlying: str, direction: str, 
                               current_price: float) -> Optional[str]:
        """
        选择期权合约（选择平值或虚值合约）
        
        :param underlying: 标的代码
        :param direction: 方向（LONG/SHORT）
        :param current_price: 当前价格
        :return: 期权合约代码
        """
        try:
            # 获取期权链
            option_chain = self.get_option_chain(underlying)
            if option_chain.empty:
                return None
            
            # TODO: 根据当前价格选择平值或虚值合约
            # 简化处理：返回第一个合约
            if not option_chain.empty:
                contract = option_chain.iloc[0]['合约代码']
                return contract
            
            return None
            
        except Exception as e:
            logger.error(f"选择期权合约失败: {e}")
            return None
    
    def run_backtest(self, start_date: str = "20250101", end_date: str = "20250225"):
        """
        回测模式
        
        :param start_date: 开始日期
        :param end_date: 结束日期
        """
        logger.info("=" * 70)
        logger.info("开始回测...")
        logger.info(f"回测区间: {start_date} - {end_date}")
        logger.info("=" * 70)
        
        results = []
        
        for code, name in self.underlying_codes.items():
            logger.info(f"\n回测标的: {name} ({code})")
            
            # 获取历史数据
            df = self.get_option_underlying_klines(code, period="30")
            if df.empty:
                continue
            
            # 过滤回测区间
            df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
            
            if len(df) < self.channel_period + 1:
                continue
            
            # 遍历K线
            position = None
            trades = []
            
            for i in range(self.channel_period, len(df)):
                current_df = df.iloc[:i+1]
                current_kline = df.iloc[i]
                
                if position:
                    # 检查离场信号
                    if self.check_exit_signal(code, position, current_df):
                        # 平仓
                        exit_price = current_kline['close']
                        profit = (exit_price - position['entry_price']) * position['direction_mult']
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_kline['time'],
                            'direction': position['direction'],
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'profit': profit
                        })
                        position = None
                else:
                    # 检查入场信号
                    signal = self.check_entry_signal(code, current_df)
                    if signal:
                        # 开仓
                        position = {
                            'direction': signal,
                            'direction_mult': 1 if signal == 'LONG' else -1,
                            'entry_price': current_kline['close'],
                            'entry_time': current_kline['time']
                        }
            
            # 统计结果
            if trades:
                total_profit = sum([t['profit'] for t in trades])
                win_trades = len([t for t in trades if t['profit'] > 0])
                win_rate = win_trades / len(trades) * 100
                
                logger.info(f"交易次数: {len(trades)}")
                logger.info(f"盈利次数: {win_trades}")
                logger.info(f"胜率: {win_rate:.2f}%")
                logger.info(f"总盈亏: {total_profit:.4f}")
                
                results.append({
                    'code': code,
                    'name': name,
                    'trades': len(trades),
                    'win_rate': win_rate,
                    'profit': total_profit
                })
        
        # 汇总结果
        logger.info("\n" + "=" * 70)
        logger.info("回测结果汇总")
        logger.info("=" * 70)
        for r in results:
            logger.info(f"{r['name']}: 交易{r['trades']}次, 胜率{r['win_rate']:.1f}%, 盈亏{r['profit']:.4f}")
    
    def run_realtime(self):
        """实盘/仿真交易模式"""
        logger.info("=" * 70)
        logger.info("策略启动 - 实时模式")
        logger.info("=" * 70)
        logger.info("注意：此模式仅生成交易信号，不自动下单")
        logger.info("请根据信号手动在同花顺/光大证券软件中下单")
        logger.info("=" * 70)
        
        try:
            while True:
                current_time = datetime.now().strftime("%H:%M")
                logger.info(f"\n{'='*70}")
                logger.info(f"当前时间: {current_time}")
                logger.info(f"{'='*70}")
                
                for code, name in self.underlying_codes.items():
                    logger.info(f"\n检查标的: {name} ({code})")
                    
                    # 获取最新K线数据
                    df = self.get_option_underlying_klines(code, period="30")
                    if df.empty:
                        continue
                    
                    # 检查是否有持仓
                    position = self.positions.get(code)
                    
                    if position:
                        # 检查离场信号
                        if self.check_exit_signal(code, position, df):
                            logger.info(f"【平仓信号】{code} {position['direction']}")
                            logger.info(f"请手动平仓: {position['contract']}")
                            del self.positions[code]
                    else:
                        # 时间过滤：14:30-15:00不开新仓
                        if self.no_new_position_start <= current_time <= self.no_new_position_end:
                            continue
                        
                        # 检查入场信号
                        signal = self.check_entry_signal(code, df)
                        if signal:
                            current_price = df.iloc[-1]['close']
                            contract = self.select_option_contract(code, signal, current_price)
                            
                            if contract:
                                logger.info(f"【开仓信号】{code} {signal}")
                                logger.info(f"建议合约: {contract}")
                                logger.info(f"请手动开仓买入")
                                
                                self.positions[code] = {
                                    'contract': contract,
                                    'direction': signal,
                                    'entry_price': current_price,
                                    'entry_time': datetime.now()
                                }
                
                # 每5分钟检查一次
                logger.info(f"\n等待5分钟后再次检查...")
                time.sleep(300)
                
        except KeyboardInterrupt:
            logger.info("策略停止")
        except Exception as e:
            logger.error(f"策略异常: {e}")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("AKShare版本 - 股指期权趋势爆发策略")
    print("=" * 70)
    print("\n【策略说明】")
    print("- 基于30分钟K线趋势通道突破")
    print("- 通道周期: 20根K线")
    print("- 成交量阈值: 1.3倍")
    print("- K线实体阈值: 1.5倍")
    print("\n【时间过滤】")
    print("- 9:30-10:00 不入场")
    print("- 14:30-15:00 不开新仓")
    print("\n【交易标的】")
    print("- 50ETF期权 (510050)")
    print("- 300ETF期权 (510300)")
    print("- 500ETF期权 (510500)")
    print("=" * 70)
    
    # 创建策略实例
    strategy = AKShareOptionStrategy()
    
    # 选择模式
    print("\n选择运行模式:")
    print("1. 回测模式")
    print("2. 实时信号模式")
    
    mode = input("\n请输入模式 (1/2): ").strip()
    
    if mode == "1":
        strategy.run_backtest()
    elif mode == "2":
        strategy.run_realtime()
    else:
        print("无效选择，退出")


if __name__ == "__main__":
    main()
