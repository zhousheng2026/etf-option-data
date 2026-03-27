#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
波动率突破策略主程序
Volatility Breakthrough Strategy - Main Entry
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.ctp_config import CTP_CONFIG
from config.strategy_config import VOLATILITY_CONFIG, SIGNAL_CONFIG, RISK_CONFIG
from config.symbols_config import MONITORING_SYMBOLS
from core.ctp_connection import CTPConnection
from core.volatility_analyzer import VolatilityAnalyzer
from core.signal_generator import SignalGenerator
from core.report_generator import ReportGenerator


def setup_logging():
    """配置日志"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"volatility_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def collect_market_data(ctp: CTPConnection, symbols: List[str]) -> Dict:
    """
    收集市场数据
    
    Args:
        ctp: CTP连接对象
        symbols: 品种列表
    
    Returns:
        市场数据字典
    """
    logger = logging.getLogger(__name__)
    market_data = {}
    
    logger.info(f"开始收集{len(symbols)}个品种的市场数据...")
    
    for symbol in symbols:
        try:
            # 获取最新行情
            quote = ctp.get_last_quote(symbol)
            if quote:
                market_data[symbol] = {
                    'symbol': symbol,
                    'last_price': quote.get('last_price', 0),
                    'open': quote.get('open', 0),
                    'high': quote.get('high', 0),
                    'low': quote.get('low', 0),
                    'volume': quote.get('volume', 0),
                    'open_interest': quote.get('open_interest', 0),
                    'datetime': quote.get('datetime', '')
                }
        except Exception as e:
            logger.error(f"获取{symbol}行情失败: {e}")
    
    logger.info(f"成功收集{len(market_data)}个品种的市场数据")
    return market_data


def run_strategy():
    """运行波动率突破策略"""
    # 配置日志
    logger = setup_logging()
    logger.info("="*60)
    logger.info("波动率突破策略启动")
    logger.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    try:
        # 步骤1：初始化CTP连接
        logger.info("步骤1: 初始化CTP连接...")
        ctp = CTPConnection(CTP_CONFIG)
        
        # 连接CTP（如果配置了CTP连接）
        # if not ctp.connect():
        #     logger.error("CTP连接失败，使用模拟数据")
        #     market_data = generate_mock_data(SYMBOLS_CONFIG['symbols'])
        # else:
        #     market_data = collect_market_data(ctp, SYMBOLS_CONFIG['symbols'])
        
        # 暂时使用模拟数据进行测试
        logger.info("使用模拟数据进行策略测试...")
        market_data = generate_mock_data(MONITORING_SYMBOLS)
        
        # 步骤2：分析波动率
        logger.info("步骤2: 分析波动率...")
        # 使用模拟数据中的波动率信息
        volatility_data = {}
        for symbol, data in market_data.items():
            volatility_data[symbol] = {
                'iv': data.get('iv', 0.25),
                'hv': data.get('hv', 0.22),
                'iv_rank': data.get('iv_rank', 50),
                'hv_rank': data.get('hv_rank', 50),
                'close': data.get('last_price', 0)
            }
        
        # 步骤3：生成交易信号
        logger.info("步骤3: 生成交易信号...")
        generator = SignalGenerator(SIGNAL_CONFIG)
        signals = generator.generate_signals(volatility_data)
        
        # 步骤4：生成市场概况
        logger.info("步骤4: 生成市场概况...")
        market_summary = {
            'total_symbols': len(market_data),
            'total_signals': len(signals),
            'long_vol_count': sum(1 for s in signals if s.signal_type == "LONG_VOL"),
            'short_vol_count': sum(1 for s in signals if s.signal_type == "SHORT_VOL"),
            'avg_iv_rank': sum(d.get('iv_rank', 0) for d in volatility_data.values()) / len(volatility_data) if volatility_data else 0,
            'avg_hv_rank': sum(d.get('hv_rank', 0) for d in volatility_data.values()) / len(volatility_data) if volatility_data else 0
        }
        
        # 步骤5：生成报告
        logger.info("步骤5: 生成报告...")
        report_gen = ReportGenerator("reports")
        report_path = report_gen.generate_daily_report(signals, volatility_data, market_summary)
        
        # 步骤6：输出结果
        logger.info("="*60)
        logger.info("策略执行完成")
        logger.info(f"监控品种: {market_summary['total_symbols']}")
        logger.info(f"有效信号: {market_summary['total_signals']}")
        logger.info(f"  - 做多波动率: {market_summary['long_vol_count']}")
        logger.info(f"  - 做空波动率: {market_summary['short_vol_count']}")
        logger.info(f"报告路径: {report_path}")
        logger.info("="*60)
        
        # 打印信号详情
        if signals:
            print("\n" + "="*80)
            print("交易信号详情")
            print("="*80)
            for i, signal in enumerate(signals, 1):
                print(f"\n信号 {i}: {signal.symbol}")
                print(f"  类型: {signal.signal_type} | 方向: {signal.direction} | 强度: {signal.strength:.2f}")
                print(f"  入场价: {signal.entry_price:.2f} | 止损: {signal.stop_loss:.2f} | 止盈: {signal.take_profit:.2f}")
                print(f"  IV百分位: {signal.iv_rank:.1f}% | HV百分位: {signal.hv_rank:.1f}%")
                print(f"  理由: {signal.reason}")
        else:
            print("\n当前无有效交易信号")
        
        return signals, volatility_data, report_path
        
    except Exception as e:
        logger.error(f"策略执行失败: {e}", exc_info=True)
        raise


def generate_mock_data(symbols: List[str]) -> Dict:
    """
    生成模拟数据（用于测试）
    
    Args:
        symbols: 品种列表
    
    Returns:
        模拟市场数据
    """
    import random
    
    logger = logging.getLogger(__name__)
    logger.info("生成模拟数据...")
    
    # 真实价格参考
    price_map = {
        "sc2604": 550, "bu2606": 3800, "pg2604": 4700, "ec2604": 1800,
        "fu2605": 3100, "px2605": 7200, "pp2605": 7500, "ta2605": 6000,
        "sm2605": 6800, "sa2605": 1500, "ma2605": 2500, "rb2610": 3200,
        "hc2610": 3400, "i2609": 800, "j2609": 2000, "jm2609": 1300
    }
    
    mock_data = {}
    
    # 设置固定的随机种子以确保生成信号
    random.seed(42)
    
    for i, symbol in enumerate(symbols):
        base_price = price_map.get(symbol, 1000)
        
        # 为部分品种设置产生信号的波动率数据
        if i < 3:  # 前3个品种：低IV，适合做多波动率
            iv = random.uniform(0.15, 0.22)  # 低IV
            iv_rank = random.uniform(10, 25)  # IV百分位低
            hv = random.uniform(0.20, 0.26)   # 相对稳定的HV
            hv_rank = random.uniform(40, 55)
        elif i < 6:  # 中间3个品种：高IV，适合做空波动率
            iv = random.uniform(0.38, 0.48)  # 高IV
            iv_rank = random.uniform(75, 90)  # IV百分位高
            hv = random.uniform(0.25, 0.32)   # HV相对较低
            hv_rank = random.uniform(45, 60)
        else:  # 其余品种：中性
            iv = random.uniform(0.25, 0.35)
            iv_rank = random.uniform(40, 60)
            hv = random.uniform(0.22, 0.32)
            hv_rank = random.uniform(35, 65)
        
        # 模拟行情数据
        mock_data[symbol] = {
            'symbol': symbol,
            'last_price': base_price * (1 + random.uniform(-0.02, 0.02)),
            'open': base_price * (1 + random.uniform(-0.01, 0.01)),
            'high': base_price * (1 + random.uniform(0, 0.03)),
            'low': base_price * (1 + random.uniform(-0.03, 0)),
            'volume': random.randint(10000, 100000),
            'open_interest': random.randint(50000, 200000),
            'datetime': datetime.now().strftime('%Y%m%d %H:%M:%S'),
            # 波动率数据
            'iv': iv,
            'hv': hv,
            'iv_rank': iv_rank,
            'hv_rank': hv_rank
        }
    
    return mock_data


if __name__ == "__main__":
    """主程序入口"""
    try:
        signals, vol_data, report_path = run_strategy()
        print(f"\n[OK] 策略执行成功！报告已保存: {report_path}")
    except KeyboardInterrupt:
        print("\n用户中断程序")
        sys.exit(0)
    except Exception as e:
        print(f"\n[X] 策略执行失败: {e}")
        sys.exit(1)
