#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
波动率突破信号生成器
Volatility Breakthrough Signal Generator
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """交易信号数据类"""
    symbol: str              # 合约代码
    signal_type: str         # 信号类型：LONG_VOL/SHORT_VOL
    direction: str           # 方向：BUY/SELL
    strength: float          # 信号强度（0-1）
    entry_price: float       # 建议入场价格
    stop_loss: float         # 止损价
    take_profit: float       # 止盈价
    iv_rank: float           # IV百分位
    hv_rank: float           # HV百分位
    iv_hv_diff: float        # IV-HV差值
    timestamp: str           # 时间戳
    reason: str              # 信号理由


class SignalGenerator:
    """波动率突破信号生成器"""
    
    def __init__(self, config: Dict):
        """
        初始化信号生成器
        
        Args:
            config: 策略配置
        """
        self.config = config
        self.iv_rank_threshold_low = config.get('iv_rank_threshold_low', 30)
        self.iv_rank_threshold_high = config.get('iv_rank_threshold_high', 70)
        self.iv_hv_spread_threshold = config.get('iv_hv_spread_threshold', 0.05)
        self.min_signal_strength = config.get('min_signal_strength', 0.6)
        
        logger.info(f"信号生成器初始化完成 - IV低阈值:{self.iv_rank_threshold_low}, IV高阈值:{self.iv_rank_threshold_high}")
    
    def generate_signals(self, volatility_data: Dict[str, Dict]) -> List[TradingSignal]:
        """
        生成交易信号
        
        Args:
            volatility_data: 波动率数据 {symbol: {iv_rank, hv_rank, iv, hv, ...}}
        
        Returns:
            交易信号列表
        """
        signals = []
        
        for symbol, data in volatility_data.items():
            try:
                signal = self._generate_single_signal(symbol, data)
                if signal and signal.strength >= self.min_signal_strength:
                    signals.append(signal)
                    logger.info(f"生成信号: {symbol} {signal.signal_type} 强度{signal.strength:.2f}")
            except Exception as e:
                logger.error(f"生成{symbol}信号失败: {e}")
        
        logger.info(f"共生成{len(signals)}个有效信号")
        return signals
    
    def _generate_single_signal(self, symbol: str, data: Dict) -> Optional[TradingSignal]:
        """
        生成单个品种的交易信号
        
        Args:
            symbol: 合约代码
            data: 波动率数据
        
        Returns:
            交易信号（如果有效）
        """
        iv_rank = data.get('iv_rank', 50)
        hv_rank = data.get('hv_rank', 50)
        iv = data.get('iv', 0)
        hv = data.get('hv', 0)
        current_price = data.get('close', 0)
        
        if current_price == 0:
            return None
        
        # 计算IV-HV差值
        iv_hv_diff = iv - hv
        iv_hv_diff_pct = iv_hv_diff / hv if hv > 0 else 0
        
        signal_type = None
        direction = None
        strength = 0
        reason = ""
        
        # 策略逻辑
        # 1. 低IV + HV相对稳定 → 做多波动率（买入期权）
        if (iv_rank < self.iv_rank_threshold_low and 
            abs(iv_hv_diff_pct) < self.iv_hv_spread_threshold):
            
            signal_type = "LONG_VOL"
            direction = "BUY"
            strength = self._calculate_signal_strength(
                iv_rank=iv_rank,
                hv_rank=hv_rank,
                iv_hv_diff=iv_hv_diff_pct,
                signal_type="long"
            )
            reason = f"IV处于低位({iv_rank:.1f}%)，HV稳定，适合做多波动率"
        
        # 2. 高IV + IV显著高于HV → 做空波动率（卖出期权）
        elif (iv_rank > self.iv_rank_threshold_high and 
              iv_hv_diff_pct > self.iv_hv_spread_threshold):
            
            signal_type = "SHORT_VOL"
            direction = "SELL"
            strength = self._calculate_signal_strength(
                iv_rank=iv_rank,
                hv_rank=hv_rank,
                iv_hv_diff=iv_hv_diff_pct,
                signal_type="short"
            )
            reason = f"IV处于高位({iv_rank:.1f}%)，IV显著高于HV({iv_hv_diff_pct:.2%})，适合做空波动率"
        
        # 3. IV与HV差异过大 → 波动率回归交易
        elif abs(iv_hv_diff_pct) > self.iv_hv_spread_threshold * 2:
            
            if iv > hv:  # IV高估
                signal_type = "SHORT_VOL"
                direction = "SELL"
                strength = 0.5
                reason = f"IV显著高于HV({iv_hv_diff_pct:.2%})，预期波动率回归"
            else:  # IV低估
                signal_type = "LONG_VOL"
                direction = "BUY"
                strength = 0.5
                reason = f"IV显著低于HV({iv_hv_diff_pct:.2%})，预期波动率回归"
        
        # 无有效信号
        if not signal_type:
            return None
        
        # 计算止损止盈
        stop_loss, take_profit = self._calculate_risk_levels(
            current_price=current_price,
            iv=iv,
            signal_type=signal_type
        )
        
        # 创建信号对象
        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            direction=direction,
            strength=strength,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            iv_rank=iv_rank,
            hv_rank=hv_rank,
            iv_hv_diff=iv_hv_diff,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            reason=reason
        )
        
        return signal
    
    def _calculate_signal_strength(self, iv_rank: float, hv_rank: float, 
                                   iv_hv_diff: float, signal_type: str) -> float:
        """
        计算信号强度
        
        Args:
            iv_rank: IV百分位
            hv_rank: HV百分位
            iv_hv_diff: IV-HV差值百分比
            signal_type: 信号类型
        
        Returns:
            信号强度（0-1）
        """
        strength = 0.0
        
        if signal_type == "long":
            # 做多波动率：IV越低，信号越强
            iv_score = (self.iv_rank_threshold_low - iv_rank) / self.iv_rank_threshold_low
            iv_score = max(0, min(1, iv_score))
            
            # HV稳定性加分
            hv_score = 1 - abs(hv_rank - 50) / 50
            
            strength = (iv_score * 0.6 + hv_score * 0.4)
        
        elif signal_type == "short":
            # 做空波动率：IV越高，信号越强
            iv_score = (iv_rank - self.iv_rank_threshold_high) / (100 - self.iv_rank_threshold_high)
            iv_score = max(0, min(1, iv_score))
            
            # IV-HV差异加分
            diff_score = min(1, abs(iv_hv_diff) / 0.15)
            
            strength = (iv_score * 0.7 + diff_score * 0.3)
        
        return round(strength, 2)
    
    def _calculate_risk_levels(self, current_price: float, iv: float, 
                               signal_type: str) -> tuple:
        """
        计算止损止盈价位
        
        Args:
            current_price: 当前价格
            iv: 隐含波动率
            signal_type: 信号类型
        
        Returns:
            (止损价, 止盈价)
        """
        # 基于波动率计算风险水平
        # 假设1个标准差的价格变动
        price_std = current_price * iv / np.sqrt(252)  # 日波动
        
        if signal_type == "LONG_VOL":
            # 做多波动率：止损设为-2σ，止盈设为+3σ
            stop_loss = current_price - 2 * price_std
            take_profit = current_price + 3 * price_std
        else:
            # 做空波动率：止损设为+2σ，止盈设为-3σ
            stop_loss = current_price + 2 * price_std
            take_profit = current_price - 3 * price_std
        
        return round(stop_loss, 2), round(take_profit, 2)


# 导入numpy（如果可用）
try:
    import numpy as np
except ImportError:
    # 简单实现sqrt函数
    import math
    class np:
        @staticmethod
        def sqrt(x):
            return math.sqrt(x)


if __name__ == "__main__":
    # 测试信号生成器
    test_data = {
        "sc2604": {
            "iv_rank": 25,
            "hv_rank": 45,
            "iv": 0.28,
            "hv": 0.26,
            "close": 550.0
        },
        "bu2606": {
            "iv_rank": 75,
            "hv_rank": 50,
            "iv": 0.35,
            "hv": 0.28,
            "close": 3800.0
        }
    }
    
    config = {
        'iv_rank_threshold_low': 30,
        'iv_rank_threshold_high': 70,
        'iv_hv_spread_threshold': 0.05
    }
    
    generator = SignalGenerator(config)
    signals = generator.generate_signals(test_data)
    
    for signal in signals:
        print(f"\n{signal.symbol}: {signal.signal_type}")
        print(f"  方向: {signal.direction}, 强度: {signal.strength}")
        print(f"  入场: {signal.entry_price}, 止损: {signal.stop_loss}, 止盈: {signal.take_profit}")
        print(f"  理由: {signal.reason}")
