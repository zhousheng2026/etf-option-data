"""
波动率分析模块
计算历史波动率、隐含波动率、波动率分位数等
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime, timedelta


class VolatilityAnalyzer:
    """波动率分析器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化波动率分析器
        
        Args:
            config: 波动率配置
        """
        self.config = config or {}
        self.hv_window = self.config.get('hv_window', 20)
        
    def calculate_historical_volatility(
        self, 
        prices: pd.Series,
        window: Optional[int] = None,
        method: str = 'close_to_close'
    ) -> pd.Series:
        """
        计算历史波动率
        
        Args:
            prices: 价格序列（收盘价）
            window: 窗口大小
            method: 计算方法
            
        Returns:
            历史波动率序列
        """
        window = window or self.hv_window
        
        if method == 'close_to_close':
            # 收盘价-收盘价方法
            returns = prices.pct_change()
            hv = returns.rolling(window=window).std() * np.sqrt(252)
            
        elif method == 'parkinson':
            # Parkinson方法（需要高低价）
            if not isinstance(prices, pd.DataFrame):
                raise ValueError("Parkinson方法需要DataFrame包含high和low列")
            hl_ratio = np.log(prices['high'] / prices['low'])
            hv = np.sqrt(hl_ratio ** 2 / (4 * np.log(2))) \
                 .rolling(window=window).mean() * np.sqrt(252)
                 
        elif method == 'garman_klass':
            # Garman-Klass方法（需要开高低收）
            if not isinstance(prices, pd.DataFrame):
                raise ValueError("Garman-Klass方法需要DataFrame包含open、high、low、close列")
            hl_ratio = np.log(prices['high'] / prices['low'])
            co_ratio = np.log(prices['close'] / prices['open'])
            hv = np.sqrt(
                0.5 * hl_ratio ** 2 - (2 * np.log(2) - 1) * co_ratio ** 2
            ).rolling(window=window).mean() * np.sqrt(252)
            
        else:
            raise ValueError(f"不支持的方法: {method}")
            
        return hv
        
    def calculate_implied_volatility(
        self,
        option_price: float,
        underlying_price: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float = 0.03,
        option_type: str = 'call',
        method: str = 'newton'
    ) -> Optional[float]:
        """
        计算隐含波动率
        
        Args:
            option_price: 期权价格
            underlying_price: 标的价格
            strike: 行权价
            time_to_expiry: 到期时间（年）
            risk_free_rate: 无风险利率
            option_type: 期权类型
            method: 计算方法
            
        Returns:
            隐含波动率
        """
        from scipy.stats import norm
        from scipy.optimize import brentq
        
        def black_scholes_call(S, K, T, r, sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            
        def black_scholes_put(S, K, T, r, sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
        def price_diff(sigma):
            if option_type == 'call':
                return black_scholes_call(underlying_price, strike, time_to_expiry, 
                                         risk_free_rate, sigma) - option_price
            else:
                return black_scholes_put(underlying_price, strike, time_to_expiry,
                                        risk_free_rate, sigma) - option_price
                
        try:
            # 使用Brent方法求解
            iv = brentq(price_diff, 0.001, 5.0)
            return iv
        except Exception as e:
            print(f"隐含波动率计算失败: {e}")
            return None
            
    def calculate_volatility_percentile(
        self,
        volatility: pd.Series,
        window: int = 252
    ) -> pd.Series:
        """
        计算波动率分位数
        
        Args:
            volatility: 波动率序列
            window: 滚动窗口
            
        Returns:
            分位数序列（0-100）
        """
        return volatility.rolling(window=window).rank(pct=True) * 100
        
    def calculate_iv_hv_ratio(
        self,
        iv: pd.Series,
        hv: pd.Series
    ) -> pd.Series:
        """
        计算IV/HV比率
        
        Args:
            iv: 隐含波动率
            hv: 历史波动率
            
        Returns:
            比率序列
        """
        return iv / hv
        
    def detect_volatility_regime(
        self,
        volatility: pd.Series,
        low_threshold: float = 20,
        high_threshold: float = 80
    ) -> pd.Series:
        """
        检测波动率状态
        
        Args:
            volatility: 波动率序列
            low_threshold: 低波动阈值
            high_threshold: 高波动阈值
            
        Returns:
            状态序列（'low', 'normal', 'high'）
        """
        percentile = self.calculate_volatility_percentile(volatility)
        
        regime = pd.Series('normal', index=volatility.index)
        regime[percentile < low_threshold] = 'low'
        regime[percentile > high_threshold] = 'high'
        
        return regime
        
    def generate_volatility_report(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        生成波动率报告
        
        Args:
            data: 各品种的价格数据
            
        Returns:
            波动率报告字典
        """
        report = {}
        
        for symbol, df in data.items():
            if df.empty:
                continue
                
            # 计算历史波动率
            hv = self.calculate_historical_volatility(df['close'])
            
            # 计算分位数
            percentile = self.calculate_volatility_percentile(hv)
            
            # 检测状态
            regime = self.detect_volatility_regime(hv)
            
            report[symbol] = {
                'current_hv': hv.iloc[-1],
                'hv_percentile': percentile.iloc[-1],
                'regime': regime.iloc[-1],
                'hv_20d_avg': hv.rolling(20).mean().iloc[-1],
                'hv_trend': '上升' if hv.iloc[-1] > hv.iloc[-20] else '下降'
            }
            
        return report


def test_volatility_analyzer():
    """测试波动率分析器"""
    import random
    
    print("=" * 60)
    print("波动率分析器测试")
    print("=" * 60)
    
    # 创建测试数据
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    prices = pd.Series(
        [100 + i * 0.5 + random.uniform(-2, 2) for i in range(100)],
        index=dates
    )
    
    # 创建分析器
    analyzer = VolatilityAnalyzer()
    
    # 计算历史波动率
    print("\n计算历史波动率...")
    hv = analyzer.calculate_historical_volatility(prices)
    print(f"当前HV: {hv.iloc[-1]:.4f}")
    
    # 计算分位数
    print("\n计算波动率分位数...")
    percentile = analyzer.calculate_volatility_percentile(hv)
    print(f"当前分位数: {percentile.iloc[-1]:.2f}")
    
    # 检测状态
    print("\n检测波动率状态...")
    regime = analyzer.detect_volatility_regime(hv)
    print(f"当前状态: {regime.iloc[-1]}")
    
    # 测试隐含波动率计算
    print("\n计算隐含波动率...")
    iv = analyzer.calculate_implied_volatility(
        option_price=5.0,
        underlying_price=100,
        strike=100,
        time_to_expiry=0.25,
        option_type='call'
    )
    print(f"隐含波动率: {iv:.4f}")
    
    print("\n✅ 测试完成")


if __name__ == '__main__':
    test_volatility_analyzer()
