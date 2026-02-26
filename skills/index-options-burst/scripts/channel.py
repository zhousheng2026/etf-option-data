"""
趋势通道绘制算法
基于30分钟K线自动识别上涨/下跌通道或平台整理
"""
import numpy as np
from typing import Tuple, Optional, Literal
from dataclasses import dataclass


@dataclass
class Channel:
    """趋势通道"""
    type: Literal['ascending', 'descending', 'flat', 'none']
    upper_slope: float  # 上轨斜率
    upper_intercept: float
    lower_slope: float  # 下轨斜率
    lower_intercept: float
    upper_points: list  # 上轨连接点 [(idx, price), ...]
    lower_points: list  # 下轨连接点 [(idx, price), ...]
    touches_upper: int  # 触碰上轨次数
    touches_lower: int  # 触碰下轨次数
    
    def price_at(self, idx: int, line: str = 'upper') -> float:
        """计算指定位置的上轨/下轨价格"""
        if line == 'upper':
            return self.upper_slope * idx + self.upper_intercept
        return self.lower_slope * idx + self.lower_intercept
    
    def middle_at(self, idx: int) -> float:
        """计算通道中轨价格"""
        return (self.price_at(idx, 'upper') + self.price_at(idx, 'lower')) / 2
    
    def is_breakout_up(self, idx: int, close: float) -> bool:
        """判断是否突破上轨"""
        return close > self.price_at(idx, 'upper')
    
    def is_breakdown(self, idx: int, close: float) -> bool:
        """判断是否跌破下轨"""
        return close < self.price_at(idx, 'lower')


def find_pivots(highs: np.ndarray, lows: np.ndarray, window: int = 3) -> Tuple[list, list]:
    """
    寻找波段高低点
    
    Args:
        highs: 最高价数组
        lows: 最低价数组
        window: 确认窗口大小
    
    Returns:
        (高点列表, 低点列表) - 每个元素为 (索引, 价格)
    """
    high_pivots = []
    low_pivots = []
    
    for i in range(window, len(highs) - window):
        # 高点: 当前高点为窗口内最高
        if highs[i] == max(highs[i-window:i+window+1]):
            high_pivots.append((i, highs[i]))
        
        # 低点: 当前低点为窗口内最低
        if lows[i] == min(lows[i-window:i+window+1]):
            low_pivots.append((i, lows[i]))
    
    return high_pivots, low_pivots


def fit_line(points: list) -> Tuple[float, float]:
    """
    用最小二乘法拟合直线
    
    Returns:
        (斜率, 截距)
    """
    if len(points) < 2:
        return 0, points[0][1] if points else 0
    
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    
    # 最小二乘法
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    
    return slope, intercept


def calculate_slope_angle(slope: float, bar_interval: int = 30) -> float:
    """
    计算斜率角度
    
    Args:
        slope: 价格/索引 的斜率
        bar_interval: 每根K线分钟数
    """
    # 转换为每分钟的斜率，再算角度
    slope_per_minute = slope / bar_interval
    angle = np.degrees(np.arctan(slope_per_minute))
    return angle


def count_touches(prices: np.ndarray, line_slope: float, line_intercept: float, 
                  tolerance: float = 0.002) -> int:
    """
    统计价格触碰趋势线的次数
    
    Args:
        prices: 价格数组（高点或低点）
        line_slope: 趋势线斜率
        line_intercept: 趋势线截距
        tolerance: 触碰容差（默认0.2%）
    """
    touches = 0
    for i, price in enumerate(prices):
        line_price = line_slope * i + line_intercept
        if abs(price - line_price) / line_price < tolerance:
            touches += 1
    return touches


def draw_channel(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                 min_bars: int = 20, min_touches: int = 2,
                 flat_angle_threshold: float = 15.0,
                 flat_range_threshold: float = 0.02) -> Channel:
    """
    绘制趋势通道
    
    Args:
        highs: 最高价数组
        lows: 最低价数组
        closes: 收盘价数组
        min_bars: 最少K线数
        min_touches: 最少触碰次数
        flat_angle_threshold: 平台斜率阈值（度）
        flat_range_threshold: 平台波动范围阈值
    
    Returns:
        Channel对象
    """
    if len(highs) < min_bars:
        return Channel('none', 0, 0, 0, 0, [], [], 0, 0)
    
    # 使用最近min_bars根K线
    highs = highs[-min_bars:]
    lows = lows[-min_bars:]
    closes = closes[-min_bars:]
    
    # 寻找波段高低点
    high_pivots, low_pivots = find_pivots(highs, lows)
    
    if len(high_pivots) < 2 or len(low_pivots) < 2:
        # 尝试平台整理识别
        price_range = (np.max(highs) - np.min(lows)) / np.mean(closes)
        if price_range < flat_range_threshold:
            # 认定为平台
            upper_price = np.max(highs)
            lower_price = np.min(lows)
            return Channel(
                type='flat',
                upper_slope=0, upper_intercept=upper_price,
                lower_slope=0, lower_intercept=lower_price,
                upper_points=[(0, upper_price), (min_bars-1, upper_price)],
                lower_points=[(0, lower_price), (min_bars-1, lower_price)],
                touches_upper=sum(1 for h in highs if abs(h - upper_price)/upper_price < 0.005),
                touches_lower=sum(1 for l in lows if abs(l - lower_price)/lower_price < 0.005)
            )
        return Channel('none', 0, 0, 0, 0, [], [], 0, 0)
    
    # 尝试构建上涨通道: 上轨连接两个递增高
    ascending_upper = None
    for i in range(len(high_pivots)):
        for j in range(i+1, len(high_pivots)):
            if high_pivots[j][1] > high_pivots[i][1]:  # 第二个高点更高
                slope, intercept = fit_line([high_pivots[i], high_pivots[j]])
                angle = calculate_slope_angle(slope)
                if 0 < angle < 45:  # 合理的上涨角度
                    touches = count_touches(highs, slope, intercept)
                    if touches >= min_touches:
                        ascending_upper = (slope, intercept, [high_pivots[i], high_pivots[j]], touches)
                        break
        if ascending_upper:
            break
    
    # 尝试构建下跌通道: 下轨连接两个递减低
    descending_lower = None
    for i in range(len(low_pivots)):
        for j in range(i+1, len(low_pivots)):
            if low_pivots[j][1] < low_pivots[i][1]:  # 第二个低点更低
                slope, intercept = fit_line([low_pivots[i], low_pivots[j]])
                angle = calculate_slope_angle(slope)
                if -45 < angle < 0:  # 合理的下跌角度
                    touches = count_touches(lows, slope, intercept)
                    if touches >= min_touches:
                        descending_lower = (slope, intercept, [low_pivots[i], low_pivots[j]], touches)
                        break
        if descending_lower:
            break
    
    # 判断通道类型
    if ascending_upper:
        # 构建上涨通道，下轨平行
        upper_slope, upper_intercept, upper_pts, upper_touches = ascending_upper
        
        # 找与上轨平行的下轨（最低点的平行线）
        min_low_idx = np.argmin(lows)
        min_low_price = lows[min_low_idx]
        lower_intercept = min_low_price - upper_slope * min_low_idx
        lower_touches = count_touches(lows, upper_slope, lower_intercept)
        
        angle = calculate_slope_angle(upper_slope)
        if abs(angle) < flat_angle_threshold:
            channel_type = 'flat'
        else:
            channel_type = 'ascending'
        
        return Channel(
            type=channel_type,
            upper_slope=upper_slope, upper_intercept=upper_intercept,
            lower_slope=upper_slope, lower_intercept=lower_intercept,
            upper_points=upper_pts,
            lower_points=[(min_low_idx, min_low_price)],
            touches_upper=upper_touches,
            touches_lower=lower_touches
        )
    
    if descending_lower:
        # 构建下跌通道，上轨平行
        lower_slope, lower_intercept, lower_pts, lower_touches = descending_lower
        
        # 找与下轨平行的上轨（最高点的平行线）
        max_high_idx = np.argmax(highs)
        max_high_price = highs[max_high_idx]
        upper_intercept = max_high_price - lower_slope * max_high_idx
        upper_touches = count_touches(highs, lower_slope, upper_intercept)
        
        angle = calculate_slope_angle(lower_slope)
        if abs(angle) < flat_angle_threshold:
            channel_type = 'flat'
        else:
            channel_type = 'descending'
        
        return Channel(
            type=channel_type,
            upper_slope=lower_slope, upper_intercept=upper_intercept,
            lower_slope=lower_slope, lower_intercept=lower_intercept,
            upper_points=[(max_high_idx, max_high_price)],
            lower_points=lower_pts,
            touches_upper=upper_touches,
            touches_lower=lower_touches
        )
    
    # 无有效通道，检查是否为平台
    price_range = (np.max(highs) - np.min(lows)) / np.mean(closes)
    if price_range < flat_range_threshold:
        upper_price = np.max(highs)
        lower_price = np.min(lows)
        return Channel(
            type='flat',
            upper_slope=0, upper_intercept=upper_price,
            lower_slope=0, lower_intercept=lower_price,
            upper_points=[(0, upper_price), (min_bars-1, upper_price)],
            lower_points=[(0, lower_price), (min_bars-1, lower_price)],
            touches_upper=sum(1 for h in highs if abs(h - upper_price)/upper_price < 0.005),
            touches_lower=sum(1 for l in lows if abs(l - lower_price)/lower_price < 0.005)
        )
    
    return Channel('none', 0, 0, 0, 0, [], [], 0, 0)


if __name__ == '__main__':
    # 测试代码
    import matplotlib.pyplot as plt
    
    # 生成测试数据: 上涨通道
    np.random.seed(42)
    n = 40
    base = 4000
    trend = np.linspace(0, 100, n)
    noise = np.random.randn(n) * 20
    
    highs = base + trend + noise + 30
    lows = base + trend + noise - 30
    closes = base + trend + noise
    
    channel = draw_channel(highs, lows, closes)
    
    print(f"通道类型: {channel.type}")
    print(f"上轨斜率: {channel.upper_slope:.4f}")
    print(f"下轨斜率: {channel.lower_slope:.4f}")
    print(f"触碰上轨: {channel.touches_upper}次")
    print(f"触碰下轨: {channel.touches_lower}次")
    
    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(closes, 'b-', label='Close', alpha=0.7)
    plt.plot(highs, 'g-', alpha=0.3)
    plt.plot(lows, 'r-', alpha=0.3)
    
    x = np.arange(len(closes))
    plt.plot(x, channel.upper_slope * x + channel.upper_intercept, 'g--', label='Upper Channel')
    plt.plot(x, channel.lower_slope * x + channel.lower_intercept, 'r--', label='Lower Channel')
    
    plt.legend()
    plt.title(f'Channel Detection: {channel.type}')
    plt.show()
