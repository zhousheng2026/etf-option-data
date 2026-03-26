"""
策略参数配置文件
包含波动率突破策略的所有参数
"""

# ==================== 波动率参数 ====================

VOLATILITY_CONFIG = {
    # 历史波动率计算
    'hv_window': 20,  # 历史波动率窗口（天）
    'hv_type': 'close_to_close',  # 波动率类型：close_to_close, parkinson, garman_klass
    
    # 隐含波动率计算
    'iv_window': 20,  # 隐含波动率均线窗口
    
    # 波动率比率
    'iv_hv_ratio_threshold_low': 0.8,  # IV/HV低于此值做多波动率
    'iv_hv_ratio_threshold_high': 1.3,  # IV/HV高于此值做空波动率
    
    # 波动率分位数
    'iv_percentile_low': 20,  # IV分位数低于此值做多波动率
    'iv_percentile_high': 80,  # IV分位数高于此值做空波动率
}

# ==================== 交易信号参数 ====================

SIGNAL_CONFIG = {
    # 信号阈值
    'signal_threshold': 0.7,  # 信号强度阈值（0-1）
    'min_hold_period': 5,  # 最小持仓周期（天）
    
    # 止损止盈
    'stop_loss_pct': 0.05,  # 止损比例（5%）
    'take_profit_pct': 0.15,  # 止盈比例（15%）
    
    # 仓位管理
    'max_position_size': 10,  # 单品种最大持仓手数
    'total_position_limit': 50,  # 总持仓限制（手）
    'margin_usage_limit': 0.3,  # 保证金使用上限（30%）
}

# ==================== 合约选择参数 ====================

CONTRACT_CONFIG = {
    # 期权合约选择
    'delta_range': (0.3, 0.7),  # Delta范围
    'min_days_to_expiry': 10,  # 最小到期天数
    'max_days_to_expiry': 60,  # 最大到期天数
    'min_volume': 100,  # 最小成交量
    'min_open_interest': 500,  # 最小持仓量
    
    # 执行价选择
    'moneyness_range': (0.9, 1.1),  # 平值程度范围
}

# ==================== 风控参数 ====================

RISK_CONFIG = {
    # 单笔风险
    'max_loss_per_trade': 5000,  # 单笔最大亏损（元）
    'max_loss_per_day': 15000,  # 单日最大亏损（元）
    'max_loss_per_month': 50000,  # 单月最大亏损（元）
    
    # 回撤控制
    'max_drawdown_pct': 0.2,  # 最大回撤（20%）
    
    # 波动率限制
    'max_iv_change': 0.1,  # 单日IV最大变化
    'max_hv_change': 0.05,  # 单日HV最大变化
}

# ==================== 报告参数 ====================

REPORT_CONFIG = {
    # 报告生成
    'report_dir': 'reports',
    'report_format': 'pdf',
    'report_language': 'zh_CN',  # 中文
    
    # 数据保存
    'save_trades': True,
    'save_signals': True,
    'save_volatility': True,
}
