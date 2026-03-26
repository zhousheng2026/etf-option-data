# 波动率突破策略 Volatility Breakthrough Strategy

## 📋 策略概述

这是一个基于波动率分析的商品期权交易策略，核心思想是：

- **低IV时做多波动率**：当隐含波动率处于历史低位，且相对稳定时，买入期权
- **高IV时做空波动率**：当隐含波动率处于历史高位，且显著高于历史波动率时，卖出期权

## 🎯 策略参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `iv_rank_threshold_low` | 30 | IV低位阈值（百分位） |
| `iv_rank_threshold_high` | 70 | IV高位阈值（百分位） |
| `iv_hv_spread_threshold` | 0.05 | IV-HV差异阈值 |
| `min_signal_strength` | 0.6 | 最小信号强度 |

## 📁 文件结构

```
strategies/
└── volatility_breakthrough.py    # 主策略入口

config/
├── ctp_config.py                 # CTP连接配置
├── strategy_config.py            # 策略参数配置
└── symbols_config.py             # 监控品种配置

core/
├── ctp_connection.py             # CTP连接模块
├── volatility_analyzer.py        # 波动率分析
├── signal_generator.py           # 信号生成
└── report_generator.py           # 报告生成
```

## 🚀 运行方式

### 本地运行

```bash
# 安装依赖
pip install numpy pandas reportlab

# 运行策略
python strategies/volatility_breakthrough.py
```

### GitHub Actions自动运行

- **日盘**：9:00-15:00，每小时运行一次（北京时间）
- **夜盘**：21:00-01:00，每小时运行一次（北京时间）
- **手动触发**：支持workflow_dispatch

## 📊 报告输出

- **PDF报告**：每日收盘后生成，包含市场概况、交易信号、波动率数据
- **日志文件**：记录策略运行过程和错误信息
- **GitHub Artifacts**：保留30天历史报告

## 🔧 配置修改

### 修改CTP连接参数

编辑 `config/ctp_config.py`:

```python
CTP_CONFIG = {
    'broker_id': '6000',
    'investor_id': '00001920',
    'password': 'aa888888',
    # ...
}
```

### 修改策略参数

编辑 `config/strategy_config.py`:

```python
STRATEGY_CONFIG = {
    'iv_rank_threshold_low': 30,
    'iv_rank_threshold_high': 70,
    # ...
}
```

### 修改监控品种

编辑 `config/symbols_config.py`:

```python
SYMBOLS_CONFIG = {
    'symbols': ['sc2604', 'bu2606', ...]
}
```

## ⚠️ 注意事项

1. **测试环境**：当前使用CTP主席测试环境
2. **模拟数据**：首次运行使用模拟数据进行测试
3. **实盘连接**：需要配置真实的CTP账号和密码
4. **风险提示**：期权交易存在高风险，请谨慎操作

## 📝 更新日志

- **v1.0.0** (2026-03-27): 初始版本
  - 实现波动率分析模块
  - 实现信号生成器
  - 实现PDF报告生成
  - 集成GitHub Actions自动运行

## 📧 联系方式

- **作者**: zhousheng2026
- **项目**: etf-option-data
- **分支**: main
