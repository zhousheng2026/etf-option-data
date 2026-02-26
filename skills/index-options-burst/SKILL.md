---
name: index-options-burst
description: 股指期权趋势爆发策略回测与执行。用于基于30分钟K线趋势通道突破的股指期权策略回测、胜率统计、资金曲线分析。当用户需要测试股指期权策略、计算历史胜率、分析资金曲线或优化通道参数时触发。
---

# 股指期权趋势爆发策略

## 策略核心

- **周期**: 30分钟K线
- **入场**: 趋势通道/平台突破
- **离场**: 通道跌破
- **合约**: 虚值2-3档期权
- **原则**: 只做买方（买入认购/认沽），不做卖方

## 策略版本

### 1. 通道突破策略（主策略）

**买入条件**: 30分钟K线突破趋势通道上轨 + 成交量放大 + K线实体放大
**卖出条件**: 价格跌破通道中轨

| 版本 | 文件路径 | 说明 |
|------|----------|------|
| 双向版本 | `scripts/backtest.py` | 做多+做空 |
| 纯做多 | `scripts/backtest_long_only.py` | 只做多（推荐）|
| 日线版本 | `scripts/backtest_v1_daily.py` | 日线周期 |

**回测结果（纯做多）**:
| 指标 | 数值 |
|------|------|
| 胜率 | 55.56% |
| 盈亏比 | 2.46 |
| 做多胜率 | 56% |
| 做空胜率 | 41.67% |

### 2. 三重底背离策略（新策略）

**买入条件**: 布林带底背离 + MACD底背离 + 价格底背离（至少2种同时出现）
**卖出条件**: 30分钟通道跌破中轨

| 版本 | 文件路径 | 说明 |
|------|----------|------|
| 本地回测 | `scripts/triple_divergence_strategy.py` | 主程序 |
| 聚宽版本 | `scripts/聚宽_三重底背离策略.py` | JoinQuant平台 |
| 米筐版本 | `scripts/ricequant_triple_divergence.py` | RiceQuant平台 |

**回测结果（模拟数据）**:
| 指标 | 数值 |
|------|------|
| 胜率 | 54.55% |
| 盈亏比 | 3.23 |
| 总收益率 | 25.42% |
| 最大回撤 | 12.63% |

## 多平台支持

### 已开发平台版本

| 平台 | 状态 | 文件 | 网址 |
|------|------|------|------|
| **本地回测** | ✅ 可用 | `scripts/backtest.py` | - |
| **聚宽(JoinQuant)** | ⏳ 待实名认证 | `scripts/聚宽策略_30分钟通道突破.py` | https://www.joinquant.com |
| **米筐(RiceQuant)** | ✅ 可用 | `scripts/ricequant_strategy.py` | https://www.ricequant.com |
| **同花顺期货通** | ✅ 可用 | `tonghuashun_option_trading.py` | https://www.10jqka.com.cn |
| **CTP接口** | ⏳ 待确认服务器 | `ctp_option_trading.py` | - |
| **AKShare** | ⚠️ 网络受限 | `scripts/backtest_akshare.py` | - |

### 推荐优先级

1. **米筐(RiceQuant)** - 首选，支持期权回测，无需本地数据
2. **聚宽(JoinQuant)** - 次选，需要实名认证
3. **同花顺期货通** - 适合手动交易验证
4. **CTP接口** - 最终实盘方案，需确认券商服务器

## 快速开始

### 本地回测
```bash
# 通道突破策略
python scripts/backtest.py --symbol 000300 --start 2020-01-01 --end 2024-12-31 --capital 1000000

# 三重底背离策略
python scripts/triple_divergence_strategy.py --symbol 588000 --capital 1000000
```

### 聚宽平台
1. 访问 https://www.joinquant.com
2. 注册并完成实名认证
3. 上传策略文件：`scripts/聚宽策略_30分钟通道突破.py`
4. 设置回测参数并运行

### 米筐平台
1. 访问 https://www.ricequant.com
2. 注册账号
3. 上传策略文件：`scripts/ricequant_strategy.py`
4. 运行回测或模拟交易

## 标的配置

所有策略监控以下7个ETF期权标的（按优先级排序）：

| 优先级 | 代码 | 名称 | 说明 |
|--------|------|------|------|
| 1 | 588000 | 科创50ETF | 首选，科技股，国家队干扰少 |
| 2 | 588080 | 科创板50ETF | 备选 |
| 3 | 510500 | 中证500ETF | 中小盘 |
| 4 | 159845 | 中证1000ETF | 小盘 |
| 5 | 159915 | 创业板ETF | 创业板 |
| 6 | 510050 | 50ETF | 国家队护盘，趋势失真 |
| 7 | 510300 | 300ETF | 国家队护盘，放最后 |

## 策略参数

### 通道突破策略
```python
channel_bars = 20          # 通道周期
volume_threshold = 1.3     # 成交量阈值
body_threshold = 1.5       # K线实体阈值
iv_rank_long_max = 55      # 做多IV Rank上限
iv_rank_short_max = 65     # 做空IV Rank上限
stop_loss_1 = 0.30         # 第一止损位
stop_loss_2 = 0.50         # 第二止损位
```

### 三重底背离策略
```python
bb_period = 20             # 布林带周期
bb_std = 2.0               # 布林带标准差
macd_fast = 12             # MACD快线
macd_slow = 26             # MACD慢线
macd_signal = 9            # MACD信号线
divergence_lookback = 30   # 背离检测回溯周期
min_divergence_bars = 5    # 最小背离间距
```

## 时间过滤

- **9:30-10:00**: 不入场（开盘波动大）
- **14:30-15:00**: 不开新仓（收盘前）

## 风险管理

- **只做买方**: 买入认购/认沽期权，风险可控（最大亏损权利金）
- **不做卖方**: 坚决禁止卖出开仓，风险无限
- **仓位控制**: 2种背离信号30%仓位，3种背离信号50%仓位
- **止损**: 期权价格跌30%减仓50%，跌50%清仓

## 账号信息

| 平台 | 账号 | 状态 |
|------|------|------|
| 聚宽 | 13803545165 | 待实名认证 |
| 光大证券CTP | 43503750 | 待确认交易服务器 |

## 下一步行动

1. [ ] 完成聚宽实名认证
2. [ ] 确认光大证券CTP交易服务器和BrokerID（打客服95525）
3. [ ] 申请JQData SDK进行本地真实数据回测
4. [ ] 在米筐平台进行三重底背离策略回测
5. [ ] 选择最终自动下单平台

## 参考文件

- `scripts/channel.py` - 通道绘制算法
- `references/strategy_params.md` - 策略参数详解
- `references/backtest_results.md` - 回测结果汇总
- `memory/期权交易原则_只做买方.md` - 交易原则
