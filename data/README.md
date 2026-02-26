# ETF期权数据抓取

使用GitHub Actions定时抓取ETF和期货数据，计算布林带突破信号。

## 数据文件

- `data/latest_signals.json` - 最新信号
- `data/history_YYYYMMDD.json` - 历史记录

## 信号说明

| 字段 | 说明 |
|------|------|
| `breakout_up` | 突破布林带上轨 |
| `breakout_down` | 突破布林带下轨 |
| `confirmed_up` | 确认突破上轨（连续2天） |
| `confirmed_down` | 确认突破下轨（连续2天） |
| `z_score` | 偏离中轨的标准差数 |

## 运行频率

- 交易时间每30分钟运行一次
- 每天晚上22:00备份
