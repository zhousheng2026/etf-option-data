# ETF期权数据抓取 - GitHub部署指南

## 快速部署步骤

### 1. 创建GitHub仓库
1. 访问 https://github.com/new
2. 仓库名填写：`etf-option-data`
3. 选择 **Public**（公开）
4. 点击 **Create repository**

### 2. 上传代码文件
在新建的仓库页面：
1. 点击 **"uploading an existing file"**
2. 上传以下文件（从本目录复制）：
   - `.github/workflows/fetch_etf_data.yml`
   - `scripts/github_fetch_etf_data.py`
   - `scripts/github_fetch_option_data.py`

### 3. 启用GitHub Actions
1. 点击仓库顶部的 **Actions** 标签
2. 点击绿色按钮 **"I understand my workflows, go ahead and enable them"**

### 4. 运行工作流
1. 在Actions页面，点击 **"抓取ETF和期权数据"**
2. 点击 **"Run workflow"** → **"Run workflow"**
3. 等待运行完成（约2-3分钟）

### 5. 查看数据
运行完成后，在仓库的 `data/` 目录下查看：
- `latest_signals.json` - ETF信号
- `options/etf_options_latest.csv` - 期权数据

## 自动运行
部署后，GitHub Actions会自动：
- 交易时间每30分钟抓取一次
- 每天晚上22:00备份

## 数据访问链接
部署后，数据可通过以下链接访问：
```
https://raw.githubusercontent.com/你的用户名/etf-option-data/master/data/latest_signals.json
https://raw.githubusercontent.com/你的用户名/etf-option-data/master/data/options/etf_options_latest.json
```
