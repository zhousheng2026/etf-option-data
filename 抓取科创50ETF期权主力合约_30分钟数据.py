"""
东方财富 科创50ETF期权主力合约 30分钟数据抓取
自动筛选成交量最大的主力合约
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

def fetch_option_contract_list():
    """获取科创50ETF期权所有合约列表"""
    print("正在获取科创50ETF期权合约列表...")
    
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    
    params = {
        'pn': 1,
        'pz': 500,
        'po': 1,
        'np': 1,
        'fltt': 2,
        'invt': 2,
        'fid': 'f12',
        'fs': 'm:10',  # 期权市场
        'fields': 'f12,f13,f14,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f20,f21'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://quote.eastmoney.com/'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        data = response.json()
        
        contracts = []
        if data.get('data') and data['data'].get('diff'):
            items = data['data']['diff']
            print(f"  获取到 {len(items)} 个期权合约")
            
            for item in items:
                name = item.get('f14', '')
                code = item.get('f12', '')
                
                # 筛选科创50相关期权
                if '科创50' in name:
                    contracts.append({
                        '代码': code,
                        '名称': name,
                        '最新价': float(item.get('f2', 0) or 0) / 10000,
                        '涨跌幅': float(item.get('f3', 0) or 0) / 100,
                        '成交量': int(item.get('f5', 0) or 0),
                        '成交额': float(item.get('f6', 0) or 0),
                        '持仓量': int(item.get('f7', 0) or 0),
                        '行权价': float(item.get('f8', 0) or 0) / 10000,
                        '剩余天数': int(item.get('f9', 0) or 0),
                        '隐含波动率': float(item.get('f10', 0) or 0) / 100,
                    })
            
            print(f"  筛选出 {len(contracts)} 个科创50ETF期权")
            
            # 按成交量排序
            contracts.sort(key=lambda x: x['成交量'], reverse=True)
            
            return contracts
        else:
            print(f"  警告: 没有获取到合约列表")
            
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
    
    return []


def fetch_option_30min_data(contract_code, contract_name, days=180):
    """
    获取指定期权合约的30分钟K线数据
    
    参数:
        contract_code: 期权合约代码，如 '10010037'
        contract_name: 期权合约名称
        days: 获取多少天的历史数据
    """
    print(f"  正在获取 {contract_name} ({contract_code}) 的30分钟数据...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    
    params = {
        'secid': f'10.{contract_code}',
        'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '30',   # 30分钟线
        'fqt': '0',
        'beg': start_date.strftime('%Y%m%d'),
        'end': end_date.strftime('%Y%m%d'),
        'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'http://quote.eastmoney.com/'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        data = response.json()
        
        if data.get('data') and data['data'].get('klines'):
            klines = data['data']['klines']
            
            if len(klines) == 0:
                print(f"    该合约没有历史数据（可能是新上市合约）")
                return None
            
            records = []
            for line in klines:
                parts = line.split(',')
                records.append({
                    '日期时间': parts[0],
                    '开盘价': float(parts[1]),
                    '收盘价': float(parts[2]),
                    '最高价': float(parts[3]),
                    '最低价': float(parts[4]),
                    '成交量': int(float(parts[5])),
                    '成交额': float(parts[6]),
                    '涨跌幅': float(parts[8]),
                })
            
            df = pd.DataFrame(records)
            print(f"    成功获取 {len(df)} 条30分钟K线")
            return df
        else:
            print(f"    无数据")
            
    except Exception as e:
        print(f"    错误: {e}")
    
    return None


def main():
    print("="*70)
    print("东方财富 科创50ETF期权主力合约 30分钟数据抓取")
    print("="*70)
    
    # 创建输出目录
    output_dir = f"科创50ETF期权主力合约_30分钟数据_{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}/")
    
    # 1. 获取期权合约列表
    print("\n" + "="*70)
    print("【1】获取科创50ETF期权合约列表并筛选主力合约")
    print("="*70)
    
    contracts = fetch_option_contract_list()
    
    if not contracts:
        print("\n✗ 没有获取到期权合约")
        input("按回车键退出...")
        return
    
    # 显示前20个主力合约
    print("\n主力合约排行（按成交量）:")
    print("-"*70)
    print(f"{'排名':<4} {'代码':<10} {'名称':<20} {'成交量':<12} {'持仓量':<12}")
    print("-"*70)
    
    for i, c in enumerate(contracts[:20], 1):
        print(f"{i:<4} {c['代码']:<10} {c['名称']:<20} {c['成交量']:<12} {c['持仓量']:<12}")
    
    # 保存合约列表
    list_file = os.path.join(output_dir, "期权合约列表.xlsx")
    pd.DataFrame(contracts).to_excel(list_file, index=False)
    print(f"\n✓ 合约列表已保存: {list_file}")
    
    # 2. 获取前10个主力合约的30分钟数据
    print("\n" + "="*70)
    print("【2】获取前10个主力合约的30分钟历史数据")
    print("="*70)
    
    top_contracts = contracts[:10]
    success_count = 0
    
    for i, contract in enumerate(top_contracts, 1):
        print(f"\n[{i}/10] {contract['名称']} (成交量: {contract['成交量']})")
        
        df = fetch_option_30min_data(
            contract['代码'], 
            contract['名称'],
            days=180  # 抓6个月数据
        )
        
        if df is not None and not df.empty:
            # 清理文件名
            safe_name = contract['名称'].replace('/', '-').replace('\\', '-')
            file_name = f"{contract['代码']}_{safe_name}_30分钟.xlsx"
            file_path = os.path.join(output_dir, file_name)
            
            df.to_excel(file_path, index=False)
            print(f"  ✓ 已保存: {file_name} ({len(df)}条)")
            success_count += 1
        else:
            print(f"  ✗ 获取失败")
        
        time.sleep(0.5)  # 避免请求过快
    
    # 3. 完成总结
    print("\n" + "="*70)
    print("抓取完成")
    print("="*70)
    
    files = os.listdir(output_dir)
    print(f"\n输出文件夹: {output_dir}/")
    print(f"成功获取: {success_count}/10 个主力合约")
    print(f"文件数量: {len(files)}")
    
    if files:
        print("\n文件列表:")
        for f in files:
            file_path = os.path.join(output_dir, f)
            size = os.path.getsize(file_path)
            print(f"  - {f} ({size/1024:.1f} KB)")
    
    # 生成数据说明
    readme = f"""科创50ETF期权主力合约 30分钟数据抓取结果
抓取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

数据说明:
- 数据来源: 东方财富
- 时间周期: 30分钟K线
- 数据范围: 约6个月历史数据
- 合约数量: {success_count}个主力合约

主力合约筛选标准:
按成交量排序，取前10名

数据字段:
- 日期时间: 交易日期和时间
- 开盘价/收盘价/最高价/最低价: 期权价格（元）
- 成交量: 成交张数
- 成交额: 成交金额（元）
- 涨跌幅: 涨跌幅百分比

用途:
用于30分钟周期的三重底背离策略回测
"""
    
    readme_file = os.path.join(output_dir, "数据说明.txt")
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme)
    
    print(f"\n请把 '{output_dir}' 文件夹压缩后发给我")
    input("\n按回车键退出...")


if __name__ == '__main__':
    main()
