"""
东方财富 科创50ETF期权数据抓取脚本
在你本地电脑上运行，抓取历史数据
"""
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import os

def fetch_option_history(contract_code, days=365):
    """
    获取单个期权合约的历史日线数据
    
    参数:
        contract_code: 期权合约代码，如 '10010037'
        days: 获取多少天的历史数据
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = f"http://push2his.eastmoney.com/api/qt/stock/kline/get"
    
    params = {
        'secid': f'10.{contract_code}',
        'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '101',  # 101=日线
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
            
            records = []
            for line in klines:
                # 格式: 日期,开盘价,收盘价,最高价,最低价,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
                parts = line.split(',')
                records.append({
                    '日期': parts[0],
                    '开盘价': float(parts[1]),
                    '收盘价': float(parts[2]),
                    '最高价': float(parts[3]),
                    '最低价': float(parts[4]),
                    '成交量': int(float(parts[5])),
                    '成交额': float(parts[6]),
                    '振幅': float(parts[7]),
                    '涨跌幅': float(parts[8]),
                    '涨跌额': float(parts[9]),
                    '换手率': float(parts[10]) if len(parts) > 10 else 0,
                })
            
            df = pd.DataFrame(records)
            return df
        else:
            return None
            
    except Exception as e:
        print(f"获取 {contract_code} 数据失败: {e}")
        return None


def fetch_option_list():
    """获取科创50ETF期权合约列表"""
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
        'fields': 'f12,f13,f14,f8,f9,f10,f11,f12,f13,f14'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://quote.eastmoney.com/'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        data = response.json()
        
        options = []
        if data.get('data') and data['data'].get('diff'):
            for item in data['data']['diff']:
                name = item.get('f14', '')
                code = item.get('f12', '')
                # 筛选科创50相关期权
                if '科创50' in name:
                    options.append({
                        '代码': code,
                        '名称': name,
                        '行权价': item.get('f8', 0) / 10000 if item.get('f8') else 0,
                        '剩余天数': item.get('f9', 0),
                    })
        
        return options
        
    except Exception as e:
        print(f"获取合约列表失败: {e}")
        return []


def fetch_588000_etf_history(days=730):
    """获取科创50ETF(588000)历史数据"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    
    params = {
        'secid': '1.588000',
        'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '101',
        'fqt': '1',  # 前复权
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
            
            records = []
            for line in klines:
                parts = line.split(',')
                records.append({
                    '日期': parts[0],
                    '开盘价': float(parts[1]),
                    '收盘价': float(parts[2]),
                    '最高价': float(parts[3]),
                    '最低价': float(parts[4]),
                    '成交量': int(float(parts[5])),
                    '成交额': float(parts[6]),
                    '振幅': float(parts[7]),
                    '涨跌幅': float(parts[8]),
                    '涨跌额': float(parts[9]),
                })
            
            df = pd.DataFrame(records)
            return df
        else:
            return None
            
    except Exception as e:
        print(f"获取ETF数据失败: {e}")
        return None


def main():
    """主函数"""
    print("="*60)
    print("东方财富 科创50ETF期权数据抓取工具")
    print("="*60)
    
    # 创建输出目录
    output_dir = f"科创50ETF期权数据_{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 获取ETF现货数据（最长时间：2年）
    print("\n【1】获取科创50ETF(588000)现货历史数据...")
    etf_data = fetch_588000_etf_history(days=730)
    if etf_data is not None:
        etf_file = os.path.join(output_dir, "科创50ETF_588000_日线数据.xlsx")
        etf_data.to_excel(etf_file, index=False)
        print(f"   获取到 {len(etf_data)} 条数据，已保存: {etf_file}")
    else:
        print("   获取失败")
    
    # 2. 获取期权合约列表
    print("\n【2】获取科创50ETF期权合约列表...")
    option_list = fetch_option_list()
    print(f"   找到 {len(option_list)} 个科创50ETF期权合约")
    
    if option_list:
        # 保存合约列表
        list_file = os.path.join(output_dir, "科创50ETF期权合约列表.xlsx")
        pd.DataFrame(option_list).to_excel(list_file, index=False)
        print(f"   合约列表已保存: {list_file}")
        
        # 3. 获取每个合约的历史数据
        print(f"\n【3】获取期权合约历史数据（每个合约最多365天）...")
        
        for i, opt in enumerate(option_list[:20]):  # 先抓前20个合约
            print(f"   [{i+1}/{min(20, len(option_list))}] {opt['名称']} ({opt['代码']})...", end=" ")
            
            hist_data = fetch_option_history(opt['代码'], days=365)
            
            if hist_data is not None and not hist_data.empty:
                hist_file = os.path.join(output_dir, f"{opt['代码']}_{opt['名称']}.xlsx")
                hist_data.to_excel(hist_file, index=False)
                print(f"成功 ({len(hist_data)}条)")
            else:
                print("失败或无数据")
            
            time.sleep(0.5)  # 避免请求过快
    
    print("\n" + "="*60)
    print(f"数据抓取完成！文件保存在: {output_dir}/")
    print("="*60)
    
    # 生成数据说明文件
    readme = f"""科创50ETF期权数据抓取结果
抓取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

文件说明:
1. 科创50ETF_588000_日线数据.xlsx - 现货ETF历史行情（前复权）
2. 科创50ETF期权合约列表.xlsx - 所有期权合约基本信息
3. [代码]_[名称].xlsx - 各期权合约历史日线数据

数据字段说明:
- 日期: 交易日期
- 开盘价/收盘价/最高价/最低价: 价格数据（元）
- 成交量: 成交数量
- 成交额: 成交金额（元）
- 涨跌幅: 当日涨跌百分比
- 涨跌额: 当日涨跌金额
"""
    
    readme_file = os.path.join(output_dir, "数据说明.txt")
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme)
    
    print(f"\n请把 '{output_dir}' 文件夹压缩后发给我")
    print("我会用这些数据跑策略回测")


if __name__ == '__main__':
    main()
