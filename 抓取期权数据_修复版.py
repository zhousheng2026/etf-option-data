"""
东方财富 科创50ETF期权主力合约 30分钟数据抓取
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os


def safe_float(value, default=0):
    """安全转换为float"""
    if value is None or value == '' or value == '-':
        return default
    try:
        return float(value)
    except:
        return default


def safe_int(value, default=0):
    """安全转换为int"""
    if value is None or value == '' or value == '-':
        return default
    try:
        return int(float(value))
    except:
        return default


def fetch_option_contract_list():
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
        'fs': 'm:10',
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
                
                if '科创50' in name:
                    contracts.append({
                        '代码': code,
                        '名称': name,
                        '最新价': safe_float(item.get('f2')) / 10000,
                        '涨跌幅': safe_float(item.get('f3')) / 100,
                        '成交量': safe_int(item.get('f5')),
                        '成交额': safe_float(item.get('f6')),
                        '持仓量': safe_int(item.get('f7')),
                        '行权价': safe_float(item.get('f8')) / 10000,
                        '剩余天数': safe_int(item.get('f9')),
                        '隐含波动率': safe_float(item.get('f10')) / 100,
                    })
            
            print(f"  筛选出 {len(contracts)} 个科创50ETF期权")
            contracts.sort(key=lambda x: x['成交量'], reverse=True)
            return contracts
            
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
    
    return []


def fetch_option_30min_data(contract_code, contract_name, days=180):
    print(f"  正在获取 {contract_name} 的30分钟数据...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    
    params = {
        'secid': f'10.{contract_code}',
        'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '30',
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
            
            return pd.DataFrame(records)
            
    except Exception as e:
        print(f"    错误: {e}")
    
    return None


def main():
    print("="*70)
    print("东方财富 科创50ETF期权主力合约 30分钟数据抓取")
    print("="*70)
    
    output_dir = f"科创50ETF期权数据_{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}/")
    
    contracts = fetch_option_contract_list()
    
    if not contracts:
        print("没有获取到期权合约")
        input("按回车键退出...")
        return
    
    print(f"\n前10个主力合约:")
    for i, c in enumerate(contracts[:10], 1):
        print(f"{i}. {c['名称']} (成交量: {c['成交量']})")
    
    # 保存合约列表
    pd.DataFrame(contracts).to_excel(os.path.join(output_dir, "合约列表.xlsx"), index=False)
    
    # 获取前5个合约的数据
    print(f"\n开始获取前5个合约的30分钟数据...")
    for i, contract in enumerate(contracts[:5], 1):
        print(f"\n[{i}/5] {contract['名称']}")
        
        df = fetch_option_30min_data(contract['代码'], contract['名称'])
        
        if df is not None and not df.empty:
            file_name = f"{contract['代码']}_{contract['名称']}.xlsx"
            file_path = os.path.join(output_dir, file_name)
            df.to_excel(file_path, index=False)
            print(f"  已保存: {file_name} ({len(df)}条)")
        else:
            print(f"  获取失败")
        
        time.sleep(0.5)
    
    print(f"\n完成！文件夹: {output_dir}/")
    input("按回车键退出...")


if __name__ == '__main__':
    main()
