"""
东方财富 科创50ETF期权数据抓取脚本 v2
增加错误处理和调试信息
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import json

def test_connection():
    """测试网络连接"""
    try:
        response = requests.get("https://www.eastmoney.com", timeout=10)
        print(f"网络连接正常，状态码: {response.status_code}")
        return True
    except Exception as e:
        print(f"网络连接失败: {e}")
        return False

def fetch_588000_etf_history(days=730):
    """获取科创50ETF(588000)历史数据"""
    print(f"\n正在获取科创50ETF(588000) {days}天历史数据...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    
    params = {
        'secid': '1.588000',
        'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '101',
        'fqt': '1',
        'beg': start_date.strftime('%Y%m%d'),
        'end': end_date.strftime('%Y%m%d'),
        'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'http://quote.eastmoney.com/'
    }
    
    try:
        print(f"  请求URL: {url}")
        print(f"  参数: {params}")
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        print(f"  响应状态: {response.status_code}")
        
        data = response.json()
        
        if data.get('data'):
            print(f"  数据键: {list(data['data'].keys())}")
            
            if data['data'].get('klines'):
                klines = data['data']['klines']
                print(f"  获取到 {len(klines)} 条K线数据")
                
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
                print(f"  成功创建DataFrame: {len(df)} 行")
                return df
            else:
                print(f"  警告: 没有klines数据")
                print(f"  data内容: {data['data']}")
        else:
            print(f"  警告: 没有data字段")
            print(f"  响应内容: {data}")
            
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
    
    return None


def fetch_option_list():
    """获取期权合约列表"""
    print("\n正在获取期权合约列表...")
    
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
        'fields': 'f12,f13,f14,f8,f9,f10,f11'
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
            items = data['data']['diff']
            print(f"  获取到 {len(items)} 个合约")
            
            for item in items:
                name = item.get('f14', '')
                code = item.get('f12', '')
                # 筛选科创50相关期权
                if '科创50' in name or '588000' in name:
                    options.append({
                        '代码': code,
                        '名称': name,
                        '行权价': item.get('f8', 0) / 10000 if item.get('f8') else 0,
                        '剩余天数': item.get('f9', 0),
                    })
            
            print(f"  筛选出 {len(options)} 个科创50ETF期权")
            return options
        else:
            print(f"  警告: 没有获取到合约列表")
            print(f"  响应: {data}")
            
    except Exception as e:
        print(f"  错误: {e}")
    
    return []


def fetch_option_history(contract_code, days=365):
    """获取单个期权合约的历史日线数据"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = f"http://push2his.eastmoney.com/api/qt/stock/kline/get"
    
    params = {
        'secid': f'10.{contract_code}',
        'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '101',
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
            
    except Exception as e:
        print(f"  获取 {contract_code} 失败: {e}")
    
    return None


def main():
    print("="*60)
    print("东方财富 科创50ETF期权数据抓取工具 v2")
    print("="*60)
    
    # 测试网络
    if not test_connection():
        print("\n网络连接有问题，请检查网络后重试")
        input("按回车键退出...")
        return
    
    # 创建输出目录
    output_dir = f"科创50ETF期权数据_{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}/")
    
    # 1. 获取ETF现货数据
    print("\n" + "="*60)
    print("【1】获取科创50ETF(588000)现货历史数据")
    print("="*60)
    
    etf_data = fetch_588000_etf_history(days=730)
    
    if etf_data is not None and not etf_data.empty:
        etf_file = os.path.join(output_dir, "科创50ETF_588000_日线数据.xlsx")
        etf_data.to_excel(etf_file, index=False)
        print(f"✓ 成功保存: {etf_file} ({len(etf_data)} 条)")
    else:
        print("✗ 获取ETF数据失败")
    
    # 2. 获取期权合约列表
    print("\n" + "="*60)
    print("【2】获取科创50ETF期权合约列表")
    print("="*60)
    
    option_list = fetch_option_list()
    
    if option_list:
        list_file = os.path.join(output_dir, "科创50ETF期权合约列表.xlsx")
        pd.DataFrame(option_list).to_excel(list_file, index=False)
        print(f"✓ 合约列表已保存: {list_file}")
        
        # 3. 获取每个合约的历史数据
        print("\n" + "="*60)
        print(f"【3】获取前10个期权合约历史数据")
        print("="*60)
        
        success_count = 0
        for i, opt in enumerate(option_list[:10]):
            print(f"\n[{i+1}/10] {opt['名称']} ({opt['代码']})")
            
            hist_data = fetch_option_history(opt['代码'], days=365)
            
            if hist_data is not None and not hist_data.empty:
                # 清理文件名中的非法字符
                safe_name = opt['名称'].replace('/', '-').replace('\\', '-')
                hist_file = os.path.join(output_dir, f"{opt['代码']}_{safe_name}.xlsx")
                hist_data.to_excel(hist_file, index=False)
                print(f"  ✓ 成功 ({len(hist_data)}条) -> {hist_file}")
                success_count += 1
            else:
                print(f"  ✗ 失败或无数据")
            
            time.sleep(0.5)  # 避免请求过快
        
        print(f"\n成功获取 {success_count}/10 个合约数据")
    else:
        print("✗ 没有获取到期权合约列表")
    
    # 检查输出
    print("\n" + "="*60)
    print("抓取完成")
    print("="*60)
    
    files = os.listdir(output_dir)
    print(f"\n输出文件夹: {output_dir}/")
    print(f"文件数量: {len(files)}")
    
    if files:
        print("\n文件列表:")
        for f in files:
            file_path = os.path.join(output_dir, f)
            size = os.path.getsize(file_path)
            print(f"  - {f} ({size} 字节)")
    else:
        print("\n警告: 文件夹为空！")
        print("可能原因:")
        print("  1. 网络连接问题")
        print("  2. 东方财富接口限制")
        print("  3. 非交易时间无数据")
    
    print(f"\n请把 '{output_dir}' 文件夹压缩后发给我")
    input("\n按回车键退出...")


if __name__ == '__main__':
    main()
