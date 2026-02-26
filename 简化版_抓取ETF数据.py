"""
东方财富 科创50ETF数据抓取 - 简化版
"""
import requests
import pandas as pd
from datetime import datetime
import os

print("="*60)
print("科创50ETF数据抓取工具")
print("="*60)

# 创建输出目录
output_dir = f"ETF数据_{datetime.now().strftime('%Y%m%d')}"
os.makedirs(output_dir, exist_ok=True)
print(f"\n输出目录: {output_dir}/")

# 获取科创50ETF数据
print("\n正在获取科创50ETF(588000)数据...")

url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
params = {
    'secid': '1.588000',
    'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
    'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
    'klt': '101',
    'fqt': '1',
    'beg': '20230101',
    'end': '20251231',
    'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'http://quote.eastmoney.com/'
}

try:
    response = requests.get(url, params=params, headers=headers, timeout=30)
    print(f"  服务器响应: {response.status_code}")
    
    data = response.json()
    
    if data.get('data') and data['data'].get('klines'):
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
        
        # 保存为Excel
        file_path = os.path.join(output_dir, "科创50ETF_588000.xlsx")
        df.to_excel(file_path, index=False)
        
        print(f"\n✓ 成功保存!")
        print(f"  文件: {file_path}")
        print(f"  数据条数: {len(df)}")
        print(f"\n数据预览:")
        print(df.head().to_string())
        
    else:
        print(f"\n✗ 数据为空")
        print(f"  响应内容: {data}")
        
except Exception as e:
    print(f"\n✗ 出错: {e}")
    import traceback
    traceback.print_exc()

input("\n按回车键退出...")
