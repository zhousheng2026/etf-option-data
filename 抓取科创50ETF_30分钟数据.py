"""
东方财富 科创50ETF 30分钟数据抓取脚本
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

def fetch_588000_30min_data(days=365):
    """
    获取科创50ETF(588000) 30分钟K线数据
    
    参数:
        days: 获取多少天的历史数据
    """
    print(f"正在获取科创50ETF(588000) {days}天 30分钟K线数据...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    
    params = {
        'secid': '1.588000',
        'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '30',   # 30=30分钟线
        'fqt': '1',    # 前复权
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
        print(f"  时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        print(f"  响应状态: {response.status_code}")
        
        data = response.json()
        
        if data.get('data') and data['data'].get('klines'):
            klines = data['data']['klines']
            print(f"  获取到 {len(klines)} 条30分钟K线数据")
            
            records = []
            for line in klines:
                # 格式: 日期时间,开盘价,收盘价,最高价,最低价,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
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
            print(f"  成功创建DataFrame: {len(df)} 行")
            return df
        else:
            print(f"  警告: 没有获取到数据")
            if data.get('data'):
                print(f"  data内容: {data['data']}")
            
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
    
    return None


def main():
    print("="*60)
    print("东方财富 科创50ETF 30分钟数据抓取工具")
    print("="*60)
    
    # 创建输出目录
    output_dir = f"科创50ETF_30分钟数据_{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}/")
    
    # 获取30分钟数据（抓2年数据，约8000根30分钟K线）
    print("\n" + "="*60)
    print("【1】获取科创50ETF(588000) 30分钟K线数据")
    print("="*60)
    
    df = fetch_588000_30min_data(days=730)  # 2年数据
    
    if df is not None and not df.empty:
        # 保存为Excel
        file_path = os.path.join(output_dir, "科创50ETF_588000_30分钟数据.xlsx")
        df.to_excel(file_path, index=False)
        
        print(f"\n✓ 成功保存!")
        print(f"  文件: {file_path}")
        print(f"  数据条数: {len(df)}")
        print(f"\n数据预览:")
        print(df.head(10).to_string())
        print(f"\n...")
        print(df.tail(5).to_string())
        
        # 生成数据说明
        readme = f"""科创50ETF 30分钟数据抓取结果
抓取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
数据范围: {df['日期时间'].min()} 至 {df['日期时间'].max()}
数据条数: {len(df)}
时间周期: 30分钟K线

数据字段说明:
- 日期时间: 交易日期和时间
- 开盘价/收盘价/最高价/最低价: 价格数据（元）
- 成交量: 成交数量
- 成交额: 成交金额（元）
- 涨跌幅: 涨跌幅百分比

用途:
用于30分钟周期的三重底背离策略回测
"""
        
        readme_file = os.path.join(output_dir, "数据说明.txt")
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme)
        
        print(f"\n" + "="*60)
        print(f"抓取完成！")
        print(f"="*60)
        print(f"\n请把 '{output_dir}' 文件夹压缩后发给我")
        
    else:
        print("\n✗ 获取数据失败")
    
    input("\n按回车键退出...")


if __name__ == '__main__':
    main()
