"""
东方财富 科创50ETF期权数据抓取
"""
import requests
import pandas as pd
import json
from datetime import datetime

def fetch_kcb50_option_data():
    """获取科创50ETF期权数据"""
    
    # 科创50ETF期权合约列表API
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    
    params = {
        'pn': 1,  # 页码
        'pz': 500,  # 每页数量
        'po': 1,
        'np': 1,
        'fltt': 2,
        'invt': 2,
        'fid': 'f12',  # 按代码排序
        'fs': 'm:10',  # 期权市场
        'fields': 'f12,f13,f14,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65,f66,f67,f68,f69,f70,f71,f72,f73,f74,f75,f76,f77,f78,f79,f80,f81,f82,f83,f84,f85,f86,f87,f88,f89,f90,f91,f92,f93,f94,f95,f96,f97,f98,f99,f100'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://quote.eastmoney.com/'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.encoding = 'utf-8'
        
        data = response.json()
        
        if data.get('data') and data['data'].get('diff'):
            items = data['data']['diff']
            
            # 过滤科创50ETF期权
            kcb_options = []
            for item in items:
                name = item.get('f14', '')
                # 筛选科创50相关期权
                if '科创50' in name or '588000' in str(item.get('f12', '')):
                    kcb_options.append({
                        '代码': item.get('f12', ''),
                        '名称': name,
                        '最新价': item.get('f2', 0) / 10000 if item.get('f2') else 0,
                        '涨跌幅': item.get('f3', 0) / 100 if item.get('f3') else 0,
                        '涨跌额': item.get('f4', 0) / 10000 if item.get('f4') else 0,
                        '成交量': item.get('f5', 0),
                        '成交额': item.get('f6', 0),
                        '持仓量': item.get('f7', 0),
                        '行权价': item.get('f8', 0) / 10000 if item.get('f8') else 0,
                        '剩余天数': item.get('f9', 0),
                        '隐含波动率': item.get('f10', 0) / 100 if item.get('f10') else 0,
                        'Delta': item.get('f11', 0) / 10000 if item.get('f11') else 0,
                        'Gamma': item.get('f12', 0) / 10000 if item.get('f12') else 0,
                        'Vega': item.get('f13', 0) / 10000 if item.get('f13') else 0,
                        'Theta': item.get('f14', 0) / 10000 if item.get('f14') else 0,
                        'Rho': item.get('f15', 0) / 10000 if item.get('f15') else 0,
                    })
            
            df = pd.DataFrame(kcb_options)
            return df
        else:
            print("未获取到数据")
            return None
            
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None


def fetch_588000_spot():
    """获取科创50ETF(588000)现货行情"""
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    
    params = {
        'secid': '1.588000',
        'fields': 'f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f57,f58,f59,f60,f61,f62,f63,f64,f170'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://quote.eastmoney.com/'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        data = response.json()
        
        if data.get('data'):
            d = data['data']
            spot_data = {
                '代码': '588000',
                '名称': '科创50ETF',
                '最新价': d.get('f43', 0) / 100 if d.get('f43') else 0,
                '开盘价': d.get('f46', 0) / 100 if d.get('f46') else 0,
                '最高价': d.get('f44', 0) / 100 if d.get('f44') else 0,
                '最低价': d.get('f45', 0) / 100 if d.get('f45') else 0,
                '昨收': d.get('f60', 0) / 100 if d.get('f60') else 0,
                '成交量': d.get('f47', 0),
                '成交额': d.get('f48', 0),
                '涨跌幅': d.get('f170', 0) / 100 if d.get('f170') else 0,
            }
            return spot_data
    except Exception as e:
        print(f"获取现货数据失败: {e}")
        return None


if __name__ == '__main__':
    print("开始抓取东方财富 科创50ETF期权数据...")
    
    # 获取期权数据
    df_options = fetch_kcb50_option_data()
    
    if df_options is not None and not df_options.empty:
        print(f"获取到 {len(df_options)} 条科创50ETF期权数据")
        print(df_options.head(10))
        
        # 保存为Excel
        output_file = f'科创50ETF期权数据_{datetime.now().strftime("%Y%m%d")}.xlsx'
        df_options.to_excel(output_file, index=False)
        print(f"\n数据已保存: {output_file}")
    else:
        print("未能获取期权数据")
    
    # 获取现货数据
    spot = fetch_588000_spot()
    if spot:
        print(f"\n科创50ETF(588000)现货行情:")
        for k, v in spot.items():
            print(f"  {k}: {v}")
