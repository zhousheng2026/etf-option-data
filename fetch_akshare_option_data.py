"""
使用AKShare获取科创50ETF期权数据
"""
import akshare as ak
import pandas as pd
from datetime import datetime

print("开始获取科创50ETF期权数据...")

# 获取ETF期权合约信息
try:
    # 获取上海证券交易所ETF期权合约
    option_sse = ak.option_sse_spot_price_em()
    print(f"获取到上海证券交易所ETF期权 {len(option_sse)} 条数据")
    
    # 筛选科创50ETF期权 (588000)
    kcb_options = option_sse[option_sse['代码'].str.contains('100', na=False)]
    print(f"筛选后科创50相关期权 {len(kcb_options)} 条")
    
    if not kcb_options.empty:
        print("\n科创50ETF期权数据预览:")
        print(kcb_options.head(20).to_string())
        
        # 保存为Excel
        output_file = f'科创50ETF期权数据_{datetime.now().strftime("%Y%m%d")}.xlsx'
        kcb_options.to_excel(output_file, index=False)
        print(f"\n数据已保存: {output_file}")
    else:
        print("未找到科创50ETF期权数据，显示全部数据:")
        print(option_sse.head(20).to_string())
        
except Exception as e:
    print(f"获取数据失败: {e}")
    import traceback
    traceback.print_exc()

# 获取科创50ETF现货行情
try:
    print("\n获取科创50ETF(588000)现货行情...")
    etf_spot = ak.fund_etf_hist_em(symbol="588000", period="daily", start_date="20250101", end_date="20250225", adjust="")
    print(f"获取到 {len(etf_spot)} 天历史数据")
    print(etf_spot.tail(10).to_string())
    
    # 保存现货数据
    spot_file = f'科创50ETF现货数据_{datetime.now().strftime("%Y%m%d")}.xlsx'
    etf_spot.to_excel(spot_file, index=False)
    print(f"\n现货数据已保存: {spot_file}")
    
except Exception as e:
    print(f"获取现货数据失败: {e}")
    import traceback
    traceback.print_exc()
