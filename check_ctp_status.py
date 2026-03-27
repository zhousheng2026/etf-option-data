#!/usr/bin/env python3
"""
CTP连接状态检查工具
检查CTP服务器是否在线，并显示连接状态
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime

try:
    from openctp_ctp.thostmduserapi import CThostFtdcMdApi, CThostFtdcMdSpi
    from openctp_ctp.thosttraderapi import CThostFtdcTraderApi, CThostFtdcTraderSpi
    print("[OK] openctp-ctp 已导入")
except ImportError as e:
    print(f"[错误] openctp-ctp 未安装: {e}")
    sys.exit(1)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('CTP')

# CTP配置
SERVERS = {
    'md_servers': [
        ('主服务器', 'tcp://124.74.248.10:41213'),
        ('备用服务器', 'tcp://120.136.170.202:41213'),
    ],
    'td_servers': [
        ('主服务器', 'tcp://124.74.248.10:41205'),
        ('备用服务器', 'tcp://120.136.170.202:41205'),
    ],
}

# 全局状态
g_connected = False
g_event = threading.Event()


class CheckMdSpi(CThostFtdcMdSpi):
    def OnFrontConnected(self):
        global g_connected
        g_connected = True
        g_event.set()
        logger.info("  ✓ 连接成功！")
    
    def OnFrontDisconnected(self, nReason):
        global g_connected
        g_connected = False
        logger.info(f"  ✗ 断开连接: {nReason}")


class CheckTdSpi(CThostFtdcTraderSpi):
    def OnFrontConnected(self):
        global g_connected
        g_connected = True
        g_event.set()
        logger.info("  ✓ 连接成功！")
    
    def OnFrontDisconnected(self, nReason):
        global g_connected
        g_connected = False
        logger.info(f"  ✗ 断开连接: {nReason}")


def check_server(server_type, servers):
    """检查服务器连接状态"""
    
    global g_connected, g_event
    
    print(f"\n{'='*60}")
    print(f"检查{server_type}服务器")
    print(f"{'='*60}")
    
    for name, address in servers:
        print(f"\n[{name}] {address}")
        print("  尝试连接...")
        
        g_connected = False
        g_event.clear()
        
        # 创建API
        if '行情' in server_type:
            api = CThostFtdcMdApi.CreateFtdcMdApi(f"check_md_{int(time.time())}")
            spi = CheckMdSpi()
        else:
            api = CThostFtdcTraderApi.CreateFtdcTraderApi(f"check_td_{int(time.time())}")
            spi = CheckTdSpi()
        
        api.RegisterSpi(spi)
        api.RegisterFront(address)
        api.Init()
        
        # 等待连接
        if g_event.wait(timeout=15):
            logger.info(f"  [OK] {name} 在线")
            return True, address
        else:
            logger.info(f"  [X] {name} 超时（可能离线或非交易时间）")
    
    return False, None


def check_trading_time():
    """检查当前是否在交易时间"""
    
    now = datetime.now()
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute
    
    print(f"\n{'='*60}")
    print(f"当前时间检查")
    print(f"{'='*60}")
    print(f"时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"星期: {['一','二','三','四','五','六','日'][weekday]}")
    
    # 判断是否交易时间
    is_trading = False
    trading_period = ""
    
    if weekday < 5:  # 周一到周五
        if (9 <= hour < 15) or (hour == 15 and minute == 0):
            is_trading = True
            trading_period = "日盘交易时间"
        elif 21 <= hour or hour < 1 or (hour == 1 and minute == 0):
            is_trading = True
            trading_period = "夜盘交易时间"
    
    if is_trading:
        print(f"状态: [OK] {trading_period}")
    else:
        print(f"状态: [X] 非交易时间")
        print("\n提示: CTP测试服务器通常只在交易时间开放")
        print("  日盘: 09:00-15:00")
        print("  夜盘: 21:00-次日01:00")
    
    return is_trading


def main():
    print("\n" + "="*60)
    print("CTP主席测试环境连接状态检查")
    print("="*60)
    
    # 检查交易时间
    is_trading = check_trading_time()
    
    # 检查行情服务器
    md_online, md_address = check_server('行情', SERVERS['md_servers'])
    
    # 检查交易服务器
    td_online, td_address = check_server('交易', SERVERS['td_servers'])
    
    # 总结
    print(f"\n{'='*60}")
    print("检查结果总结")
    print(f"{'='*60}")
    print(f"行情服务器: {'在线' if md_online else '离线'}")
    print(f"交易服务器: {'在线' if td_online else '离线'}")
    
    if md_online and td_online:
        print(f"\n✓ CTP测试环境可用，可以开始交易测试！")
        print(f"\n建议操作:")
        print(f"  1. 运行: python test_ctp_trading_simple.py")
        print(f"  2. 确认账号登录成功")
        print(f"  3. 谨慎测试下单功能")
    elif not is_trading:
        print(f"\n⚠️  当前非交易时间，CTP服务器可能关闭")
        print(f"\n建议操作:")
        print(f"  1. 等待交易时间再测试")
        print(f"  2. 日盘: 09:00-15:00")
        print(f"  3. 夜盘: 21:00-次日01:00")
    else:
        print(f"\n✗ CTP测试环境不可用")
        print(f"\n可能原因:")
        print(f"  1. 网络连接问题")
        print(f"  2. 服务器维护中")
        print(f"  3. 防火墙阻止连接")
    
    print("="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}", exc_info=True)
