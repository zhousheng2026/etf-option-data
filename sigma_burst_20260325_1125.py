#!/usr/bin/env python3
"""
Sigma Burst v0.5.0 - CTP主席测试环境真实连接版
================================================================================
创建时间: 2026-03-25
用途: 接入CTP主席测试环境，获取实时行情数据

使用方法:
    python sigma_burst_20260325_1125.py --mode connect    # 仅测试连接
    python sigma_burst_20260325_1125.py --mode monitor    # 持续监测行情
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional

# 导入CTP库
CTP_AVAILABLE = False
CTP_LIB_NAME = None

try:
    from openctp_ctp.thostmduserapi import (
        CThostFtdcMdApi, CThostFtdcMdSpi,
        CThostFtdcReqUserLoginField,
    )
    from openctp_ctp.thosttraderapi import (
        CThostFtdcTraderApi, CThostFtdcTraderSpi,
        CThostFtdcReqUserLoginField as TdLoginField,
        CThostFtdcReqAuthenticateField,
    )
    CTP_AVAILABLE = True
    CTP_LIB_NAME = 'openctp_ctp'
    import openctp_ctp
    _ver = getattr(openctp_ctp, '__version__', 'unknown')
    print(f"[系统] {CTP_LIB_NAME} 导入成功 (v{_ver})")
except ImportError:
    pass

if not CTP_AVAILABLE:
    print("[错误] CTP库导入失败")
    print("[提示] 请运行: pip install openctp-ctp")
    sys.exit(1)

# ========== 配置 ==========
CONFIG = {
    'version': '0.5.0',
    'create_time': '2026-03-25 11:25',

    # CTP主席测试环境
    'md_servers': [
        'tcp://124.74.248.10:41213',
        'tcp://120.136.170.202:41213',
    ],
    'td_servers': [
        'tcp://124.74.248.10:41205',
        'tcp://120.136.170.202:41205',
    ],
    'broker_id': '6000',
    'auth_code': 'Y1CTMMUNQFWB69KV',
    'app_id': 'client_sigmaburst_1.0.00',
    'investor_id': '00001920',
    'password': 'aa888888',

    # 监测品种 (2026年主力合约)
    'symbols': [
        'sc2604', 'bu2606', 'pg2604', 'ec2604', 'fu2605', 'px2605',
        'pp2605', 'ta2605', 'sm2605', 'sa2605', 'ma2605', 'rb2610',
        'hc2610', 'i2609', 'j2609', 'jm2609',
    ],
}

# ========== 日志 ==========
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOG_DIR, f'ctp_{datetime.now().strftime("%Y%m%d")}.log'),
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger('CTP')


# ========== 全局状态 ==========
g_market_data = {}
g_md_connected = False
g_md_login = False
g_md_event = threading.Event()

g_td_connected = False
g_td_login = False
g_td_authenticated = False
g_td_event = threading.Event()
g_login_event = threading.Event()

g_request_id = 0
g_data_lock = threading.Lock()
g_price_count = 0
g_md_api = None
g_td_api = None
g_front_id = 0
g_session_id = 0
g_trading_day = ''


def next_req_id():
    global g_request_id
    g_request_id += 1
    return g_request_id


# ========== 行情回调 ==========
class MyMdSpi(CThostFtdcMdSpi):

    def OnFrontConnected(self):
        global g_md_connected
        g_md_connected = True
        g_md_event.set()
        logger.info("[MD] 行情前置机连接成功")

    def OnFrontDisconnected(self, nReason):
        global g_md_connected
        g_md_connected = False
        logger.error(f"[MD] 行情前置机断开, 原因码: {nReason}")

    def OnRspUserLogin(self, pRspUserLogin, pRspInfo, nRequestID, bIsLast):
        global g_md_login, g_trading_day
        if pRspInfo.ErrorID == 0:
            g_md_login = True
            g_trading_day = str(pRspUserLogin.TradingDay)
            logger.info(f"[MD] 行情登录成功, 交易日: {g_trading_day}")
        else:
            logger.error(f"[MD] 行情登录失败: {pRspInfo.ErrorID} - {pRspInfo.ErrorMsg}")

    def OnRspSubMarketData(self, pSpecificInstrument, pRspInfo, nRequestID, bIsLast):
        if pRspInfo and pRspInfo.ErrorID != 0:
            logger.error(f"[MD] 订阅失败: {pRspInfo.ErrorID} - {pRspInfo.ErrorMsg}")

    def OnRtnDepthMarketData(self, pDepthMarketData):
        global g_price_count
        if pDepthMarketData is None:
            return
        try:
            symbol = str(pDepthMarketData.InstrumentID)
            last_price = float(pDepthMarketData.LastPrice) if pDepthMarketData.LastPrice else 0
            data = {
                'symbol': symbol,
                'last_price': last_price if last_price > 0 else None,
                'pre_close': float(pDepthMarketData.PreClosePrice) if pDepthMarketData.PreClosePrice else 0,
                'pre_settle': float(pDepthMarketData.PreSettlementPrice) if pDepthMarketData.PreSettlementPrice else 0,
                'bid1': float(pDepthMarketData.BidPrice1) if pDepthMarketData.BidPrice1 > 0 else None,
                'ask1': float(pDepthMarketData.AskPrice1) if pDepthMarketData.AskPrice1 > 0 else None,
                'volume': int(pDepthMarketData.Volume) if pDepthMarketData.Volume else 0,
                'update_time': str(pDepthMarketData.UpdateTime),
            }
            with g_data_lock:
                g_market_data[symbol] = data
                g_price_count += 1
        except Exception:
            pass


# ========== 交易回调 ==========
class MyTdSpi(CThostFtdcTraderSpi):

    def OnFrontConnected(self):
        global g_td_connected
        g_td_connected = True
        g_td_event.set()
        logger.info("[TD] 交易前置机连接成功")

    def OnFrontDisconnected(self, nReason):
        global g_td_connected
        g_td_connected = False
        logger.error(f"[TD] 交易前置机断开, 原因码: {nReason}")

    def OnRspAuthenticate(self, pRspAuthenticateField, pRspInfo, nRequestID, bIsLast):
        global g_td_authenticated
        if pRspInfo.ErrorID == 0:
            g_td_authenticated = True
            logger.info("[TD] 客户端认证成功")
        else:
            logger.error(f"[TD] 客户端认证失败: {pRspInfo.ErrorID} - {pRspInfo.ErrorMsg}")

    def OnRspUserLogin(self, pRspUserLogin, pRspInfo, nRequestID, bIsLast):
        global g_td_login, g_front_id, g_session_id, g_trading_day
        if pRspInfo.ErrorID == 0:
            g_td_login = True
            g_front_id = pRspUserLogin.FrontID
            g_session_id = pRspUserLogin.SessionID
            g_trading_day = str(pRspUserLogin.TradingDay)
            g_login_event.set()
            logger.info(f"[TD] 交易登录成功, 交易日: {g_trading_day}")
            logger.info(f"[TD] 前置机ID: {g_front_id}, 会话号: {g_session_id}")
        else:
            logger.error(f"[TD] 交易登录失败: {pRspInfo.ErrorID} - {pRspInfo.ErrorMsg}")

    def OnRspError(self, pRspInfo, nRequestID, bIsLast):
        if pRspInfo and pRspInfo.ErrorID != 0:
            logger.error(f"[TD] 错误: {pRspInfo.ErrorID} - {pRspInfo.ErrorMsg}")

    def OnRtnOrder(self, pOrder):
        if pOrder:
            logger.info(f"[TD-报单] {pOrder.InstrumentID} 状态:{pOrder.OrderStatus}")

    def OnRtnTrade(self, pTrade):
        if pTrade:
            logger.info(f"[TD-成交] {pTrade.InstrumentID} {pTrade.Volume}手@{pTrade.Price}")


# ========== 连接函数 ==========

def connect_md():
    """连接行情服务器"""
    global g_md_api, g_md_connected, g_md_login

    logger.info("=" * 60)
    logger.info("[MD] 正在连接行情服务器...")
    logger.info(f"[MD] BrokerID: {CONFIG['broker_id']}")
    logger.info(f"[MD] 投资者ID: {CONFIG['investor_id']}")
    for addr in CONFIG['md_servers']:
        logger.info(f"[MD] 行情地址: {addr}")
    logger.info("=" * 60)

    try:
        g_md_event.clear()
        md_flow = os.path.join(LOG_DIR, 'md_flow')

        md_spi = MyMdSpi()
        g_md_api = CThostFtdcMdApi.CreateFtdcMdApi(md_flow)
        g_md_api.RegisterSpi(md_spi)

        for addr in CONFIG['md_servers']:
            g_md_api.RegisterFront(addr)

        g_md_api.Init()

        if not g_md_event.wait(timeout=20):
            logger.error("[MD] 连接行情前置机超时(20秒)")
            return False

        login_field = CThostFtdcReqUserLoginField()
        login_field.BrokerID = CONFIG['broker_id']
        login_field.UserID = CONFIG['investor_id']
        login_field.Password = CONFIG['password']
        g_md_api.ReqUserLogin(login_field, next_req_id())

        for _ in range(40):
            time.sleep(0.5)
            if g_md_login:
                logger.info("[MD] 行情连接和登录完成!")
                return True

        logger.error("[MD] 行情登录超时(20秒)")
        return False

    except Exception as e:
        logger.error(f"[MD] 连接异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def connect_td():
    """连接交易服务器"""
    global g_td_api, g_td_connected, g_td_login, g_td_authenticated

    logger.info("=" * 60)
    logger.info("[TD] 正在连接交易服务器...")
    for addr in CONFIG['td_servers']:
        logger.info(f"[TD] 交易地址: {addr}")
    logger.info("=" * 60)

    try:
        g_td_event.clear()
        td_flow = os.path.join(LOG_DIR, 'td_flow')

        td_spi = MyTdSpi()
        g_td_api = CThostFtdcTraderApi.CreateFtdcTraderApi(td_flow)
        g_td_api.RegisterSpi(td_spi)

        for addr in CONFIG['td_servers']:
            g_td_api.RegisterFront(addr)

        g_td_api.SubscribePrivateTopic(1)
        g_td_api.SubscribePublicTopic(1)

        g_td_api.Init()

        if not g_td_event.wait(timeout=20):
            logger.error("[TD] 连接交易前置机超时(20秒)")
            return False

        # 认证
        auth_field = CThostFtdcReqAuthenticateField()
        auth_field.BrokerID = CONFIG['broker_id']
        auth_field.UserID = CONFIG['investor_id']
        auth_field.AuthCode = CONFIG['auth_code']
        auth_field.AppID = CONFIG['app_id']
        g_td_api.ReqAuthenticate(auth_field, next_req_id())

        for _ in range(40):
            time.sleep(0.5)
            if g_td_authenticated:
                break
        else:
            logger.error("[TD] 认证超时(20秒)")
            return False

        # 登录
        g_login_event.clear()
        login_field = TdLoginField()
        login_field.BrokerID = CONFIG['broker_id']
        login_field.UserID = CONFIG['investor_id']
        login_field.Password = CONFIG['password']
        g_td_api.ReqUserLogin(login_field, next_req_id())

        for _ in range(40):
            time.sleep(0.5)
            if g_td_login:
                logger.info("[TD] 交易连接、认证和登录完成!")
                return True

        logger.error("[TD] 交易登录超时(20秒)")
        return False

    except Exception as e:
        logger.error(f"[TD] 连接异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def subscribe_quotes(symbols: List[str]):
    """订阅行情"""
    if g_md_api and g_md_login:
        logger.info(f"[MD] 订阅 {len(symbols)} 个合约: {', '.join(symbols)}")
        g_md_api.SubscribeMarketData(symbols, len(symbols))


def print_quote_table():
    """打印行情表格"""
    logger.info("-" * 65)
    logger.info(f" {'合约':<10} {'最新价':>10} {'昨结算':>10} {'买一':>10} {'卖一':>10} {'成交量':>10}")
    logger.info("-" * 65)

    with g_data_lock:
        for symbol in CONFIG['symbols']:
            data = g_market_data.get(symbol)
            if data and data.get('last_price'):
                pre = data.get('pre_settle') or data.get('pre_close') or 0
                bid = f"{data['bid1']:.1f}" if data.get('bid1') else '-'
                ask = f"{data['ask1']:.1f}" if data.get('ask1') else '-'
                logger.info(
                    f" {symbol:<10} {data['last_price']:>10.1f} {pre:>10.1f} "
                    f"{bid:>10} {ask:>10} {data.get('volume', 0):>10}"
                )
            else:
                logger.info(f" {symbol:<10} {'无数据':>10}")
    logger.info("-" * 65)
    logger.info(f" 累计收到行情数据: {g_price_count} 条")
    logger.info("-" * 65)


def disconnect():
    """断开连接"""
    logger.info("[系统] 正在断开连接...")
    if g_md_api:
        try: g_md_api.Release()
        except: pass
    if g_td_api:
        try: g_td_api.Release()
        except: pass
    logger.info("[系统] 已断开")


# ========== 生成测试报告 ==========
def generate_report():
    """生成测试报告（写入文件和打印）"""
    report_lines = []
    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("  CTP主席测试环境连接测试报告")
    report_lines.append(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"  CTP库: {CTP_LIB_NAME}")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append("  1. 连接配置:")
    report_lines.append(f"     行情服务器: {CONFIG['md_servers']}")
    report_lines.append(f"     交易服务器: {CONFIG['td_servers']}")
    report_lines.append(f"     BrokerID: {CONFIG['broker_id']}")
    report_lines.append(f"     投资者ID: {CONFIG['investor_id']}")
    report_lines.append(f"     AppID: {CONFIG['app_id']}")
    report_lines.append(f"     AuthCode: {CONFIG['auth_code']}")
    report_lines.append("")
    report_lines.append("  2. 测试结果:")
    report_lines.append(f"     行情前置机连接: {'PASS' if g_md_connected else 'FAIL'}")
    report_lines.append(f"     行情登录: {'PASS' if g_md_login else 'FAIL'}")
    report_lines.append(f"     交易前置机连接: {'PASS' if g_td_connected else 'FAIL'}")
    report_lines.append(f"     客户端认证: {'PASS' if g_td_authenticated else 'FAIL'}")
    report_lines.append(f"     交易登录: {'PASS' if g_td_login else 'FAIL'}")
    report_lines.append(f"     交易日: {g_trading_day or '未获取'}")
    report_lines.append(f"     收到行情数据: {g_price_count} 条")
    report_lines.append("")
    report_lines.append("  3. 行情数据快照:")
    with g_data_lock:
        for symbol in CONFIG['symbols']:
            data = g_market_data.get(symbol)
            if data and data.get('last_price'):
                report_lines.append(
                    f"     {symbol}: 最新价={data['last_price']:.1f}, "
                    f"昨结算={data.get('pre_settle', 0):.1f}, "
                    f"买一={data.get('bid1', '-')}, 卖一={data.get('ask1', '-')}, "
                    f"成交量={data.get('volume', 0)}"
                )
            else:
                report_lines.append(f"     {symbol}: 无数据")
    report_lines.append("")

    # 总评
    all_pass = g_md_connected and g_md_login and g_td_connected and g_td_authenticated and g_td_login
    report_lines.append("  4. 总评:")
    if all_pass:
        report_lines.append("     ALL PASS - CTP主席测试环境连接完全正常")
    elif g_md_connected and g_md_login:
        report_lines.append("     PARTIAL PASS - 行情连接正常，交易连接异常")
    elif g_td_connected and g_td_login:
        report_lines.append("     PARTIAL PASS - 交易连接正常，行情连接异常")
    else:
        report_lines.append("     FAIL - 连接全部失败，请检查网络和配置")
    report_lines.append("")
    report_lines.append("=" * 60)

    # 打印
    for line in report_lines:
        logger.info(line)

    # 写入报告文件
    report_path = os.path.join(LOG_DIR, f'ctp_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    logger.info(f"  测试报告已保存: {report_path}")

    return report_path, all_pass


# ========== 主程序 ==========
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sigma Burst v0.5.0 - CTP主席测试环境')
    parser.add_argument('--mode', choices=['connect', 'monitor'], default='connect',
                       help='connect=测试连接+生成报告, monitor=持续监测行情')
    args = parser.parse_args()

    logger.info("")
    logger.info("*" * 60)
    logger.info("  Sigma Burst v0.5.0 - CTP主席测试环境")
    logger.info(f"  CTP库: {CTP_LIB_NAME}")
    logger.info(f"  BrokerID: {CONFIG['broker_id']}")
    logger.info(f"  投资者ID: {CONFIG['investor_id']}")
    for addr in CONFIG['md_servers']:
        logger.info(f"  行情服务器: {addr}")
    for addr in CONFIG['td_servers']:
        logger.info(f"  交易服务器: {addr}")
    logger.info("*" * 60)
    logger.info("")

    # 1. 连接行情
    md_ok = connect_md()

    # 2. 连接交易
    td_ok = connect_td()

    # 3. 订阅行情
    if md_ok:
        subscribe_quotes(CONFIG['symbols'])
        # 等待接收行情数据
        logger.info("[系统] 等待10秒接收行情数据...")
        time.sleep(10)

    # 4. 生成测试报告
    report_path, all_pass = generate_report()

    if args.mode == 'monitor' and (md_ok or td_ok):
        logger.info("[系统] 进入持续监测模式 (每10秒刷新), Ctrl+C 停止")
        try:
            while True:
                time.sleep(10)
                print_quote_table()
        except KeyboardInterrupt:
            logger.info("[系统] 用户停止监测")

    disconnect()

    # 退出码: 0=全通过, 1=部分通过, 2=全失败
    if all_pass:
        sys.exit(0)
    elif md_ok or td_ok:
        sys.exit(1)
    else:
        sys.exit(2)
