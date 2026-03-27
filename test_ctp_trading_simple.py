#!/usr/bin/env python3
"""
CTP真实交易测试 - 简化版
使用真实的CTP主席测试环境
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime

# 导入CTP库
try:
    from openctp_ctp.thostmduserapi import (
        CThostFtdcMdApi, CThostFtdcMdSpi,
        CThostFtdcReqUserLoginField,
    )
    from openctp_ctp.thosttraderapi import (
        CThostFtdcTraderApi, CThostFtdcTraderSpi,
        CThostFtdcReqUserLoginField as TdLoginField,
        CThostFtdcReqAuthenticateField,
        CThostFtdcInputOrderField,
        CThostFtdcQryTradingAccountField,
        CThostFtdcQryInvestorPositionField,
    )
    print(f"[OK] openctp-ctp 导入成功")
except ImportError as e:
    print(f"[错误] CTP库导入失败: {e}")
    print("[提示] 请运行: pip install openctp-ctp")
    sys.exit(1)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/ctp_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('CTP')

# CTP配置
CTP_CONFIG = {
    'md_servers': [
        'tcp://124.74.248.10:41213',
        'tcp://120.136.170.202:41213',
    ],
    'td_servers': [
        'tcp://124.74.248.10:41205',
        'tcp://120.136.170.202:41205',
    ],
    'broker_id': '6000',
    'investor_id': '00001920',
    'password': 'aa888888',
    'app_id': 'client_sigmaburst_1.0.00',
    'auth_code': 'Y1CTMMUNQFWB69KV',
    'test_symbol': 'rb2610',  # 测试合约
}

# 全局状态
g_md_connected = False
g_md_login = False
g_td_connected = False
g_td_login = False
g_td_auth = False
g_md_event = threading.Event()
g_td_event = threading.Event()
g_auth_event = threading.Event()
g_login_event = threading.Event()
g_market_data = {}
g_request_id = 0


def next_req_id():
    global g_request_id
    g_request_id += 1
    return g_request_id


# ==================== 行情SPI ====================
class MyMdSpi(CThostFtdcMdSpi):
    def OnFrontConnected(self):
        global g_md_connected
        g_md_connected = True
        g_md_event.set()
        logger.info("[MD] 行情服务器连接成功")

    def OnFrontDisconnected(self, nReason):
        global g_md_connected
        g_md_connected = False
        logger.error(f"[MD] 行情服务器断开: {nReason}")

    def OnRspUserLogin(self, pRspUserLogin, pRspInfo, nRequestID, bIsLast):
        global g_md_login
        if pRspInfo.ErrorID == 0:
            g_md_login = True
            logger.info(f"[MD] 行情登录成功 - 交易日: {pRspUserLogin.TradingDay}")
        else:
            logger.error(f"[MD] 行情登录失败: {pRspInfo.ErrorMsg}")

    def OnRspSubMarketData(self, pSpecificInstrument, pRspInfo, nRequestID, bIsLast):
        if pRspInfo and pRspInfo.ErrorID == 0:
            logger.info(f"[MD] 订阅成功: {pSpecificInstrument.InstrumentID}")
        elif pRspInfo:
            logger.error(f"[MD] 订阅失败: {pRspInfo.ErrorMsg}")

    def OnRtnDepthMarketData(self, pDepthMarketData):
        symbol = pDepthMarketData.InstrumentID
        last_price = pDepthMarketData.LastPrice
        if last_price > 0:
            logger.info(f"[行情] {symbol}: 价格={last_price:.2f} "
                       f"量={pDepthMarketData.Volume} "
                       f"时间={pDepthMarketData.UpdateTime}")
            g_market_data[symbol] = {
                'last_price': last_price,
                'volume': pDepthMarketData.Volume,
                'datetime': pDepthMarketData.UpdateTime,
            }


# ==================== 交易SPI ====================
class MyTdSpi(CThostFtdcTraderSpi):
    def OnFrontConnected(self):
        global g_td_connected
        g_td_connected = True
        g_td_event.set()
        logger.info("[TD] 交易服务器连接成功")

    def OnFrontDisconnected(self, nReason):
        global g_td_connected
        g_td_connected = False
        logger.error(f"[TD] 交易服务器断开: {nReason}")

    def OnRspAuthenticate(self, pRspAuthenticateField, pRspInfo, nRequestID, bIsLast):
        global g_td_auth
        if pRspInfo.ErrorID == 0:
            g_td_auth = True
            g_auth_event.set()
            logger.info("[TD] 客户端认证成功")
        else:
            logger.error(f"[TD] 认证失败: {pRspInfo.ErrorMsg}")

    def OnRspUserLogin(self, pRspUserLogin, pRspInfo, nRequestID, bIsLast):
        global g_td_login
        if pRspInfo.ErrorID == 0:
            g_td_login = True
            g_login_event.set()
            logger.info(f"[TD] 交易登录成功 - 交易日: {pRspUserLogin.TradingDay}")
        else:
            logger.error(f"[TD] 登录失败: {pRspInfo.ErrorMsg}")

    def OnRspOrderInsert(self, pInputOrder, pRspInfo, nRequestID, bIsLast):
        if pRspInfo.ErrorID == 0:
            logger.info(f"[TD] 下单成功: {pInputOrder.InstrumentID}")
        else:
            logger.error(f"[TD] 下单失败: {pRspInfo.ErrorMsg}")

    def OnRtnOrder(self, pOrder):
        logger.info(f"[订单回报] {pOrder.InstrumentID} "
                   f"状态={pOrder.OrderStatus} "
                   f"成交={pOrder.VolumeTraded}/{pOrder.VolumeTotalOriginal}")

    def OnRtnTrade(self, pTrade):
        logger.info(f"[成交回报] {pTrade.InstrumentID} "
                   f"价格={pTrade.Price} "
                   f"数量={pTrade.Volume} "
                   f"方向={pTrade.Direction}")

    def OnRspQryTradingAccount(self, pTradingAccount, pRspInfo, nRequestID, bIsLast):
        if pTradingAccount:
            logger.info(f"[账户资金]")
            logger.info(f"  可用: {pTradingAccount.Available:.2f}")
            logger.info(f"  余额: {pTradingAccount.Balance:.2f}")
            logger.info(f"  冻结保证金: {pTradingAccount.FrozenMargin:.2f}")
            logger.info(f"  持仓保证金: {pTradingAccount.CurrMargin:.2f}")
            logger.info(f"  今日盈亏: {pTradingAccount.CloseProfit:.2f}")
            logger.info(f"  持仓盈亏: {pTradingAccount.PositionProfit:.2f}")

    def OnRspQryInvestorPosition(self, pInvestorPosition, pRspInfo, nRequestID, bIsLast):
        if pInvestorPosition:
            logger.info(f"[持仓] {pInvestorPosition.InstrumentID} "
                       f"多头={pInvestorPosition.LongPosition} "
                       f"空头={pInvestorPosition.ShortPosition}")


# ==================== 主测试流程 ====================
def test_ctp():
    """CTP连接和交易测试"""
    
    logger.info("=" * 60)
    logger.info("CTP真实交易环境测试")
    logger.info("=" * 60)
    
    # 创建数据目录
    os.makedirs("ctp_flow_md", exist_ok=True)
    os.makedirs("ctp_flow_td", exist_ok=True)
    
    # 1. 创建行情API
    logger.info("\n步骤1: 创建行情API...")
    md_api = CThostFtdcMdApi.CreateFtdcMdApi("ctp_flow_md")
    md_spi = MyMdSpi()
    md_api.RegisterSpi(md_spi)
    
    # 2. 创建交易API
    logger.info("步骤2: 创建交易API...")
    td_api = CThostFtdcTraderApi.CreateFtdcTraderApi("ctp_flow_td")
    td_spi = MyTdSpi()
    td_api.RegisterSpi(td_spi)
    
    # 3. 注册服务器
    logger.info(f"\n步骤3: 注册服务器...")
    logger.info(f"  行情: {CTP_CONFIG['md_servers'][0]}")
    logger.info(f"  交易: {CTP_CONFIG['td_servers'][0]}")
    
    md_api.RegisterFront(CTP_CONFIG['md_servers'][0])
    td_api.RegisterFront(CTP_CONFIG['td_servers'][0])
    
    # 4. 初始化
    logger.info("\n步骤4: 初始化API...")
    md_api.Init()
    td_api.Init()
    
    # 5. 等待连接
    logger.info("\n步骤5: 等待连接建立...")
    if not g_md_event.wait(timeout=10):
        logger.error("行情服务器连接超时")
        return False
    if not g_td_event.wait(timeout=10):
        logger.error("交易服务器连接超时")
        return False
    
    logger.info("✓ 服务器连接成功！")
    
    # 6. 客户端认证
    logger.info("\n步骤6: 客户端认证...")
    auth_req = CThostFtdcReqAuthenticateField()
    auth_req.BrokerID = CTP_CONFIG['broker_id']
    auth_req.InvestorID = CTP_CONFIG['investor_id']
    auth_req.AppID = CTP_CONFIG['app_id']
    auth_req.AuthCode = CTP_CONFIG['auth_code']
    
    td_api.ReqAuthenticate(auth_req, next_req_id())
    
    if not g_auth_event.wait(timeout=5):
        logger.warning("认证超时，尝试继续...")
    
    # 7. 行情登录
    logger.info("\n步骤7: 行情登录...")
    md_login_req = CThostFtdcReqUserLoginField()
    md_login_req.BrokerID = CTP_CONFIG['broker_id']
    md_login_req.UserID = CTP_CONFIG['investor_id']
    md_login_req.Password = CTP_CONFIG['password']
    
    md_api.ReqUserLogin(md_login_req, next_req_id())
    time.sleep(2)
    
    # 8. 交易登录
    logger.info("\n步骤8: 交易登录...")
    td_login_req = TdLoginField()
    td_login_req.BrokerID = CTP_CONFIG['broker_id']
    td_login_req.UserID = CTP_CONFIG['investor_id']
    td_login_req.Password = CTP_CONFIG['password']
    
    td_api.ReqUserLogin(td_login_req, next_req_id())
    
    if not g_login_event.wait(timeout=10):
        logger.error("交易登录超时")
        return False
    
    logger.info("✓ 登录成功！")
    
    # 9. 查询资金
    logger.info("\n步骤9: 查询账户资金...")
    account_req = CThostFtdcQryTradingAccountField()
    account_req.BrokerID = CTP_CONFIG['broker_id']
    account_req.InvestorID = CTP_CONFIG['investor_id']
    
    td_api.ReqQryTradingAccount(account_req, next_req_id())
    time.sleep(2)
    
    # 10. 查询持仓
    logger.info("\n步骤10: 查询持仓...")
    position_req = CThostFtdcQryInvestorPositionField()
    position_req.BrokerID = CTP_CONFIG['broker_id']
    position_req.InvestorID = CTP_CONFIG['investor_id']
    
    td_api.ReqQryInvestorPosition(position_req, next_req_id())
    time.sleep(2)
    
    # 11. 订阅行情
    logger.info(f"\n步骤11: 订阅行情 {CTP_CONFIG['test_symbol']}...")
    md_api.SubscribeMarketData([CTP_CONFIG['test_symbol']])
    
    # 12. 接收行情
    logger.info("\n步骤12: 接收行情数据（10秒）...")
    time.sleep(10)
    
    # 13. 测试下单（需要确认）
    logger.info("\n步骤13: 是否测试下单？")
    logger.info("  ⚠️  警告：将发送真实订单！")
    logger.info("  输入 'yes' 确认下单测试，其他键跳过")
    
    try:
        user_input = input(">>> ")
        if user_input.lower() == 'yes':
            logger.warning("⚠️  即将发送测试订单...")
            test_order(td_api, CTP_CONFIG)
        else:
            logger.info("跳过下单测试")
    except:
        logger.info("跳过下单测试（非交互模式）")
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("测试总结:")
    logger.info(f"  行情连接: {'✓' if g_md_connected else '✗'}")
    logger.info(f"  行情登录: {'✓' if g_md_login else '✗'}")
    logger.info(f"  交易连接: {'✓' if g_td_connected else '✗'}")
    logger.info(f"  交易登录: {'✓' if g_td_login else '✗'}")
    logger.info(f"  客户端认证: {'✓' if g_td_auth else '✗'}")
    logger.info(f"  接收行情: {'✓' if g_market_data else '✗'}")
    logger.info("=" * 60)
    
    return True


def test_order(td_api, config):
    """测试下单（最小手数）"""
    
    logger.info("\n" + "!" * 60)
    logger.info("⚠️  即将发送真实订单！")
    logger.info("!" * 60)
    
    # 构造订单
    order = CThostFtdcInputOrderField()
    order.BrokerID = config['broker_id']
    order.InvestorID = config['investor_id']
    order.InstrumentID = config['test_symbol']
    order.OrderRef = f"TEST_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # 订单参数
    order.OrderPriceType = '2'  # 限价单
    order.Direction = '0'  # 买入
    order.CombOffsetFlag = '0'  # 开仓
    order.LimitPrice = 3200.0  # 价格
    order.VolumeTotalOriginal = 1  # 1手
    order.TimeCondition = '3'  # 当日有效
    order.VolumeCondition = '1'  # 任何数量
    order.MinVolume = 1
    order.ContingentCondition = '1'  # 立即
    order.ForceCloseReason = '0'  # 非强平
    order.IsAutoSuspend = 0
    order.UserForceClose = 0
    
    logger.info(f"发送订单: {order.InstrumentID} "
               f"买入 开仓 价格={order.LimitPrice} 数量={order.VolumeTotalOriginal}")
    
    # 发送订单
    result = td_api.ReqOrderInsert(order, next_req_id())
    logger.info(f"下单请求返回: {result}")
    
    # 等待回报
    logger.info("等待订单回报...")
    time.sleep(5)


if __name__ == '__main__':
    try:
        success = test_ctp()
        if success:
            logger.info("\n✓ 测试完成！")
        else:
            logger.error("\n✗ 测试失败！")
    except KeyboardInterrupt:
        logger.info("\n用户中断")
    except Exception as e:
        logger.error(f"\n✗ 测试异常: {e}", exc_info=True)
