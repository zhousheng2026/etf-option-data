"""
CTP连接模块
处理与CTP服务器的连接、登录、认证等
"""

import time
import logging
import threading
from typing import Optional, Dict, List
from pathlib import Path

try:
    from openctp_ctp import MdApi, TdApi, UserApiStruct
except ImportError:
    # 如果openctp-ctp未安装，使用模拟接口
    print("警告: openctp-ctp未安装，将使用模拟接口")
    MdApi = object
    TdApi = object
    UserApiStruct = None

import sys
sys.path.append('..')
from config import CTP_CONFIG, RECONNECT_CONFIG, TIMEOUT_CONFIG, LOG_CONFIG


class CTPConnectionError(Exception):
    """CTP连接异常"""
    pass


class CTPAuthError(Exception):
    """CTP认证异常"""
    pass


class CTPLoginError(Exception):
    """CTP登录异常"""
    pass


class MarketDataAPI(MdApi):
    """行情API包装类"""
    
    def __init__(self, connection: 'CTPConnection'):
        super().__init__()
        self.connection = connection
        self.logger = logging.getLogger(__name__)
        self.connected = False
        self.logged_in = False
        self.subscribed_symbols = []
        
    def OnFrontConnected(self):
        """行情服务器连接成功"""
        self.connected = True
        self.logger.info("行情服务器连接成功")
        self.connection._on_md_connected()
        
    def OnFrontDisconnected(self, nReason: int):
        """行情服务器断开连接"""
        self.connected = False
        self.logged_in = False
        self.logger.error(f"行情服务器断开连接: {nReason}")
        self.connection._on_md_disconnected(nReason)
        
    def OnRspUserLogin(self, pRspUserLogin: 'dict', pRspInfo: 'dict', nRequestID: int, bIsLast: bool):
        """行情登录响应"""
        if pRspInfo and pRspInfo['ErrorID'] == 0:
            self.logged_in = True
            self.logger.info("行情服务器登录成功")
            self.connection._on_md_login_success()
        else:
            self.logger.error(f"行情服务器登录失败: {pRspInfo}")
            self.connection._on_md_login_failed(pRspInfo)
            
    def OnRspSubMarketData(self, pSpecificInstrument: 'dict', pRspInfo: 'dict', nRequestID: int, bIsLast: bool):
        """订阅行情响应"""
        if pRspInfo and pRspInfo['ErrorID'] == 0:
            self.logger.info(f"订阅行情成功: {pSpecificInstrument['InstrumentID']}")
        else:
            self.logger.error(f"订阅行情失败: {pRspInfo}")
            
    def OnRtnDepthMarketData(self, pDepthMarketData: 'dict'):
        """收到行情数据"""
        self.connection._on_market_data(pDepthMarketData)


class TradingAPI(TdApi):
    """交易API包装类"""
    
    def __init__(self, connection: 'CTPConnection'):
        super().__init__()
        self.connection = connection
        self.logger = logging.getLogger(__name__)
        self.connected = False
        self.logged_in = False
        self.authenticated = False
        
    def OnFrontConnected(self):
        """交易服务器连接成功"""
        self.connected = True
        self.logger.info("交易服务器连接成功")
        self.connection._on_td_connected()
        
    def OnFrontDisconnected(self, nReason: int):
        """交易服务器断开连接"""
        self.connected = False
        self.logged_in = False
        self.authenticated = False
        self.logger.error(f"交易服务器断开连接: {nReason}")
        self.connection._on_td_disconnected(nReason)
        
    def OnRspAuthenticate(self, pRspAuthenticate: 'dict', pRspInfo: 'dict', nRequestID: int, bIsLast: bool):
        """认证响应"""
        if pRspInfo and pRspInfo['ErrorID'] == 0:
            self.authenticated = True
            self.logger.info("交易服务器认证成功")
            self.connection._on_td_auth_success()
        else:
            self.logger.error(f"交易服务器认证失败: {pRspInfo}")
            self.connection._on_td_auth_failed(pRspInfo)
            
    def OnRspUserLogin(self, pRspUserLogin: 'dict', pRspInfo: 'dict', nRequestID: int, bIsLast: bool):
        """交易登录响应"""
        if pRspInfo and pRspInfo['ErrorID'] == 0:
            self.logged_in = True
            self.logger.info("交易服务器登录成功")
            self.connection._on_td_login_success()
        else:
            self.logger.error(f"交易服务器登录失败: {pRspInfo}")
            self.connection._on_td_login_failed(pRspInfo)


class CTPConnection:
    """CTP连接管理类"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化CTP连接
        
        Args:
            config: CTP配置字典，默认使用CTP_CONFIG
        """
        self.config = config or CTP_CONFIG
        self.logger = self._setup_logger()
        
        # API实例
        self.md_api: Optional[MarketDataAPI] = None
        self.td_api: Optional[TradingAPI] = None
        
        # 连接状态
        self.md_connected = False
        self.td_connected = False
        self.md_logged_in = False
        self.td_logged_in = False
        
        # 事件回调
        self.on_market_data_callback = None
        self.on_connected_callback = None
        self.on_disconnected_callback = None
        
        # 数据缓存
        self.market_data_cache = {}
        self.instrument_info = {}
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger(__name__)
        logger.setLevel(LOG_CONFIG['log_level'])
        
        # 创建日志目录
        log_dir = Path(LOG_CONFIG['log_dir'])
        log_dir.mkdir(exist_ok=True)
        
        # 文件处理器
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_dir / LOG_CONFIG['log_file'],
            maxBytes=LOG_CONFIG['max_file_size'],
            backupCount=LOG_CONFIG['backup_count'],
            encoding='utf-8'
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(console_handler)
        
        return logger
        
    def connect(self) -> bool:
        """
        连接CTP服务器
        
        Returns:
            bool: 连接是否成功
        """
        try:
            # 创建行情API
            self.md_api = MarketDataAPI(self)
            self.md_api.CreateFtdcMdApi('./md_flow/')
            self.md_api.RegisterFront(self.config['md_address'])
            self.md_api.Init()
            
            # 创建交易API
            self.td_api = TradingAPI(self)
            self.td_api.CreateFtdcTraderApi('./td_flow/')
            self.td_api.RegisterFront(self.config['td_address'])
            self.td_api.Init()
            
            # 等待连接
            timeout = TIMEOUT_CONFIG['connect_timeout']
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self.md_connected and self.td_connected:
                    self.logger.info("CTP服务器连接成功")
                    return True
                time.sleep(0.1)
                
            raise CTPConnectionError("连接超时")
            
        except Exception as e:
            self.logger.error(f"CTP连接失败: {e}")
            return False
            
    def login(self) -> bool:
        """
        登录CTP服务器
        
        Returns:
            bool: 登录是否成功
        """
        try:
            if not self.md_connected or not self.td_connected:
                raise CTPConnectionError("未连接到服务器")
                
            # 行情登录
            req = UserApiStruct.ReqUserLoginField()
            req.BrokerID = self.config['broker_id']
            req.UserID = self.config['investor_id']
            req.Password = self.config['password']
            
            self.md_api.ReqUserLogin(req, 0)
            
            # 交易认证
            auth_req = UserApiStruct.CThostFtdcReqAuthenticateField()
            auth_req.BrokerID = self.config['broker_id']
            auth_req.UserID = self.config['investor_id']
            auth_req.AppID = self.config['app_id']
            auth_req.AuthCode = self.config['auth_code']
            
            self.td_api.ReqAuthenticate(auth_req, 0)
            
            # 等待登录
            timeout = TIMEOUT_CONFIG['login_timeout']
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self.md_logged_in and self.td_logged_in:
                    self.logger.info("CTP登录成功")
                    if self.on_connected_callback:
                        self.on_connected_callback()
                    return True
                time.sleep(0.1)
                
            raise CTPLoginError("登录超时")
            
        except Exception as e:
            self.logger.error(f"CTP登录失败: {e}")
            return False
            
    def subscribe_market_data(self, symbols: List[str]) -> bool:
        """
        订阅行情数据
        
        Args:
            symbols: 合约代码列表
            
        Returns:
            bool: 订阅是否成功
        """
        if not self.md_api or not self.md_logged_in:
            self.logger.error("未登录行情服务器")
            return False
            
        try:
            self.md_api.SubscribeMarketData(symbols)
            self.md_api.subscribed_symbols.extend(symbols)
            self.logger.info(f"订阅行情: {symbols}")
            return True
        except Exception as e:
            self.logger.error(f"订阅行情失败: {e}")
            return False
            
    def _on_md_connected(self):
        """行情服务器连接成功回调"""
        self.md_connected = True
        
    def _on_td_connected(self):
        """交易服务器连接成功回调"""
        self.td_connected = True
        
    def _on_md_disconnected(self, reason: int):
        """行情服务器断开连接回调"""
        self.md_connected = False
        self.md_logged_in = False
        if self.on_disconnected_callback:
            self.on_disconnected_callback('md', reason)
            
    def _on_td_disconnected(self, reason: int):
        """交易服务器断开连接回调"""
        self.td_connected = False
        self.td_logged_in = False
        if self.on_disconnected_callback:
            self.on_disconnected_callback('td', reason)
            
    def _on_md_login_success(self):
        """行情登录成功回调"""
        self.md_logged_in = True
        
    def _on_md_login_failed(self, error_info: dict):
        """行情登录失败回调"""
        self.md_logged_in = False
        self.logger.error(f"行情登录失败: {error_info}")
        
    def _on_td_auth_success(self):
        """交易认证成功回调"""
        # 认证成功后登录
        req = UserApiStruct.ReqUserLoginField()
        req.BrokerID = self.config['broker_id']
        req.UserID = self.config['investor_id']
        req.Password = self.config['password']
        self.td_api.ReqUserLogin(req, 0)
        
    def _on_td_auth_failed(self, error_info: dict):
        """交易认证失败回调"""
        self.logger.error(f"交易认证失败: {error_info}")
        
    def _on_td_login_success(self):
        """交易登录成功回调"""
        self.td_logged_in = True
        
    def _on_td_login_failed(self, error_info: dict):
        """交易登录失败回调"""
        self.td_logged_in = False
        self.logger.error(f"交易登录失败: {error_info}")
        
    def _on_market_data(self, data: dict):
        """收到行情数据回调"""
        # 缓存数据
        symbol = data.get('InstrumentID')
        if symbol:
            self.market_data_cache[symbol] = data
            
        # 调用用户回调
        if self.on_market_data_callback:
            self.on_market_data_callback(data)
            
    def disconnect(self):
        """断开连接"""
        if self.md_api:
            self.md_api.Release()
            self.md_api = None
            
        if self.td_api:
            self.td_api.Release()
            self.td_api = None
            
        self.md_connected = False
        self.td_connected = False
        self.md_logged_in = False
        self.td_logged_in = False
        
        self.logger.info("CTP连接已断开")


def test_connection():
    """测试CTP连接"""
    print("=" * 60)
    print("CTP连接测试")
    print("=" * 60)
    
    # 创建连接实例
    conn = CTPConnection()
    
    # 设置回调
    def on_market_data(data):
        print(f"收到行情: {data.get('InstrumentID')} - 最新价: {data.get('LastPrice')}")
        
    conn.on_market_data_callback = on_market_data
    
    # 连接
    print("\n正在连接CTP服务器...")
    if not conn.connect():
        print("❌ 连接失败")
        return False
        
    # 登录
    print("正在登录...")
    if not conn.login():
        print("❌ 登录失败")
        return False
        
    # 订阅行情
    print("\n订阅行情...")
    test_symbols = ['sc2604', 'rb2610']
    if not conn.subscribe_market_data(test_symbols):
        print("❌ 订阅失败")
        return False
        
    # 等待数据
    print("\n等待行情数据...")
    time.sleep(10)
    
    # 断开连接
    conn.disconnect()
    print("\n✅ 测试完成")
    return True


if __name__ == '__main__':
    test_connection()
