"""
CTP连接配置文件
包含服务器地址、账户信息、认证信息等
"""

# ==================== CTP服务器配置 ====================

# 主席测试环境
CTP_CONFIG = {
    # 服务器地址
    'md_address': 'tcp://124.74.248.10:41213',  # 行情服务器主
    'md_address_backup': 'tcp://120.136.170.202:41213',  # 行情服务器备
    'td_address': 'tcp://124.74.248.10:41205',  # 交易服务器主
    'td_address_backup': 'tcp://120.136.170.202:41205',  # 交易服务器备
    
    # 账户信息
    'broker_id': '6000',
    'investor_id': '00001920',
    'password': 'aa888888',
    
    # 认证信息
    'app_id': 'client_sigmaburst_1.0.00',
    'auth_code': 'Y1CTMMUNQFWB69KV',
}

# ==================== 连接参数 ====================

# 重连配置
RECONNECT_CONFIG = {
    'max_retry': 3,  # 最大重试次数
    'retry_interval': 5,  # 重试间隔（秒）
    'heartbeat_interval': 30,  # 心跳间隔（秒）
}

# 超时配置
TIMEOUT_CONFIG = {
    'connect_timeout': 10,  # 连接超时（秒）
    'request_timeout': 30,  # 请求超时（秒）
    'login_timeout': 10,  # 登录超时（秒）
}

# ==================== 日志配置 ====================

LOG_CONFIG = {
    'log_level': 'INFO',
    'log_dir': 'logs',
    'log_file': 'ctp_connection.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}
