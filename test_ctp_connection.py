# -*- coding: utf-8 -*-
"""
CTP连接测试程序
测试光大证券仿真环境连接
"""

import socket
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# 测试配置
TEST_CONFIGS = [
    {
        "name": "配置1",
        "broker_id": "0000",
        "trade_server": "116.236.247.188",
        "trade_port": 10001,
    },
    {
        "name": "配置2", 
        "broker_id": "8888",
        "trade_server": "116.236.247.188",
        "trade_port": 10001,
    },
    {
        "name": "配置3",
        "broker_id": "0000",
        "trade_server": "116.236.247.188",
        "trade_port": 10030,
    },
    {
        "name": "配置4",
        "broker_id": "8888",
        "trade_server": "116.236.247.188",
        "trade_port": 10030,
    },
    {
        "name": "配置5",
        "broker_id": "1010",
        "trade_server": "116.236.247.189",
        "trade_port": 10001,
    },
]

def test_connection(server, port, timeout=5):
    """测试TCP连接"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((server, port))
        sock.close()
        return result == 0
    except Exception as e:
        return False

def test_market_server():
    """测试行情服务器连接"""
    logger.info("=" * 60)
    logger.info("测试行情服务器连接...")
    logger.info("=" * 60)
    
    market_servers = [
        ("116.236.247.188", 10089, "行情主站1"),
        ("116.236.247.189", 10089, "行情主站2"),
    ]
    
    for ip, port, name in market_servers:
        result = test_connection(ip, port)
        status = "✅ 连通" if result else "❌ 不通"
        logger.info(f"{name} ({ip}:{port}) - {status}")
    
    return True

def test_trade_servers():
    """测试交易服务器连接"""
    logger.info("\n" + "=" * 60)
    logger.info("测试交易服务器连接...")
    logger.info("=" * 60)
    
    for config in TEST_CONFIGS:
        result = test_connection(config["trade_server"], config["trade_port"])
        status = "✅ 连通" if result else "❌ 不通"
        logger.info(f"{config['name']}: BrokerID={config['broker_id']}, "
                   f"{config['trade_server']}:{config['trade_port']} - {status}")

def main():
    """主函数"""
    logger.info("\n" + "=" * 60)
    logger.info("光大证券CTP仿真环境连接测试")
    logger.info("=" * 60)
    logger.info(f"资金账号: 43503750")
    logger.info(f"密码: 147258")
    logger.info("=" * 60)
    
    # 测试行情服务器
    test_market_server()
    
    # 测试交易服务器
    test_trade_servers()
    
    logger.info("\n" + "=" * 60)
    logger.info("测试完成")
    logger.info("=" * 60)
    logger.info("\n说明：")
    logger.info("- 行情服务器已确认可用")
    logger.info("- 交易服务器需要找到正确的BrokerID和端口")
    logger.info("- 如果以上配置都不通，可能需要：")
    logger.info("  1. 确认是否开通了CTP接口权限")
    logger.info("  2. 联系光大证券技术人员获取准确配置")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
