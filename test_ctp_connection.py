#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTP主机环境连接测试脚本
测试CTP交易服务器和行情服务器的网络连接
"""

import socket
import time
from datetime import datetime

# CTP测试环境配置
CTP_CONFIG = {
    "交易服务器": [
        {"ip": "124.74.248.10", "port": 41205, "desc": "CTP主席测试交易服务器1"},
        {"ip": "120.136.170.202", "port": 41205, "desc": "CTP主席测试交易服务器2"},
    ],
    "行情服务器": [
        {"ip": "124.74.248.10", "port": 41213, "desc": "CTP主席测试行情服务器1"},
        {"ip": "120.136.170.202", "port": 41213, "desc": "CTP主席测试行情服务器2"},
    ],
    "BrokerID": "6000",
    "测试账号": "00001920",
}

def test_network_connectivity(ip, port, timeout=5):
    """
    测试指定IP和端口的网络连接
    
    Args:
        ip: 目标IP地址
        port: 目标端口
        timeout: 连接超时时间（秒）
    
    Returns:
        tuple: (是否成功, 延迟毫秒, 错误信息)
    """
    try:
        start_time = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((ip, port))
        sock.close()
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        return True, latency_ms, "连接成功"
    except socket.timeout:
        return False, 0, f"连接超时（{timeout}秒）"
    except ConnectionRefusedError:
        return False, 0, "连接被拒绝"
    except Exception as e:
        return False, 0, f"连接错误: {str(e)}"

def run_ctp_connection_test():
    """运行完整的CTP连接测试"""
    print("=" * 60)
    print("CTP主机环境连接测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    total_tests = 0
    successful_tests = 0
    
    # 测试交易服务器
    print("\n[1] 测试CTP交易服务器连接:")
    print("-" * 40)
    
    for server in CTP_CONFIG["交易服务器"]:
        total_tests += 1
        ip = server["ip"]
        port = server["port"]
        desc = server["desc"]
        
        print(f"\n测试 {desc}:")
        print(f"  地址: {ip}:{port}")
        
        success, latency, message = test_network_connectivity(ip, port)
        
        if success:
            successful_tests += 1
            print(f"  [OK] {message}")
            print(f"  延迟: {latency:.2f} ms")
        else:
            print(f"  [ERROR] {message}")
    
    # 测试行情服务器
    print("\n[2] 测试CTP行情服务器连接:")
    print("-" * 40)
    
    for server in CTP_CONFIG["行情服务器"]:
        total_tests += 1
        ip = server["ip"]
        port = server["port"]
        desc = server["desc"]
        
        print(f"\n测试 {desc}:")
        print(f"  地址: {ip}:{port}")
        
        success, latency, message = test_network_connectivity(ip, port)
        
        if success:
            successful_tests += 1
            print(f"  [OK] {message}")
            print(f"  延迟: {latency:.2f} ms")
        else:
            print(f"  [ERROR] {message}")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("[SUMMARY] 测试结果汇总:")
    print("=" * 60)
    
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"总测试数: {total_tests}")
    print(f"成功数: {successful_tests}")
    print(f"失败数: {total_tests - successful_tests}")
    print(f"成功率: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("\n[SUCCESS] 所有CTP服务器连接测试通过！")
        print("网络环境正常，可以进行下一步的API连接测试。")
    elif success_rate >= 50:
        print("\n[WARNING] 部分CTP服务器连接失败！")
        print("建议检查网络设置和防火墙配置。")
    else:
        print("\n[ERROR] CTP服务器连接失败！")
        print("请检查:")
        print("  1. 网络连接是否正常")
        print("  2. 防火墙是否允许访问相关端口")
        print("  3. CTP服务器是否正常运行")
    
    # 配置信息提示
    print("\n[CONFIG] CTP测试环境配置:")
    print(f"  BrokerID: {CTP_CONFIG['BrokerID']}")
    print(f"  测试账号: {CTP_CONFIG['测试账号']}")
    
    print("\n[NOTE] 注意事项:")
    print("  1. CTP服务器仅在交易日的工作时段提供服务")
    print("  2. 非交易时段服务器可能关闭或限制连接")
    print("  3. 如需API层面的完整测试，请安装vnpy-ctp组件")
    
    return success_rate

if __name__ == "__main__":
    run_ctp_connection_test()