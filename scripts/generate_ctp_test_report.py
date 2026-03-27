#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTP交易测试报告生成器
生成中文版PDF测试报告
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# 确保reports目录存在
Path("reports").mkdir(exist_ok=True)

# 尝试导入reportlab
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("reportlab未安装，将生成Markdown报告")


def check_chinese_font():
    """检查中文字体"""
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]
    
    for font_path in font_paths:
        try:
            pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
            return 'ChineseFont'
        except:
            continue
    
    return None


def generate_markdown_report():
    """生成Markdown格式测试报告"""
    
    timestamp = datetime.now()
    date_str = timestamp.strftime("%Y-%m-%d")
    time_str = timestamp.strftime("%H-%M")
    
    report_content = f"""# CTP交易测试报告

**测试时间**: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}

---

## 一、测试环境

### 1.1 服务器信息

| 类型 | 主服务器 | 备用服务器 |
|------|----------|-----------|
| **行情服务器** | tcp://124.74.248.10:41213 | tcp://120.136.170.202:41213 |
| **交易服务器** | tcp://124.74.248.10:41205 | tcp://120.136.170.202:41205 |

### 1.2 账户信息

- **BrokerID**: 6000
- **投资者ID**: 00001920
- **AppID**: client_sigmaburst_1.0.00

---

## 二、测试结果

### 2.1 连接测试

"""

    # 读取日志文件
    log_files = list(Path("logs").glob("*.log"))
    if log_files:
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()
        
        # 分析日志
        if "连接成功" in log_content or "Connected" in log_content:
            report_content += "- **行情服务器**: ✓ 连接成功\n"
        else:
            report_content += "- **行情服务器**: ✗ 连接失败或超时\n"
        
        if "交易服务器连接成功" in log_content or "TD.*Connected" in log_content:
            report_content += "- **交易服务器**: ✓ 连接成功\n"
        else:
            report_content += "- **交易服务器**: ✗ 连接失败或超时\n"
        
        if "登录成功" in log_content:
            report_content += "- **登录状态**: ✓ 登录成功\n"
        else:
            report_content += "- **登录状态**: ✗ 登录失败\n"
        
        if "认证成功" in log_content:
            report_content += "- **客户端认证**: ✓ 认证成功\n"
        else:
            report_content += "- **客户端认证**: ✗ 认证失败\n"
    else:
        report_content += "- **日志文件**: 未找到日志文件\n"
    
    report_content += """
### 2.2 功能测试

"""
    
    # 检查是否有行情数据
    if "行情" in log_content or "LastPrice" in log_content:
        report_content += "- **行情订阅**: ✓ 成功接收行情数据\n"
    else:
        report_content += "- **行情订阅**: ✗ 未接收到行情数据\n"
    
    # 检查是否有资金查询
    if "账户资金" in log_content or "Available" in log_content:
        report_content += "- **资金查询**: ✓ 成功查询账户资金\n"
    else:
        report_content += "- **资金查询**: ✗ 资金查询失败\n"
    
    # 检查是否有持仓查询
    if "持仓" in log_content or "Position" in log_content:
        report_content += "- **持仓查询**: ✓ 成功查询持仓信息\n"
    else:
        report_content += "- **持仓查询**: ✗ 持仓查询失败\n"
    
    report_content += """
---

## 三、测试总结

### 3.1 测试状态

"""
    
    # 判断整体测试结果
    if "登录成功" in log_content:
        report_content += "**结论**: ✓ CTP连接测试成功，可以进行交易操作\n"
    else:
        report_content += "**结论**: ✗ CTP连接测试失败，请检查服务器状态或等待交易时间\n"
    
    report_content += """
### 3.2 可能的问题

如果连接失败，可能原因：

1. **非交易时间**: CTP测试服务器通常只在交易时间开放
   - 日盘: 09:00-15:00
   - 夜盘: 21:00-次日01:00

2. **网络问题**: 检查网络连接和防火墙设置

3. **服务器维护**: 测试服务器可能定期维护

4. **账号问题**: 测试账号可能需要提前激活

---

## 四、下一步操作

### 4.1 如果测试成功

1. 运行完整交易测试：`python test_ctp_trading_simple.py`
2. 谨慎测试下单功能（最小手数）
3. 查看成交回报和持仓更新

### 4.2 如果测试失败

1. 等待交易时间再测试
2. 检查网络连接
3. 联系期货公司确认测试账号状态

---

**报告生成时间**: {timestamp}

**GitHub Actions运行号**: 见Artifacts

**测试文件**: 
- `test_ctp_trading_simple.py` - 完整交易测试脚本
- `check_ctp_status.py` - 服务器状态检查工具
""".format(timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"))
    
    # 保存Markdown报告
    md_filename = f"reports/CTP_TEST_REPORT_{date_str}_{time_str}.md"
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Markdown报告已生成: {md_filename}")
    
    # 同时生成一个固定名称的报告用于GitHub Actions
    with open("CTP_TRADING_TEST_REPORT.md", 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return md_filename


def generate_pdf_report(markdown_file):
    """从Markdown生成PDF报告"""
    
    if not HAS_REPORTLAB:
        print("reportlab未安装，跳过PDF生成")
        return None
    
    # 检查中文字体
    font_name = check_chinese_font()
    if not font_name:
        print("未找到中文字体，跳过PDF生成")
        return None
    
    # 读取Markdown内容
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 生成PDF
    timestamp = datetime.now()
    date_str = timestamp.strftime("%Y-%m-%d")
    time_str = timestamp.strftime("%H-%M")
    
    pdf_filename = f"reports/CTP_TEST_REPORT_{date_str}_{time_str}.pdf"
    
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    # 样式
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=30,
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        leading=15,
    )
    
    # 构建内容
    story = []
    
    # 简化处理：直接输出Markdown文本
    lines = content.split('\n')
    for line in lines:
        if line.startswith('# '):
            story.append(Paragraph(line[2:], title_style))
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], heading_style))
        elif line.startswith('### '):
            story.append(Paragraph(line[4:], heading_style))
        elif line.startswith('**') and line.endswith('**'):
            story.append(Paragraph(line, body_style))
        elif line.startswith('- '):
            story.append(Paragraph(line, body_style))
        elif line.strip():
            story.append(Paragraph(line, body_style))
        else:
            story.append(Spacer(1, 0.3*cm))
    
    # 生成PDF
    doc.build(story)
    
    print(f"PDF报告已生成: {pdf_filename}")
    
    return pdf_filename


def main():
    """主函数"""
    print("开始生成CTP测试报告...")
    
    # 生成Markdown报告
    md_file = generate_markdown_report()
    
    # 尝试生成PDF报告
    pdf_file = None
    if HAS_REPORTLAB:
        pdf_file = generate_pdf_report(md_file)
    
    print("\n报告生成完成！")
    print(f"- Markdown: {md_file}")
    if pdf_file:
        print(f"- PDF: {pdf_file}")


if __name__ == '__main__':
    main()
