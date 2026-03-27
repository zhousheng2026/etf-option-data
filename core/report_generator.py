#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF报告生成器
PDF Report Generator for Volatility Strategy
"""

from datetime import datetime
from typing import Dict, List, Optional
import logging
import os

logger = logging.getLogger(__name__)


class ReportGenerator:
    """PDF报告生成器"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查reportlab是否可用
        self.has_reportlab = self._check_reportlab()
        
        logger.info(f"报告生成器初始化完成 - 输出目录:{output_dir}, PDF支持:{self.has_reportlab}")
    
    def _check_reportlab(self) -> bool:
        """检查reportlab是否可用"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            return True
        except ImportError:
            logger.warning("reportlab未安装，将生成Markdown报告")
            return False
    
    def generate_daily_report(self, signals: List, volatility_data: Dict, 
                             market_summary: Dict) -> str:
        """
        生成每日报告
        
        Args:
            signals: 交易信号列表
            volatility_data: 波动率数据
            market_summary: 市场概况
        
        Returns:
            报告文件路径
        """
        timestamp = datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H-%M")
        
        # 优先使用Markdown格式，确保中文正常显示
        filename = f"volatility_strategy_report_{date_str}_{time_str}.md"
        filepath = os.path.join(self.output_dir, filename)
        self._generate_markdown_report(filepath, signals, volatility_data, market_summary)
        
        # 如果环境支持中文字体，同时生成PDF
        if self.has_reportlab and self._check_chinese_font():
            try:
                pdf_filename = f"volatility_strategy_report_{date_str}_{time_str}.pdf"
                pdf_filepath = os.path.join(self.output_dir, pdf_filename)
                self._generate_pdf_report(pdf_filepath, signals, volatility_data, market_summary)
                logger.info(f"PDF报告生成完成: {pdf_filepath}")
            except Exception as e:
                logger.warning(f"PDF生成失败，仅保留Markdown: {e}")
        
        logger.info(f"报告生成完成: {filepath}")
        return filepath
    
    def _check_chinese_font(self) -> bool:
        """检查是否支持中文字体"""
        try:
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            
            # 尝试注册中文字体
            font_paths = [
                'simhei.ttf',
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            ]
            
            for font_path in font_paths:
                try:
                    pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                    return True
                except:
                    continue
            return False
        except:
            return False
    
    def _generate_pdf_report(self, filepath: str, signals: List, 
                            volatility_data: Dict, market_summary: Dict):
        """生成PDF格式报告"""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        # 创建PDF文档
        doc = SimpleDocTemplate(filepath, pagesize=A4, 
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
        
        # 注册中文字体
        font_name = 'Helvetica'
        chinese_font_registered = False
        
        # 尝试多种中文字体方案
        font_paths = [
            'simhei.ttf',  # Windows
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # Linux
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux fallback
            'NotoSansCJK',  # Google Noto fonts
        ]
        
        for font_path in font_paths:
            try:
                if font_path.endswith('.ttf') or font_path.endswith('.ttc'):
                    pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                else:
                    # 尝试注册系统字体
                    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
                    pdfmetrics.registerFont(UnicodeCIDFont(font_path))
                font_name = 'ChineseFont'
                chinese_font_registered = True
                logger.info(f"成功注册中文字体: {font_path}")
                break
            except Exception as e:
                continue
        
        if not chinese_font_registered:
            logger.warning("中文字体注册失败，将使用英文报告")
            # 使用Markdown格式代替
            self._generate_markdown_report(filepath.replace('.pdf', '.md'), signals, volatility_data, market_summary)
            return
        
        # 创建样式
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName=font_name,
            fontSize=18,
            spaceAfter=30,
            alignment=1  # 居中
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName=font_name,
            fontSize=14,
            spaceAfter=12
        )
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=10,
            spaceAfter=8
        )
        
        # 构建内容
        story = []
        
        # 标题
        title = Paragraph(f"波动率突破策略日报<br/>{datetime.now().strftime('%Y-%m-%d')}", title_style)
        story.append(title)
        story.append(Spacer(1, 0.5*cm))
        
        # 市场概况
        story.append(Paragraph("一、市场概况", heading_style))
        market_text = f"""
        监控品种数：{market_summary.get('total_symbols', 0)}<br/>
        有效信号数：{market_summary.get('total_signals', 0)}<br/>
        做多波动率信号：{market_summary.get('long_vol_count', 0)}<br/>
        做空波动率信号：{market_summary.get('short_vol_count', 0)}<br/>
        平均IV百分位：{market_summary.get('avg_iv_rank', 0):.1f}%<br/>
        平均HV百分位：{market_summary.get('avg_hv_rank', 0):.1f}%
        """
        story.append(Paragraph(market_text, body_style))
        story.append(Spacer(1, 0.5*cm))
        
        # 交易信号
        story.append(Paragraph("二、交易信号", heading_style))
        
        if signals:
            # 创建信号表格
            table_data = [['品种', '信号类型', '方向', '强度', '入场价', '止损价', '止盈价', '理由']]
            
            for signal in signals:
                table_data.append([
                    signal.symbol,
                    signal.signal_type,
                    signal.direction,
                    f"{signal.strength:.2f}",
                    f"{signal.entry_price:.2f}",
                    f"{signal.stop_loss:.2f}",
                    f"{signal.take_profit:.2f}",
                    signal.reason[:20] + "..." if len(signal.reason) > 20 else signal.reason
                ])
            
            table = Table(table_data, colWidths=[2*cm, 2.5*cm, 1.5*cm, 1.5*cm, 2*cm, 2*cm, 2*cm, 4*cm])
            table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), font_name),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(table)
        else:
            story.append(Paragraph("当前无有效交易信号", body_style))
        
        story.append(Spacer(1, 0.5*cm))
        
        # 波动率数据
        story.append(Paragraph("三、波动率数据", heading_style))
        
        if volatility_data:
            vol_table_data = [['品种', 'IV', 'HV', 'IV百分位', 'HV百分位', 'IV-HV差']]
            
            for symbol, data in list(volatility_data.items())[:15]:  # 最多显示15个
                vol_table_data.append([
                    symbol,
                    f"{data.get('iv', 0):.2%}",
                    f"{data.get('hv', 0):.2%}",
                    f"{data.get('iv_rank', 0):.1f}%",
                    f"{data.get('hv_rank', 0):.1f}%",
                    f"{(data.get('iv', 0) - data.get('hv', 0)):.2%}"
                ])
            
            vol_table = Table(vol_table_data, colWidths=[2.5*cm, 2*cm, 2*cm, 2.5*cm, 2.5*cm, 2.5*cm])
            vol_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), font_name),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(vol_table)
        
        # 生成PDF
        doc.build(story)
    
    def _generate_markdown_report(self, filepath: str, signals: List, 
                                  volatility_data: Dict, market_summary: Dict):
        """生成Markdown格式报告"""
        timestamp = datetime.now()
        
        content = f"""# 波动率突破策略日报

**生成时间**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、市场概况

- **监控品种数**: {market_summary.get('total_symbols', 0)}
- **有效信号数**: {market_summary.get('total_signals', 0)}
- **做多波动率信号**: {market_summary.get('long_vol_count', 0)}
- **做空波动率信号**: {market_summary.get('short_vol_count', 0)}
- **平均IV百分位**: {market_summary.get('avg_iv_rank', 0):.1f}%
- **平均HV百分位**: {market_summary.get('avg_hv_rank', 0):.1f}%

---

## 二、交易信号

"""
        
        if signals:
            content += "| 品种 | 信号类型 | 方向 | 强度 | 入场价 | 止损价 | 止盈价 | 理由 |\n"
            content += "|------|----------|------|------|--------|--------|--------|------|\n"
            
            for signal in signals:
                content += f"| {signal.symbol} | {signal.signal_type} | {signal.direction} | "
                content += f"{signal.strength:.2f} | {signal.entry_price:.2f} | "
                content += f"{signal.stop_loss:.2f} | {signal.take_profit:.2f} | {signal.reason} |\n"
        else:
            content += "当前无有效交易信号\n"
        
        content += "\n---\n\n## 三、波动率数据\n\n"
        
        if volatility_data:
            content += "| 品种 | IV | HV | IV百分位 | HV百分位 | IV-HV差 |\n"
            content += "|------|----|----|----------|----------|--------|\n"
            
            for symbol, data in volatility_data.items():
                content += f"| {symbol} | {data.get('iv', 0):.2%} | {data.get('hv', 0):.2%} | "
                content += f"{data.get('iv_rank', 0):.1f}% | {data.get('hv_rank', 0):.1f}% | "
                content += f"{(data.get('iv', 0) - data.get('hv', 0)):.2%} |\n"
        
        content += f"\n---\n\n**报告生成器版本**: v1.0.0\n"
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)


if __name__ == "__main__":
    # 测试报告生成器
    from signal_generator import TradingSignal
    
    test_signals = [
        TradingSignal(
            symbol="sc2604",
            signal_type="LONG_VOL",
            direction="BUY",
            strength=0.85,
            entry_price=550.0,
            stop_loss=530.0,
            take_profit=590.0,
            iv_rank=25,
            hv_rank=45,
            iv_hv_diff=0.02,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            reason="IV处于低位(25.0%)，HV稳定，适合做多波动率"
        )
    ]
    
    test_vol_data = {
        "sc2604": {"iv": 0.28, "hv": 0.26, "iv_rank": 25, "hv_rank": 45},
        "bu2606": {"iv": 0.35, "hv": 0.28, "iv_rank": 75, "hv_rank": 50}
    }
    
    test_summary = {
        "total_symbols": 2,
        "total_signals": 1,
        "long_vol_count": 1,
        "short_vol_count": 0,
        "avg_iv_rank": 50,
        "avg_hv_rank": 47.5
    }
    
    generator = ReportGenerator("test_reports")
    filepath = generator.generate_daily_report(test_signals, test_vol_data, test_summary)
    print(f"报告已生成: {filepath}")
