"""
将README.md转换为PDF - 完美支持中文
使用markdown2 + reportlab + 中文字体
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from pathlib import Path
import re

def register_chinese_fonts():
    """注册中文字体"""
    try:
        # 尝试注册系统中文字体
        # macOS 系统字体路径
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',  # PingFang SC
            '/System/Library/Fonts/STHeiti Light.ttc',  # 黑体
            '/Library/Fonts/Arial Unicode.ttf',  # Arial Unicode
        ]

        for font_path in font_paths:
            if Path(font_path).exists():
                try:
                    pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                    print(f"✓ 成功注册字体: {font_path}")
                    return True
                except:
                    continue

        print("⚠ 未找到中文字体，使用默认字体")
        return False
    except Exception as e:
        print(f"⚠ 字体注册失败: {e}")
        return False

def create_styles(has_chinese_font):
    """创建样式"""
    styles = getSampleStyleSheet()

    font_name = 'ChineseFont' if has_chinese_font else 'Helvetica'

    # 标题样式
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=20,
        spaceBefore=10,
        leading=30
    )

    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=18,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=15,
        spaceBefore=15,
        leading=24
    )

    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontName=font_name,
        fontSize=14,
        textColor=colors.HexColor('#555555'),
        spaceAfter=10,
        spaceBefore=10,
        leading=18
    )

    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        leading=14,
        alignment=TA_LEFT
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8,
        leftIndent=20,
        backColor=colors.HexColor('#f8f8f8'),
        leading=10
    )

    return {
        'title': title_style,
        'h2': heading2_style,
        'h3': heading3_style,
        'normal': normal_style,
        'code': code_style
    }

def parse_markdown_to_pdf(md_file, pdf_file):
    """将Markdown文件转换为PDF"""

    print("=" * 60)
    print("开始生成PDF文档")
    print("=" * 60)

    # 注册中文字体
    has_chinese_font = register_chinese_fonts()

    # 读取Markdown内容
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 创建PDF文档
    doc = SimpleDocTemplate(
        str(pdf_file),
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    # 获取样式
    styles = create_styles(has_chinese_font)
    story = []

    # 解析Markdown内容
    lines = content.split('\n')
    i = 0
    in_code_block = False
    code_lines = []
    in_table = False
    table_data = []

    print(f"\n正在解析 {len(lines)} 行内容...")

    while i < len(lines):
        line = lines[i]

        # 代码块处理
        if line.strip().startswith('```'):
            if in_code_block:
                # 结束代码块
                if code_lines:
                    code_text = '\n'.join(code_lines)
                    # 限制代码块长度
                    if len(code_text) > 500:
                        code_text = code_text[:500] + '\n...'
                    story.append(Preformatted(code_text, styles['code']))
                    story.append(Spacer(1, 0.3*cm))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        line_stripped = line.strip()

        # 表格处理
        if '|' in line_stripped and line_stripped.startswith('|'):
            if not in_table:
                in_table = True
                table_data = []

            # 跳过分隔行
            if re.match(r'\|[\s\-:]+\|', line_stripped):
                i += 1
                continue

            # 解析表格行
            cells = [cell.strip() for cell in line_stripped.split('|')[1:-1]]
            # 清理markdown格式
            cells = [re.sub(r'\*\*(.*?)\*\*', r'\1', cell) for cell in cells]
            cells = [re.sub(r'`(.*?)`', r'\1', cell) for cell in cells]
            table_data.append(cells)
            i += 1

            # 检查下一行
            if i < len(lines) and '|' not in lines[i]:
                # 表格结束
                if table_data and len(table_data) > 1:
                    try:
                        t = Table(table_data, repeatRows=1)
                        t.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 9),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                            ('FONTSIZE', (0, 1), (-1, -1), 8),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ]))
                        story.append(t)
                        story.append(Spacer(1, 0.4*cm))
                    except Exception as e:
                        print(f"⚠ 表格渲染失败: {e}")
                in_table = False
                table_data = []
            continue

        # 标题处理
        if line_stripped.startswith('# '):
            text = line_stripped[2:]
            story.append(Paragraph(text, styles['title']))
        elif line_stripped.startswith('## '):
            text = line_stripped[3:]
            story.append(Paragraph(text, styles['h2']))
        elif line_stripped.startswith('### '):
            text = line_stripped[4:]
            story.append(Paragraph(text, styles['h3']))
        # 空行
        elif line_stripped == '':
            story.append(Spacer(1, 0.2*cm))
        # 分隔线
        elif line_stripped.startswith('---'):
            story.append(Spacer(1, 0.3*cm))
        # 普通文本
        else:
            if line_stripped:
                # 清理markdown格式
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line_stripped)
                text = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', text)
                try:
                    story.append(Paragraph(text, styles['normal']))
                except Exception as e:
                    # 如果段落渲染失败，尝试纯文本
                    print(f"⚠ 段落渲染失败，使用纯文本: {e}")

        i += 1

    # 生成PDF
    print(f"\n正在生成PDF文件...")
    try:
        doc.build(story)
        print(f"\n{'=' * 60}")
        print(f"✓ PDF生成成功!")
        print(f"  文件路径: {pdf_file}")
        print(f"  页数: 约 {len(story) // 30} 页")
        print(f"{'=' * 60}\n")
        return True
    except Exception as e:
        print(f"\n❌ PDF生成失败: {e}")
        return False

if __name__ == "__main__":
    project_root = Path(__file__).parent
    md_file = project_root / "README.md"
    pdf_file = project_root / "README.pdf"

    parse_markdown_to_pdf(md_file, pdf_file)
