"""Utilities for generating editable Word reports from browser-exported report content."""

import base64
import io
import re
from datetime import datetime

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

WORD_MIME_TYPE = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'


def safe_report_filename(value):
    """Return a conservative report filename for download."""
    value = str(value or 'IVIVC Report')
    value = re.sub(r'[^A-Za-z0-9._ -]+', '', value).strip()
    value = re.sub(r'\s+', '_', value)
    return value[:80] or 'IVIVC_Report'


def _clean_text(value):
    """Normalize browser-extracted text before writing to Word."""
    value = str(value or '')
    value = value.replace('\xa0', ' ')
    value = re.sub(r'\s+', ' ', value).strip()
    return value


def _add_document_styles(document):
    """Apply small, conservative style adjustments to the generated document."""
    styles = document.styles
    styles['Normal'].font.name = 'Arial'
    styles['Normal'].font.size = Pt(10)

    for style_name, size in [('Heading 1', 16), ('Heading 2', 13), ('Heading 3', 11)]:
        if style_name in styles:
            styles[style_name].font.name = 'Arial'
            styles[style_name].font.size = Pt(size)


def _add_table(document, rows):
    """Add a simple editable table to the document."""
    rows = rows or []
    rows = [[_clean_text(cell) for cell in row] for row in rows]
    rows = [row for row in rows if any(row)]
    if not rows:
        return

    max_cols = max(len(row) for row in rows)
    table = document.add_table(rows=len(rows), cols=max_cols)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for r_idx, row in enumerate(rows):
        for c_idx in range(max_cols):
            text = row[c_idx] if c_idx < len(row) else ''
            cell = table.cell(r_idx, c_idx)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
            paragraph = cell.paragraphs[0]
            paragraph.text = text
            if r_idx == 0:
                for run in paragraph.runs:
                    run.bold = True

    document.add_paragraph()


def _decode_image_data_url(data_url):
    """Decode a browser data URL for insertion into a Word document."""
    if not data_url or not isinstance(data_url, str):
        return None
    match = re.match(r'^data:image/(png|jpeg|jpg);base64,(.+)$', data_url, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    try:
        return io.BytesIO(base64.b64decode(match.group(2)))
    except Exception:
        return None


def _add_image(document, data_url, title=None):
    """Add a plot image exported by the browser."""
    image_stream = _decode_image_data_url(data_url)
    if image_stream is None:
        if title:
            document.add_paragraph(f'{title}: image was unavailable for export.')
        else:
            document.add_paragraph('Image was unavailable for export.')
        return

    if title:
        caption = document.add_paragraph()
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.add_run(_clean_text(title)).bold = True

    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    run.add_picture(image_stream, width=Inches(6.5))


def build_word_report(payload):
    """Build a DOCX report from structured content posted by the browser.

    The browser remains responsible for deciding which selected report is active
    and exporting Plotly figures to image data URLs. This function intentionally
    avoids any model selection or ranking logic.
    """
    payload = payload or {}
    report_label = _clean_text(payload.get('report_label')) or 'Selected IVIVC model'
    sections = payload.get('sections') or []

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)
    _add_document_styles(document)

    title = document.add_heading('IVIVC Model Report', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run(report_label).bold = True

    generated = document.add_paragraph()
    generated.alignment = WD_ALIGN_PARAGRAPH.CENTER
    generated.add_run(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')

    document.add_paragraph()

    for item in sections:
        item_type = item.get('type')
        if item_type == 'heading':
            text = _clean_text(item.get('text'))
            if text:
                level = int(item.get('level') or 1)
                level = max(1, min(level, 3))
                document.add_heading(text, level=level)
        elif item_type == 'paragraph':
            text = _clean_text(item.get('text'))
            if text:
                document.add_paragraph(text)
        elif item_type == 'bullet':
            text = _clean_text(item.get('text'))
            if text:
                document.add_paragraph(text, style='List Bullet')
        elif item_type == 'table':
            _add_table(document, item.get('rows'))
        elif item_type == 'image':
            _add_image(document, item.get('data_url'), item.get('title'))

    document_io = io.BytesIO()
    document.save(document_io)
    document_io.seek(0)
    return document_io
