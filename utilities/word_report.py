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


def _read_braced_group(value, start_index):
    """Return the contents and end index for a LaTeX braced group."""
    if start_index >= len(value) or value[start_index] != '{':
        return '', start_index

    depth = 0
    group_start = start_index + 1
    for index in range(start_index, len(value)):
        char = value[index]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return value[group_start:index], index + 1
    return value[group_start:], len(value)


def _latex_to_word_runs(latex):
    """Convert the app's simple LaTeX model equations to Word runs.

    This intentionally supports only the subset produced by utilities.model_display:
    multiplication, fractions rendered as text, and superscript exponents. It avoids
    a heavyweight LaTeX-to-OMML dependency while preserving the common exponential
    and power-model formatting in editable Word text.
    """
    text = str(latex or '').strip()
    text = text.replace('\\left', '').replace('\\right', '')
    text = text.replace('\\cdot', ' · ')
    text = text.replace('\\,', ' ')
    text = re.sub(r'\\mathrm\{([^{}]+)\}', r'\1', text)

    # Render simple \frac{numerator}{denominator} constructs as editable text.
    while '\\frac{' in text:
        start = text.find('\\frac{')
        numerator, after_num = _read_braced_group(text, start + len('\\frac'))
        denominator, after_den = _read_braced_group(text, after_num)
        replacement = f'({numerator})/({denominator})'
        text = text[:start] + replacement + text[after_den:]

    runs = []
    normal_buffer = []
    i = 0

    def flush_normal():
        if normal_buffer:
            runs.append((''.join(normal_buffer), False))
            normal_buffer.clear()

    while i < len(text):
        char = text[i]
        if char == '^':
            flush_normal()
            if i + 1 < len(text) and text[i + 1] == '{':
                exponent, next_index = _read_braced_group(text, i + 1)
                runs.append((_latex_to_plain_text(exponent), True))
                i = next_index
            else:
                exponent = text[i + 1:i + 2]
                if exponent:
                    runs.append((_latex_to_plain_text(exponent), True))
                i += 2
            continue
        normal_buffer.append(char)
        i += 1

    flush_normal()
    cleaned_runs = []
    for run_text, superscript in runs:
        run_text = _latex_to_plain_text(run_text)
        run_text = re.sub(r'\s+', ' ', run_text)
        if run_text:
            cleaned_runs.append((run_text, superscript))
    return cleaned_runs or [(_clean_text(latex), False)]


def _latex_to_plain_text(value):
    """Convert remaining simple LaTeX commands to readable editable text."""
    value = str(value or '')
    value = value.replace('\\cdot', ' · ')
    value = value.replace('\\times', ' × ')
    value = value.replace('\\left', '').replace('\\right', '')
    value = value.replace('{', '').replace('}', '')
    value = value.replace('\\', '')
    return value.strip()


def _add_equation(document, latex, fallback_text=None):
    """Add a simple editable Word equation paragraph using superscript runs."""
    runs = _latex_to_word_runs(latex or fallback_text)
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for text, superscript in runs:
        run = paragraph.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(11)
        run.font.superscript = bool(superscript)

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
        elif item_type == 'equation':
            _add_equation(document, item.get('latex'), item.get('text'))
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
