"""Create accessible Excel workbooks for browser table downloads."""

from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

EXCEL_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MAX_EXPORT_ROWS = 100_000
MAX_EXPORT_COLUMNS = 500

_INVALID_SHEET_CHARACTERS = re.compile(r"[\\/*?:\[\]]")
_INVALID_FILENAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._ -]+")
_INVALID_TABLE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9_]+")


def _clean_text(value: Any) -> str:
    """Return a normalized string suitable for workbook metadata or cells."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\x00", " ")).strip()


def safe_excel_filename(value: Any) -> str:
    """Return a conservative .xlsx download filename."""
    text = _INVALID_FILENAME_CHARACTERS.sub("", _clean_text(value) or "IVIVC_Data").strip()
    text = re.sub(r"\s+", "_", text).strip("._")[:80] or "IVIVC_Data"
    if not text.lower().endswith(".xlsx"):
        text += ".xlsx"
    return text


def safe_worksheet_title(value: Any) -> str:
    """Return a valid, meaningful Excel worksheet name."""
    text = _INVALID_SHEET_CHARACTERS.sub(" ", _clean_text(value) or "IVIVC Data")
    text = re.sub(r"\s+", " ", text).strip(" '")[:31]
    return text or "IVIVC Data"


def _unique_headers(headers: Iterable[Any]) -> list[str]:
    """Create non-empty, unique column headings required by Excel tables."""
    used: dict[str, int] = {}
    result: list[str] = []
    for index, value in enumerate(headers, start=1):
        base = _clean_text(value) or f"Column {index}"
        count = used.get(base.casefold(), 0) + 1
        used[base.casefold()] = count
        result.append(base if count == 1 else f"{base} ({count})")
    return result


def _cell_value(value: Any) -> Any:
    """Normalize exported values while preventing formula interpretation."""
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value

    text = _clean_text(value)
    if not text:
        return None

    # DataTables returns display text. Convert unambiguous numbers so Excel can
    # sort and calculate with them, but preserve identifiers with leading zeroes.
    if re.fullmatch(r"[-+]?\d+", text) and not re.fullmatch(r"[-+]?0\d+", text):
        try:
            return int(text)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][-+]?\d+)?", text):
        try:
            return float(text)
        except ValueError:
            pass

    # Avoid treating text beginning with formula-control characters as a formula.
    if text[0] in ("=", "+", "-", "@"):
        return "'" + text
    return text


def _table_name(value: Any) -> str:
    base = _INVALID_TABLE_NAME_CHARACTERS.sub("_", _clean_text(value) or "Data")
    base = base.strip("_") or "Data"
    # A fixed prefix avoids names that resemble cell references or conflict
    # with Excel's reserved naming rules.
    return ("IVIVC_" + base)[:200]


def _validate_payload(payload: dict[str, Any]) -> tuple[str, str, list[str], list[list[Any]]]:
    if not isinstance(payload, dict):
        raise ValueError("Excel export request must be a JSON object.")

    title = _clean_text(payload.get("title")) or "IVIVC data table"
    filename = safe_excel_filename(payload.get("filename") or title)
    headers_raw = payload.get("headers")
    rows_raw = payload.get("rows")

    if not isinstance(headers_raw, list) or not headers_raw:
        raise ValueError("Excel export requires at least one column heading.")
    if len(headers_raw) > MAX_EXPORT_COLUMNS:
        raise ValueError(f"Excel export is limited to {MAX_EXPORT_COLUMNS} columns.")
    if not isinstance(rows_raw, list):
        raise ValueError("Excel export rows must be provided as a list.")
    if len(rows_raw) > MAX_EXPORT_ROWS:
        raise ValueError(f"Excel export is limited to {MAX_EXPORT_ROWS:,} rows.")

    headers = _unique_headers(headers_raw)
    rows: list[list[Any]] = []
    for row in rows_raw:
        if not isinstance(row, list):
            raise ValueError("Each Excel export row must be a list of cell values.")
        normalized = [_cell_value(value) for value in row[: len(headers)]]
        normalized.extend([None] * (len(headers) - len(normalized)))
        rows.append(normalized)

    return title, filename, headers, rows


def build_accessible_excel(payload: dict[str, Any]) -> tuple[io.BytesIO, str]:
    """Build a formatted, metadata-rich Excel workbook from exported table data."""
    title, filename, headers, rows = _validate_payload(payload)

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = safe_worksheet_title(payload.get("sheet_name") or title)

    properties = workbook.properties
    properties.title = title
    properties.subject = "IVIVC App data table export"
    properties.creator = "IVIVC App for Absorbable Polymers"
    properties.lastModifiedBy = "IVIVC App for Absorbable Polymers"
    properties.description = (
        f"Accessible table export from the IVIVC App. Worksheet '{worksheet.title}' "
        "contains one header row followed by the exported data rows."
    )
    properties.keywords = "IVIVC, absorbable polymers, data export, accessible spreadsheet"
    properties.category = "Data export"
    properties.language = "en-US"
    properties.created = datetime.now(timezone.utc).replace(tzinfo=None)
    properties.modified = properties.created

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    body_alignment = Alignment(vertical="top", wrap_text=True)
    thin_gray = Side(style="thin", color="B7B7B7")
    cell_border = Border(left=thin_gray, right=thin_gray, top=thin_gray, bottom=thin_gray)

    for column_index, header in enumerate(headers, start=1):
        cell = worksheet.cell(row=1, column=column_index, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = header_alignment
        cell.border = cell_border

    for row_index, row in enumerate(rows, start=2):
        for column_index, value in enumerate(row, start=1):
            cell = worksheet.cell(row=row_index, column=column_index, value=value)
            cell.alignment = body_alignment
            cell.border = cell_border

    worksheet.freeze_panes = "A2"
    worksheet.print_title_rows = "1:1"
    worksheet.sheet_view.showGridLines = True
    worksheet.sheet_view.selection[0].activeCell = "A1"
    worksheet.sheet_view.selection[0].sqref = "A1"

    if rows:
        table_ref = f"A1:{get_column_letter(len(headers))}{len(rows) + 1}"
        table = Table(displayName=_table_name(title), ref=table_ref)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        worksheet.add_table(table)
    else:
        # A populated Excel table provides its own AutoFilter. Do not also add
        # a worksheet-level AutoFilter over the same cells: overlapping filter
        # definitions can cause Excel to repair and remove the table. For an
        # empty export there is no table, so a header-only worksheet filter is
        # safe and still exposes filter controls if data are added manually.
        worksheet.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"

    for column_index, header in enumerate(headers, start=1):
        values = [header]
        values.extend(row[column_index - 1] for row in rows[:1000])
        max_length = max(len(_clean_text(value)) for value in values)
        worksheet.column_dimensions[get_column_letter(column_index)].width = min(max(max_length + 2, 12), 60)

    worksheet.row_dimensions[1].height = 30
    worksheet.sheet_properties.pageSetUpPr.fitToPage = True
    worksheet.page_setup.fitToWidth = 1
    worksheet.page_setup.fitToHeight = 0
    worksheet.page_margins.left = 0.25
    worksheet.page_margins.right = 0.25
    worksheet.page_margins.top = 0.5
    worksheet.page_margins.bottom = 0.5

    output = io.BytesIO()
    workbook.save(output)
    output.seek(0)
    return output, filename
