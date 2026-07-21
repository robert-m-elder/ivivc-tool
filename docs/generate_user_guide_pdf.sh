#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 generate_user_guide.py

if ! command -v lualatex >/dev/null 2>&1; then
  echo "Error: LuaLaTeX is required to generate the tagged accessible user-guide PDF." >&2
  exit 1
fi

if ! python3 -c 'import pypdf' >/dev/null 2>&1; then
  echo "Error: pypdf is required for the post-build PDF tag repair." >&2
  echo "Install project dependencies with: python3 -m pip install -r requirements.txt" >&2
  exit 1
fi

if command -v bibtex >/dev/null 2>&1 && bibtex --version >/dev/null 2>&1; then
  BIBTEX_CMD=bibtex
elif command -v bibtex.original >/dev/null 2>&1; then
  BIBTEX_CMD=bibtex.original
else
  BIBTEX_CMD=""
fi

lualatex -interaction=nonstopmode -halt-on-error user_guide_pdf.tex
if [ -n "$BIBTEX_CMD" ]; then
  "$BIBTEX_CMD" user_guide_pdf || true
else
  echo "Warning: BibTeX command not found; citations may remain unresolved." >&2
fi
lualatex -interaction=nonstopmode -halt-on-error user_guide_pdf.tex
lualatex -interaction=nonstopmode -halt-on-error user_guide_pdf.tex

REPAIRED_PDF=user_guide_pdf_repaired.pdf
rm -f "$REPAIRED_PDF"
python3 repair_user_guide_pdf_tags.py user_guide_pdf.pdf "$REPAIRED_PDF"
mv "$REPAIRED_PDF" user_guide.pdf
rm -f user_guide_pdf.pdf

if command -v pdfinfo >/dev/null 2>&1; then
  if ! pdfinfo user_guide.pdf | grep -Eq '^Tagged:[[:space:]]+yes$'; then
    echo "Error: generated user_guide.pdf is not tagged." >&2
    exit 1
  fi
fi

rm -f user_guide_pdf.aux user_guide_pdf.bbl user_guide_pdf.blg \
      user_guide_pdf.log user_guide_pdf.out user_guide_pdf.toc \
      user_guide_pdf-mathml.html user_guide_pdf-luamml-mathml.html \
      user_guide_pdf_repaired.pdf
