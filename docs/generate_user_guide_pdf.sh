#!/usr/bin/env bash
set -euo pipefail

python generate_user_guide.py

if command -v bibtex >/dev/null 2>&1 && bibtex --version >/dev/null 2>&1; then
  BIBTEX_CMD=bibtex
elif command -v bibtex.original >/dev/null 2>&1; then
  BIBTEX_CMD=bibtex.original
else
  BIBTEX_CMD=""
fi

pdflatex -interaction=nonstopmode -halt-on-error user_guide_pdf.tex
if [ -n "$BIBTEX_CMD" ]; then
  "$BIBTEX_CMD" user_guide_pdf || true
else
  echo "Warning: BibTeX command not found; citations may remain unresolved." >&2
fi
pdflatex -interaction=nonstopmode -halt-on-error user_guide_pdf.tex
pdflatex -interaction=nonstopmode -halt-on-error user_guide_pdf.tex

mv user_guide_pdf.pdf user_guide.pdf

rm -f user_guide_pdf.aux user_guide_pdf.bbl user_guide_pdf.blg \
      user_guide_pdf.log user_guide_pdf.out user_guide_pdf.toc

