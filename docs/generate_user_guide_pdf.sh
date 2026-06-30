#!/usr/bin/env bash
set -euo pipefail

python generate_user_guide.py

pdflatex -interaction=nonstopmode -halt-on-error user_guide_pdf.tex
bibtex user_guide_pdf || true
pdflatex -interaction=nonstopmode -halt-on-error user_guide_pdf.tex
pdflatex -interaction=nonstopmode -halt-on-error user_guide_pdf.tex

mv user_guide_pdf.pdf user_guide.pdf

rm -f user_guide_pdf.aux user_guide_pdf.bbl user_guide_pdf.blg \
      user_guide_pdf.log user_guide_pdf.out user_guide_pdf.toc

