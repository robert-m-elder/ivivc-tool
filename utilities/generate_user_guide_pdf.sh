#!/usr/bin/env bash
set -euo pipefail

python3 utilities/generate_user_guide.py
(
  cd docs
  latexmk -xelatex -interaction=nonstopmode -halt-on-error user_guide.tex
  latexmk -c user_guide.tex >/dev/null
)
