#!/usr/bin/env bash

python3 utilities/generate_user_guide.py

pandoc docs/user_guide.md \
  -o docs/user_guide.pdf \
  --pdf-engine=xelatex \
  --toc \
  -V documentclass=article \
  -V papersize=letter \
  -V fontsize=10pt \
  -V geometry:margin=0.75in \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue
