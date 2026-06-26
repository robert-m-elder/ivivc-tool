#!/usr/bin/env bash
set -euo pipefail

python3 utilities/generate_user_guide.py

(
  cd docs

  build_tex="_user_guide_build.tex"
  cat > "${build_tex}" <<'LATEX'
\documentclass[10pt,letterpaper]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{amsmath}
\usepackage{array}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage{longtable}
\usepackage{tabularx}
\usepackage{xcolor}
\usepackage{xurl}
\usepackage{hyperref}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
\setlist[itemize]{leftmargin=*}
\setlist[enumerate]{leftmargin=*}
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue, breaklinks=true}

\pagestyle{fancy}
\fancyhf{}
\lhead{IVIVC App User Guide}
\rhead{\thepage}
\fancypagestyle{plain}{\fancyhf{}\renewcommand{\headrulewidth}{0pt}}

\title{IVIVC App User Guide}
\author{}
\date{}

\begin{document}

\maketitle
\tableofcontents
\newpage

\input{user_guide.tex}

\newpage
\appendix
\input{appendices/interpreting_model_performance.tex}

\end{document}
LATEX

  pdflatex -interaction=nonstopmode -halt-on-error "${build_tex}"
  pdflatex -interaction=nonstopmode -halt-on-error "${build_tex}"
  mv _user_guide_build.pdf user_guide.pdf
  rm -f _user_guide_build.aux _user_guide_build.log _user_guide_build.out _user_guide_build.toc "${build_tex}"
)
