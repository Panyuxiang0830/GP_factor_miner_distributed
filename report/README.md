# Project report

The English and Chinese reports contain the same completed experiments and conclusions.

| Language | PDF | LaTeX source |
| --- | --- | --- |
| English | [main.pdf](main.pdf) | [main.tex](main.tex) |
| Chinese | [main_zh.pdf](main_zh.pdf) | [main_zh.tex](main_zh.tex) |

Figures are included under `figures/`, including the PCA figures in `figures/pca/`. Both manuscripts compile from a clean repository checkout; they do not require private run directories or market data.

Use XeLaTeX with TeX Live, including ctex/Fandol for Chinese and the standard math, graphics, listings and TikZ packages. Run from `report/`:

```bash
xelatex -halt-on-error -interaction=nonstopmode main.tex
xelatex -halt-on-error -interaction=nonstopmode main.tex
xelatex -halt-on-error -interaction=nonstopmode main_zh.tex
xelatex -halt-on-error -interaction=nonstopmode main_zh.tex
```

Run another pass if LaTeX requests it to stabilise references. The author and student-ID fields remain for the project team to complete.
