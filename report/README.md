# Report Package

This folder is self-contained for compiling the HW2 report.

Files included:

- `report.tex`: main LaTeX source
- `references.bib`: bibliography
- `Formatting_Instructions_For_NeurIPS_2026/neurips_2026.sty`: NeurIPS style file
- `figures/`: all figures used by the report
- `extra_figures/`: additional experiment curves copied from `outputs/*/accuracy_loss_curve.png`
- `tables/`: all LaTeX tables used by the report
- `summary_results.csv`: summarized experiment results for reference

Compile from this folder:

```bash
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

Or, if `latexmk` is installed:

```bash
latexmk -pdf report.tex
```
